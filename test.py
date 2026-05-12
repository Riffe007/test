"""seg_to_coco.py — derive COCO-format detection GT from VOC 2012 segmentation masks.

For VOC dumps that ship segmentation labels but not detection annotations
(``SegmentationClass/`` + ``SegmentationObject/`` PNGs present, ``Annotations/``
stripped), this script derives per-instance bounding boxes from the
segmentation masks and emits a COCO-format ``instances_*.json`` that the
existing evaluator (``evaluation/mobilenetv2/evaluate.py``) consumes without
modification.

The output shape, category-id mapping, and bbox arithmetic match
``voc_to_coco.py`` exactly — the downstream evaluator can't tell which adapter
produced the GT.

Algorithm
---------
For each image with both ``SegmentationObject/<stem>.png`` and
``SegmentationClass/<stem>.png``:

1. Read both PNGs as palette-index ``uint8`` arrays (VOC palette stores class
   indices, not RGB, in palette mode).
2. For each unique instance id in ``SegmentationObject`` (excluding
   background=0 and void=255):

   * Form ``mask = (seg_object == instance_id)``.
   * Class label = mode of ``seg_class[mask]`` over valid (non-bg, non-void)
     pixels — majority vote tolerates a few stray boundary pixels.
   * Bounding box = tight rectangle around the mask via ``np.where``.
   * Map VOC seg class index → COCO category id (matches voc_to_coco.py).
   * Emit a COCO annotation record.

3. Emit one COCO image record per image with at least one valid instance,
   using mask dimensions as the image dimensions (VOC seg masks share their
   source JPEG's resolution).

Usage::

    python seg_to_coco.py \\
        --voc-root /home/.../VOCdevkit/VOC2012 \\
        --output dataset/voc2012_as_coco/instances_voc2012_seg.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Final

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Constants — VOC segmentation palette and the inverse-VOC→COCO category map.
# ---------------------------------------------------------------------------
#
# The VOC segmentation palette assigns class indices 1–20 alphabetically by
# class name after background. The mapping below mirrors voc_to_coco.py's
# VOC_NAME_TO_COCO_ID, just keyed by palette index instead of class name so
# we can translate mask pixels directly to COCO category ids.
VOC_SEG_INDEX_TO_COCO_ID: Final[dict[int, int]] = {
    1:  5,   # aeroplane    → airplane
    2:  2,   # bicycle      → bicycle
    3:  16,  # bird         → bird
    4:  9,   # boat         → boat
    5:  44,  # bottle       → bottle
    6:  6,   # bus          → bus
    7:  3,   # car          → car
    8:  17,  # cat          → cat
    9:  62,  # chair        → chair
    10: 21,  # cow          → cow
    11: 67,  # diningtable  → dining table
    12: 18,  # dog          → dog
    13: 19,  # horse        → horse
    14: 4,   # motorbike    → motorcycle
    15: 1,   # person       → person
    16: 64,  # pottedplant  → potted plant
    17: 20,  # sheep        → sheep
    18: 63,  # sofa         → couch
    19: 7,   # train        → train
    20: 72,  # tvmonitor    → tv
}

COCO_ID_TO_NAME: Final[dict[int, str]] = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane",
    6: "bus", 7: "train", 9: "boat", 16: "bird", 17: "cat",
    18: "dog", 19: "horse", 20: "sheep", 21: "cow", 44: "bottle",
    62: "chair", 63: "couch", 64: "potted plant", 67: "dining table", 72: "tv",
}

VOC_BACKGROUND_LABEL: Final[int] = 0
VOC_VOID_LABEL: Final[int] = 255

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
_LOG = logging.getLogger("seg_to_coco")


# ---------------------------------------------------------------------------
# Pure helpers — easy to unit-test in isolation.
# ---------------------------------------------------------------------------
def _read_palette_png(path: Path) -> np.ndarray:
    """Load a VOC-style palette PNG as a ``uint8`` index array.

    VOC segmentation PNGs are saved in mode ``"P"`` (palette indexed). Loading
    via PIL preserves the palette indices directly, which are the class /
    instance labels.
    """
    with Image.open(path) as im:
        if im.mode != "P":
            _LOG.warning("%s: expected palette ('P') mode, got %s", path, im.mode)
        return np.array(im, dtype=np.uint8)


def _instance_class_index(seg_class: np.ndarray, mask: np.ndarray) -> int | None:
    """Resolve a single VOC class index for an instance by majority vote.

    Boundary pixels can carry the void label, so we filter to valid VOC class
    indices (1..20) before counting. Returns ``None`` if no valid pixels exist
    in the instance — caller should skip the instance.
    """
    pixels = seg_class[mask]
    valid = pixels[
        (pixels != VOC_BACKGROUND_LABEL) & (pixels != VOC_VOID_LABEL)
    ]
    if valid.size == 0:
        return None
    return int(np.bincount(valid).argmax())


def _instances_from_pair(
    seg_class: np.ndarray, seg_object: np.ndarray
) -> list[tuple[int, list[int]]]:
    """Extract ``(coco_category_id, [x, y, w, h])`` per labeled instance.

    Skips background, void, and any instance whose class can't be mapped to
    a COCO category id. Bbox arithmetic matches ``voc_to_coco.py``:
    ``w = xmax - xmin``, ``h = ymax - ymin``.
    """
    instances: list[tuple[int, list[int]]] = []
    for inst_id in np.unique(seg_object):
        if inst_id == VOC_BACKGROUND_LABEL or inst_id == VOC_VOID_LABEL:
            continue

        mask = seg_object == inst_id
        if not mask.any():
            continue

        voc_class_idx = _instance_class_index(seg_class, mask)
        if voc_class_idx is None:
            continue

        coco_id = VOC_SEG_INDEX_TO_COCO_ID.get(voc_class_idx)
        if coco_id is None:
            continue

        ys, xs = np.where(mask)
        xmin, ymin = int(xs.min()), int(ys.min())
        xmax, ymax = int(xs.max()), int(ys.max())
        w, h = xmax - xmin, ymax - ymin
        if w <= 0 or h <= 0:
            # Degenerate single-pixel-line instances — not useful for detection eval.
            continue
        instances.append((coco_id, [xmin, ymin, w, h]))
    return instances


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------
def convert(voc_root: Path, output: Path) -> None:
    """Read every available (SegmentationClass, SegmentationObject) pair and
    emit a COCO-format JSON of derived detection ground truth at ``output``.
    """
    seg_class_dir = voc_root / "SegmentationClass"
    seg_object_dir = voc_root / "SegmentationObject"
    for d in (seg_class_dir, seg_object_dir):
        if not d.is_dir():
            raise FileNotFoundError(d)

    object_pngs = sorted(seg_object_dir.glob("*.png"))
    if not object_pngs:
        raise RuntimeError(f"no PNG masks found in {seg_object_dir}")
    _LOG.info("found %d segmentation pairs under %s", len(object_pngs), voc_root)

    images_out: list[dict] = []
    annotations_out: list[dict] = []
    ann_id = 1
    missing_class = shape_mismatch = empty_instances = 0

    for image_id, object_path in enumerate(object_pngs, start=1):
        stem = object_path.stem
        class_path = seg_class_dir / f"{stem}.png"
        if not class_path.is_file():
            missing_class += 1
            continue

        seg_object = _read_palette_png(object_path)
        seg_class = _read_palette_png(class_path)
        if seg_object.shape != seg_class.shape:
            _LOG.warning("%s: class/object shape mismatch, skipping", stem)
            shape_mismatch += 1
            continue

        instances = _instances_from_pair(seg_class, seg_object)
        if not instances:
            empty_instances += 1
            continue

        height, width = seg_object.shape
        images_out.append({
            "id": image_id,
            "file_name": f"{stem}.jpg",
            "width": int(width),
            "height": int(height),
        })

        for coco_id, bbox in instances:
            _, _, w, h = bbox
            annotations_out.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": coco_id,
                "bbox": bbox,
                "area": float(w * h),
                "iscrowd": 0,
            })
            ann_id += 1

    categories_out = [
        {"id": cid, "name": COCO_ID_TO_NAME[cid]} for cid in sorted(COCO_ID_TO_NAME)
    ]

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(
        {"images": images_out, "annotations": annotations_out, "categories": categories_out},
        indent=2,
    ))

    _LOG.info("wrote %s", output)
    _LOG.info("  images:      %d", len(images_out))
    _LOG.info("  annotations: %d", len(annotations_out))
    _LOG.info("  categories:  %d", len(categories_out))
    if missing_class:
        _LOG.warning("  missing SegmentationClass for %d object masks", missing_class)
    if shape_mismatch:
        _LOG.warning("  skipped %d images with class/object shape mismatch", shape_mismatch)
    if empty_instances:
        _LOG.info("  skipped %d images with no valid instances", empty_instances)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--voc-root", type=Path, required=True,
                   help="Path to VOCdevkit/VOC2012 (must contain SegmentationClass/ and SegmentationObject/)")
    p.add_argument("--output", type=Path, required=True,
                   help="Output COCO-format JSON path")
    args = p.parse_args(argv)
    convert(args.voc_root, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
