"""voc_to_coco.py — convert PASCAL VOC 2012 detection GT to COCO format.

Reads ``Annotations/*.xml`` for the image stems given by an
``ImageSets/Main/<split>.txt`` split file and emits a COCO-format
``instances_*.json`` that the existing evaluator
(``evaluation/mobilenetv2/evaluate.py``) consumes without modification.

Fallback cascade for resolving the image-stem list (some VOC dumps strip
some or all detection split files):

    1. ``ImageSets/Main/<--split>.txt``       — the requested split
    2. ``ImageSets/Main/trainval.txt``        — combined split (typical fallback)
    3. Every ``*.xml`` file in ``Annotations/`` — last resort

Categories are restricted to the 20 VOC-overlapping COCO classes, using the
same ID assignment the evaluator expects (mirrors its ``COCO_TO_VOC`` table).

Usage::

    python voc_to_coco.py \\
        --voc-root /home/.../VOCdevkit/VOC2012 \\
        --split val \\
        --output dataset/voc2012_as_coco/instances_voc2012_val.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Final

# ---------------------------------------------------------------------------
# Class-name maps — mirror the evaluator's COCO_TO_VOC inverse direction.
# ---------------------------------------------------------------------------
VOC_NAME_TO_COCO_ID: Final[dict[str, int]] = {
    "aeroplane": 5,
    "bicycle": 2,
    "bird": 16,
    "boat": 9,
    "bottle": 44,
    "bus": 6,
    "car": 3,
    "cat": 17,
    "chair": 62,
    "cow": 21,
    "diningtable": 67,
    "dog": 18,
    "horse": 19,
    "motorbike": 4,
    "person": 1,
    "pottedplant": 64,
    "sheep": 20,
    "sofa": 63,
    "train": 7,
    "tvmonitor": 72,
}

COCO_ID_TO_NAME: Final[dict[int, str]] = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane",
    6: "bus", 7: "train", 9: "boat", 16: "bird", 17: "cat",
    18: "dog", 19: "horse", 20: "sheep", 21: "cow", 44: "bottle",
    62: "chair", 63: "couch", 64: "potted plant", 67: "dining table", 72: "tv",
}

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
_LOG = logging.getLogger("voc_to_coco")


# ---------------------------------------------------------------------------
# Helpers (pure or near-pure — easy to test in isolation).
# ---------------------------------------------------------------------------
def _read_split_file(path: Path) -> list[str]:
    """Return the non-blank image stems listed in a VOC split file."""
    return [s for s in (line.strip() for line in path.read_text().splitlines()) if s]


def _resolve_stems(voc_root: Path, split: str, annotations_dir: Path) -> list[str]:
    """Resolve the image-stem list with a documented fallback cascade.

    Tries, in order:
      1. ``ImageSets/Main/<split>.txt``       (the requested split)
      2. ``ImageSets/Main/trainval.txt``      (combined split)
      3. Every ``*.xml`` in ``Annotations/``  (last resort)

    The first hit wins. Logs the source so the caller can see whether the
    eval set is canonical or derived.
    """
    main_dir = voc_root / "ImageSets" / "Main"
    primary = main_dir / f"{split}.txt"
    fallback = main_dir / "trainval.txt"

    if primary.is_file():
        stems = _read_split_file(primary)
        _LOG.info("%d stems from %s", len(stems), primary)
        return stems

    _LOG.warning("split file not found: %s", primary)

    if fallback != primary and fallback.is_file():
        stems = _read_split_file(fallback)
        _LOG.info("falling back to trainval split: %d stems from %s", len(stems), fallback)
        return stems

    _LOG.warning("no split files available, deriving from %s", annotations_dir)
    stems = sorted(p.stem for p in annotations_dir.glob("*.xml"))
    _LOG.info("derived %d stems from annotations/", len(stems))
    return stems


def _parse_xml(xml_path: Path) -> tuple[int, int, list[tuple[str, list[int], int]]]:
    """Parse one VOC XML.

    Returns ``(width, height, [(name, [xmin, ymin, xmax, ymax], difficult), ...])``.
    Objects missing either a ``<name>`` or a ``<bndbox>`` element are silently
    dropped — they have no usable detection content.
    """
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    if size is None:
        raise ValueError(f"{xml_path}: missing <size>")
    width = int(size.findtext("width"))
    height = int(size.findtext("height"))

    objects: list[tuple[str, list[int], int]] = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        bb = obj.find("bndbox")
        if name is None or bb is None:
            continue
        difficult = int(obj.findtext("difficult") or "0")
        box = [
            int(float(bb.findtext("xmin"))),
            int(float(bb.findtext("ymin"))),
            int(float(bb.findtext("xmax"))),
            int(float(bb.findtext("ymax"))),
        ]
        objects.append((name, box, difficult))
    return width, height, objects


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------
def convert(voc_root: Path, split: str, output: Path, include_difficult: bool) -> None:
    """Convert VOC GT for ``split`` to a COCO-format JSON at ``output``."""
    annotations_dir = voc_root / "Annotations"
    images_dir = voc_root / "JPEGImages"
    if not annotations_dir.is_dir():
        raise FileNotFoundError(annotations_dir)
    if not images_dir.is_dir():
        _LOG.warning(
            "JPEGImages/ not found at %s — evaluator will fail when reading images",
            images_dir,
        )

    stems = _resolve_stems(voc_root, split, annotations_dir)
    if not stems:
        raise RuntimeError(f"no image stems resolved for split={split!r} under {voc_root}")

    images_out: list[dict] = []
    annotations_out: list[dict] = []
    ann_id = 1
    missing_xml = skipped_unknown = skipped_difficult = 0

    for image_id, stem in enumerate(stems, start=1):
        xml_path = annotations_dir / f"{stem}.xml"
        if not xml_path.is_file():
            missing_xml += 1
            continue

        width, height, objs = _parse_xml(xml_path)
        images_out.append({
            "id": image_id,
            "file_name": f"{stem}.jpg",
            "width": width,
            "height": height,
        })

        for name, (xmin, ymin, xmax, ymax), difficult in objs:
            if difficult and not include_difficult:
                skipped_difficult += 1
                continue
            coco_id = VOC_NAME_TO_COCO_ID.get(name)
            if coco_id is None:
                skipped_unknown += 1
                continue
            w, h = xmax - xmin, ymax - ymin
            annotations_out.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": coco_id,
                "bbox": [xmin, ymin, w, h],
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
    if missing_xml:
        _LOG.warning("  missing XML for %d image stems", missing_xml)
    if skipped_unknown:
        _LOG.warning("  skipped %d objects with non-VOC class names", skipped_unknown)
    if skipped_difficult:
        _LOG.info(
            "  excluded %d difficult objects (use --include-difficult to keep)",
            skipped_difficult,
        )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--voc-root", type=Path, required=True,
                   help="Path to VOCdevkit/VOC2012")
    p.add_argument("--split", default="val",
                   help="Split name in ImageSets/Main/<split>.txt; "
                        "cascades to trainval.txt then all-annotations on miss (default: val)")
    p.add_argument("--output", type=Path, required=True,
                   help="Output COCO-format JSON path")
    p.add_argument("--include-difficult", action="store_true",
                   help="Include objects with difficult=1 (default: exclude, standard VOC eval)")
    args = p.parse_args(argv)
    convert(args.voc_root, args.split, args.output, args.include_difficult)
    return 0


if __name__ == "__main__":
    sys.exit(main())
