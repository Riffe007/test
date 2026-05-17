"""End-to-end detection evaluation harness for MobileNetV2-SSD-Lite.

Evaluates and compares four model families on a COCO-format detection set:

    TFLite  |  PyTorch baseline  |  ExecuTorch fp32  |  ExecuTorch int8/int4

It runs each backend, scores predictions with pycocotools ``COCOeval``, writes a
consolidated JSON, and renders an HTML report.

------------------------------------------------------------------------------
STATUS OF THIS FILE — READ BEFORE USE
------------------------------------------------------------------------------
This is a from-specification rebuild. The original ``evaluate.py`` was only ever
available as screen photographs; this file reproduces its documented behaviour
and adds two changes:

  1. A TFLite inference backend (rendered leftmost in the report).
  2. A category-id remap fix in ``decode_ssd_predictions`` — the original emitted
     raw VOC class indices (1-20) as ``category_id``, which do not match the COCO
     category ids in the ground truth. That bug produced ~0.087 mAP for the
     PyTorch and ExecuTorch backends.

Verbatim-solid in this file: ``TFLiteSSDEvaluator``, ``VOC_INDEX_TO_COCO_ID``,
``TFLITE_COCO90_INDEX_TO_COCO_ID``, the remap, and the overall structure.

Reconstructed — reconcile against your real file before relying on it. Every
such spot is tagged ``# ── VERIFY ──``. The main ones are: the qfgaohao SSD
imports / model construction / prior loading, the ExecuTorch runtime loading
API, the exact config key names, and the report-module import path.

Revision — report-contract hardening
------------------------------------
This revision makes NO change to model construction, inference, decoding, or
metric computation. It hardens only the JSON contract consumed by reporting.py
so the HTML report populates reliably regardless of which reporter path runs:

  * every backend entry now also emits ``model_name``, ``backend``, and a
    derived ``timing`` block (avg time / total time / throughput);
  * ExecuTorch variants emit ``pte_size_mb`` so the PTE+PTD size breakdown
    renders;
  * top-level results carry ``task_type`` and ``num_samples``; the ``dataset``
    block carries ``name`` / ``num_samples`` / ``samples_evaluated``;
  * ``save_evaluation_results`` no longer uses ``default=str`` — a NumPy-aware
    encoder keeps metric values as real numbers (a stringified ``np.float32``
    is silently dropped by the reporter's numeric-type checks).

The ``# ── VERIFY ──`` items below are UNCHANGED and still require checking
against the real qfgaohao repo, the ExecuTorch runtime, and the config schema.

Revision 2 — graceful backend degradation
------------------------------------------
Fixes a crash: when the qfgaohao ``vision`` package could not be imported, the
PyTorch backend was caught and skipped, but the ExecuTorch backend re-ran the
same decoder load OUTSIDE a try/except and aborted the whole run. The
ExecuTorch shared-decoder load is now guarded and degrades like any other
optional backend. A new ``_ensure_qfgaohao_repo`` pre-check turns the opaque
``No module named 'vision'`` failure into an actionable message naming the
expected repo path. This does NOT resolve the underlying cause — the qfgaohao
``pytorch-ssd`` repo path must still be set correctly in the config.

Revision 3 — dataset-agnostic scoring (VOC and COCO)
----------------------------------------------------
Enables running the same VOC-trained model against either a VOC-format ground
truth or COCO val2017 and getting comparable numbers. No new loader was needed:
``VOCDetectionDataset`` already consumes a COCO-format JSON and resolves images
by ``file_name``, so it handles both datasets as-is -- the switch is purely a
config change (point ``data_path`` / ``gt_coco_json`` at the other dataset).

The one real fix is in scoring scope. ``calculate_detection_metrics`` already
accepted a ``class_subset``, but ``main`` never passed one, so COCOeval scored
over every category in the ground truth. On a 20-category VOC ground truth that
is correct; on COCO's 80-category ground truth it silently averages AP over 60
categories the VOC model can never predict, collapsing the reported mAP. The
PyTorch and ExecuTorch call sites now pass ``class_subset=MODEL_COCO_CATEGORY_IDS``
(the 20 ids the model can predict). This is unconditional and correct for both
datasets -- a no-op for VOC, essential for COCO. The TFLite call site is left
unrestricted because that baseline is a separate COCO-90 model.

Note: the per-image GT counts in ``generate_per_image_detection_results`` are
NOT category-restricted, so on a COCO run the per-image table will count all
ground-truth objects, including non-VOC categories. The headline mAP is correct;
only that auxiliary table over-counts.
------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# =============================================================================
# Logging
# =============================================================================


def _setup_logger() -> logging.Logger:
    """Configure and return the module logger."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("evaluate")


logger = _setup_logger()


# =============================================================================
# Constants
# =============================================================================

#: Metric keys produced by :func:`calculate_detection_metrics`.
_DETECTION_METRIC_KEYS: tuple[str, ...] = (
    "mean_precision",
    "mean_recall",
    "mean_f1",
    "mean_iou",
    "mAP_0.5",
    "mAP_0.65",
    "mAP_0.75",
    "mAP_0.5_0.95",
    "num_samples",
)

# -----------------------------------------------------------------------------
# Class-space mappings.
#
# Source of truth: executorch-toolkit/evaluation/mobilenetv2/
#                  compare_tflite_pytorch_coco.py
# Copied here (not imported) so this module does not inherit that file's
# import-time side effects (a module-level TensorFlow import and a
# logging.basicConfig call).
# -----------------------------------------------------------------------------

#: VOC class index (1-20, qfgaohao alphabetical convention) -> COCO category_id.
#: The qfgaohao SSD head emits channel indices 1..20 in VOC alphabetical order
#: (1=aeroplane ... 15=person ... 20=tvmonitor). The ground-truth JSON uses real
#: COCO category ids. This table projects predictions into COCO id space so the
#: PyTorch / ExecuTorch backends are scored against the same ground truth as the
#: COCO-native TFLite backend.
VOC_INDEX_TO_COCO_ID: dict[int, int] = {
    1: 5,    # aeroplane   -> airplane
    2: 2,    # bicycle
    3: 16,   # bird
    4: 9,    # boat
    5: 44,   # bottle
    6: 6,    # bus
    7: 3,    # car
    8: 17,   # cat
    9: 62,   # chair
    10: 21,  # cow
    11: 67,  # diningtable -> dining table
    12: 18,  # dog
    13: 19,  # horse
    14: 4,   # motorbike   -> motorcycle
    15: 1,   # person
    16: 64,  # pottedplant -> potted plant
    17: 20,  # sheep
    18: 63,  # sofa        -> couch
    19: 7,   # train
    20: 72,  # tvmonitor   -> tv
}

#: The COCO category ids the qfgaohao (VOC-trained) SSD can actually predict --
#: i.e. the image of VOC_INDEX_TO_COCO_ID. Detection mAP for the PyTorch and
#: ExecuTorch backends MUST be scored against exactly this category set, not the
#: full set present in the ground truth. On a VOC-format ground truth the two
#: are identical, so this is a no-op. On a COCO ground truth (instances_val2017,
#: 80 categories) it is essential: without it COCOeval averages AP over all 80
#: categories, the model scores 0 on the ~60 it cannot predict, and the reported
#: mAP collapses into a category-space artifact rather than a real result. With
#: it, a VOC run and a COCO run score the same 20 categories and are comparable.
MODEL_COCO_CATEGORY_IDS: tuple[int, ...] = tuple(
    sorted(set(VOC_INDEX_TO_COCO_ID.values()))
)

#: COCO 90-class TFLite output index (0-89) -> COCO category_id (1-90).
#: TF Object Detection API SSD MobileNet V2 checkpoints use a 90-class output
#: head mirroring COCO's gappy 1..90 id space; the correct mapping is identity+1.
TFLITE_COCO90_INDEX_TO_COCO_ID: tuple[int, ...] = tuple(range(1, 91))


# =============================================================================
# Path & configuration helpers
# =============================================================================


def _resolve_path(value: str | Path, base: Optional[Path] = None) -> Path:
    """Expand ``~`` and resolve ``value`` (optionally relative to ``base``)."""
    path = Path(str(value)).expanduser()
    if not path.is_absolute() and base is not None:
        path = (base / path)
    return path.resolve()


def _load_config(config_path: Path) -> dict[str, Any]:
    """Load and minimally validate the JSON evaluation config.

    # ── VERIFY ── required-section names against your real config schema.
    """
    with config_path.open("r", encoding="utf-8") as handle:
        config: dict[str, Any] = json.load(handle)

    for section in ("model", "evaluation"):
        if section not in config:
            raise KeyError(f"Config is missing required section: '{section}'")
    evaluation = config["evaluation"]
    for section in ("dataset", "decoder", "backends"):
        if section not in evaluation:
            raise KeyError(
                f"Config 'evaluation' is missing required block: '{section}'"
            )
    return config


def _resolve_image_path(data_path: Path, file_name: str) -> Optional[Path]:
    """Resolve a ground-truth ``file_name`` to an image on disk.

    Shared by :class:`VOCDetectionDataset` and :func:`evaluate_tflite_model` so
    both backends locate images identically. Tries ``data_path / file_name``
    first (the COCO ``val2017/`` layout), then ``data_path/JPEGImages``.
    """
    candidates = (
        data_path / file_name,
        data_path / "JPEGImages" / file_name,
        data_path / Path(file_name).name,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _file_size_mb(path: Path) -> Optional[float]:
    """Return the size of ``path`` in megabytes, or ``None`` if absent."""
    try:
        return round(path.stat().st_size / (1024 * 1024), 4)
    except OSError:
        return None


def _to_numpy(value: Any) -> np.ndarray:
    """Coerce a torch tensor or array-like into a NumPy array."""
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


# =============================================================================
# COCO detection metrics
# =============================================================================


def _load_coco_ground_truth(labels: Any) -> Any:
    """Build a pycocotools ``COCO`` ground-truth object from ``labels``.

    Accepts an already-constructed ``COCO`` object, a path to an
    ``instances_*.json`` file, or a parsed COCO-format dict.
    """
    from pycocotools.coco import COCO

    if isinstance(labels, COCO):
        return labels

    coco = COCO()
    if isinstance(labels, (str, Path)):
        with Path(labels).open("r", encoding="utf-8") as handle:
            coco.dataset = json.load(handle)
    elif isinstance(labels, dict):
        coco.dataset = labels
    else:
        raise TypeError(f"Unsupported ground-truth type: {type(labels)!r}")
    coco.createIndex()
    return coco


def calculate_detection_metrics(
    predictions: Sequence[dict[str, Any]],
    labels: Any,
    class_subset: Optional[Sequence[int]] = None,
) -> dict[str, Any]:
    """Score COCO-format ``predictions`` against ``labels`` with ``COCOeval``.

    Model-agnostic: every backend (TFLite / PyTorch / ExecuTorch) feeds its
    predictions through this single path.

    Args:
        predictions: flat list of COCO detection dicts, each
            ``{"image_id", "category_id", "bbox": [x, y, w, h], "score"}``.
        labels: COCO ground truth (``COCO`` object, JSON path, or parsed dict).
        class_subset: optional category ids to restrict scoring to. When
            ``None``, ``COCOeval`` scores over every category in the ground
            truth.

    Returns:
        A dict keyed by :data:`_DETECTION_METRIC_KEYS`.
    """
    from pycocotools.cocoeval import COCOeval

    coco_gt = _load_coco_ground_truth(labels)

    num_predictions = len(predictions)
    if num_predictions == 0:
        logger.warning("No predictions supplied; metrics will all be zero.")
        return {key: 0.0 for key in _DETECTION_METRIC_KEYS}

    try:
        coco_dt = coco_gt.loadRes(list(predictions))
    except Exception as exc:  # malformed predictions must not crash the run
        logger.error("COCO loadRes failed (%s); returning zero metrics.", exc)
        return {key: 0.0 for key in _DETECTION_METRIC_KEYS}

    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    if class_subset is not None:
        coco_eval.params.catIds = sorted(int(c) for c in class_subset)

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    metrics: dict[str, Any] = {
        "mAP_0.5_0.95": _safe_stat(coco_eval, 0),
        "mAP_0.5": _safe_stat(coco_eval, 1),
        "mAP_0.75": _safe_stat(coco_eval, 2),
        "mAP_0.65": _average_precision_at_iou(coco_eval, 0.65),
        "mean_precision": _mean_precision_at_iou(coco_eval, 0.5),
        "mean_recall": _mean_recall(coco_eval),
        "mean_iou": _mean_iou(coco_eval),
        "num_samples": num_predictions,
    }
    metrics["mean_f1"] = _harmonic_mean(
        metrics["mean_precision"], metrics["mean_recall"]
    )

    missing = [key for key in _DETECTION_METRIC_KEYS if key not in metrics]
    if missing:
        raise RuntimeError(f"Metric computation produced no value for: {missing}")
    return {key: metrics[key] for key in _DETECTION_METRIC_KEYS}


def _safe_stat(coco_eval: Any, index: int) -> float:
    """Return ``coco_eval.stats[index]`` clamped to a non-negative float."""
    try:
        value = float(coco_eval.stats[index])
    except (IndexError, TypeError, ValueError):
        return 0.0
    return value if value >= 0.0 else 0.0


def _average_precision_at_iou(coco_eval: Any, iou_threshold: float) -> float:
    """Mean average precision at a single IoU threshold (all areas, maxDet)."""
    params = coco_eval.params
    iou_index = np.argmin(np.abs(params.iouThrs - iou_threshold))
    precision = coco_eval.eval.get("precision")
    if precision is None:
        return 0.0
    # precision dims: [T, R, K, A, M]; area index 0 = 'all', maxDet index -1.
    values = precision[iou_index, :, :, 0, -1]
    values = values[values > -1]
    return float(values.mean()) if values.size else 0.0


def _mean_precision_at_iou(coco_eval: Any, iou_threshold: float) -> float:
    """Mean precision at a single IoU threshold (alias of AP@IoU)."""
    return _average_precision_at_iou(coco_eval, iou_threshold)


def _mean_recall(coco_eval: Any) -> float:
    """Mean recall across IoU thresholds (all areas, maxDet)."""
    recall = coco_eval.eval.get("recall")
    if recall is None:
        return 0.0
    # recall dims: [T, K, A, M].
    values = recall[:, :, 0, -1]
    values = values[values > -1]
    return float(values.mean()) if values.size else 0.0


def _mean_iou(coco_eval: Any) -> float:
    """Mean IoU of matched detections across the accumulated evaluation images.

    # ── VERIFY ── the original may compute this differently; this averages the
    # per-image IoU matrices that COCOeval caches in ``coco_eval.ious``.
    """
    matched: list[float] = []
    for iou_matrix in coco_eval.ious.values():
        array = np.asarray(iou_matrix, dtype=float)
        if array.size == 0:
            continue
        # Best IoU per ground-truth column.
        best = array.max(axis=0) if array.ndim == 2 else array.ravel()
        matched.extend(float(v) for v in best if v > 0.0)
    return float(np.mean(matched)) if matched else 0.0


def _harmonic_mean(precision: float, recall: float) -> float:
    """Harmonic mean (F1) of precision and recall."""
    if precision + recall <= 0.0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


# =============================================================================
# Per-image detection results
# =============================================================================


def generate_per_image_detection_results(
    predictions: Sequence[dict[str, Any]],
    image_metadata: Sequence[dict[str, Any]],
    coco_ground_truth: Any,
    latency_by_image: Optional[dict[int, float]] = None,
) -> list[dict[str, Any]]:
    """Build a per-image summary table.

    Each row records the image index, id, prediction count, ground-truth count,
    and (when available) inference latency.
    """
    coco_gt = _load_coco_ground_truth(coco_ground_truth)
    latency_by_image = latency_by_image or {}

    predictions_by_image: dict[int, int] = {}
    for prediction in predictions:
        image_id = int(prediction["image_id"])
        predictions_by_image[image_id] = predictions_by_image.get(image_id, 0) + 1

    rows: list[dict[str, Any]] = []
    for index, meta in enumerate(image_metadata):
        image_id = int(meta["image_id"])
        try:
            gt_count = len(coco_gt.getAnnIds(imgIds=[image_id]))
        except Exception:
            gt_count = 0
        row: dict[str, Any] = {
            "index": index,
            "image_id": image_id,
            "file_name": meta.get("file_name"),
            "num_predictions": predictions_by_image.get(image_id, 0),
            "num_ground_truth": gt_count,
        }
        if image_id in latency_by_image:
            row["latency_ms"] = round(latency_by_image[image_id], 3)
        rows.append(row)
    return rows


# =============================================================================
# Dataset
# =============================================================================


class VOCDetectionDataset(Dataset):
    """Detection dataset driven by a COCO-format ground-truth JSON.

    ``__getitem__`` yields a float32 CHW tensor scaled to ``[0, 1]`` together
    with image metadata. Channel normalization is intentionally deferred to the
    model-evaluation step (see :func:`_qfgaohao_normalize`).
    """

    def __init__(
        self,
        data_path: Path,
        gt_coco_json_path: Path,
        image_size: int = 300,
    ) -> None:
        self.data_path = Path(data_path)
        self.image_size = int(image_size)

        with Path(gt_coco_json_path).open("r", encoding="utf-8") as handle:
            coco_json = json.load(handle)

        self.records: list[dict[str, Any]] = []
        skipped = 0
        for image in coco_json.get("images", []):
            resolved = _resolve_image_path(self.data_path, image["file_name"])
            if resolved is None:
                skipped += 1
                continue
            self.records.append(
                {
                    "image_id": int(image["id"]),
                    "file_name": image["file_name"],
                    "path": resolved,
                    "width": int(image.get("width", 0)),
                    "height": int(image.get("height", 0)),
                }
            )
        if skipped:
            logger.warning("Dataset: %d image(s) could not be located.", skipped)
        logger.info("Dataset ready: %d image(s).", len(self.records))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, Any]]:
        record = self.records[index]
        with Image.open(record["path"]) as pil_image:
            image = pil_image.convert("RGB")
            original_width, original_height = image.size
            resized = image.resize(
                (self.image_size, self.image_size), Image.BILINEAR
            )

        array = np.asarray(resized, dtype=np.float32) / 255.0  # HWC, [0, 1]
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()  # CHW

        metadata = {
            "image_id": record["image_id"],
            "file_name": record["file_name"],
            "width": record["width"] or original_width,
            "height": record["height"] or original_height,
        }
        return tensor, metadata


def detection_collate(
    batch: Sequence[tuple[torch.Tensor, dict[str, Any]]],
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Collate ``(tensor, metadata)`` samples into a batched tensor + list."""
    tensors = torch.stack([item[0] for item in batch], dim=0)
    metadata = [item[1] for item in batch]
    return tensors, metadata


def create_dataloader(
    data_path: Path,
    gt_coco_json_path: Path,
    image_size: int = 300,
    batch_size: int = 1,
    num_workers: int = 4,
) -> DataLoader:
    """Build a :class:`DataLoader` over :class:`VOCDetectionDataset`."""
    dataset = VOCDetectionDataset(data_path, gt_coco_json_path, image_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=detection_collate,
    )


# =============================================================================
# Model loading
# =============================================================================


def _ensure_qfgaohao_repo(repo_path: Path) -> Path:
    """Resolve the qfgaohao ``pytorch-ssd`` repo and put it on ``sys.path``.

    The PyTorch and ExecuTorch backends both import that repo's ``vision``
    package. This resolves ``repo_path``, verifies the package is actually
    present, and prepends the repo to ``sys.path``. Failing here with a clear,
    actionable message is far better than the opaque ``No module named
    'vision'`` raised deep inside an ``import`` statement.
    """
    repo_path = Path(repo_path).resolve()
    if not (repo_path / "vision").is_dir():
        raise ModuleNotFoundError(
            f"qfgaohao 'vision' package not found under repo_path={repo_path}. "
            f"Point the config's model.model_sources_repo_path (or "
            f"model.source_path) at the pytorch-ssd repo root — the directory "
            f"that contains the 'vision/' package."
        )
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))
    return repo_path


def load_pytorch_model(
    model_path: Path,
    repo_path: Path,
    num_classes: int = 21,
) -> torch.nn.Module:
    """Build a qfgaohao MobileNetV2-SSD-Lite and load its weights.

    # ── VERIFY ── qfgaohao import path, factory name, and checkpoint shape
    # against your repo. The model is created with ``is_test=True`` so its
    # forward pass returns ``(confidences, locations)`` raw SSD head outputs.
    """
    repo_path = _ensure_qfgaohao_repo(repo_path)

    from vision.ssd.mobilenet_v2_ssd_lite import (  # type: ignore
        create_mobilenetv2_ssd_lite,
    )

    net = create_mobilenetv2_ssd_lite(num_classes, is_test=True)

    checkpoint: Any = torch.load(
        str(model_path), map_location="cpu", weights_only=False
    )
    if isinstance(checkpoint, torch.nn.Module):
        net.load_state_dict(checkpoint.state_dict())
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        net.load_state_dict(checkpoint["state_dict"])
    elif isinstance(checkpoint, dict):
        net.load_state_dict(checkpoint)
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")

    net.eval()
    logger.info("PyTorch model loaded: %s", Path(model_path).name)
    return net


def load_executorch_model(pte_path: Path) -> Any:
    """Load a ``.pte`` ExecuTorch program for inference.

    # ── VERIFY ── ExecuTorch runtime API. This is the most version-sensitive
    # spot in the file. The pybindings loader below is common; replace it with
    # whatever your toolkit uses if it differs.
    """
    from executorch.extension.pybindings.portable_lib import (  # type: ignore
        _load_for_executorch,
    )

    program = _load_for_executorch(str(pte_path))
    logger.info("ExecuTorch program loaded: %s", Path(pte_path).name)
    return program


# =============================================================================
# SSD decoding (PyTorch / ExecuTorch shared path)
# =============================================================================


def _load_priors_and_boxutils(repo_path: Path) -> tuple[torch.Tensor, Any]:
    """Load SSD prior boxes and box utilities from the qfgaohao repo.

    MobileNetV2-SSD-Lite reuses the MobileNetV1-SSD prior configuration.

    # ── VERIFY ── qfgaohao module paths.
    """
    repo_path = _ensure_qfgaohao_repo(repo_path)

    from vision.ssd.config import mobilenetv1_ssd_config as ssd_config  # type: ignore
    from vision.utils import box_utils  # type: ignore

    return ssd_config.priors, box_utils


def decode_ssd_predictions(
    confidences: torch.Tensor,
    locations: torch.Tensor,
    priors: torch.Tensor,
    box_utils: Any,
    metadata: Sequence[dict[str, Any]],
    decoder_cfg: dict[str, Any],
    num_classes: int,
) -> list[dict[str, Any]]:
    """Decode raw SSD head outputs into COCO-format detection dicts.

    Applies softmax, center-to-corner box decoding, per-class score
    thresholding and hard NMS, scales boxes to original image dimensions, and
    builds COCO detection dicts.

    Category-id remap (correctness fix)
    -----------------------------------
    The SSD head emits channel indices ``1..num_classes-1`` — i.e. VOC indices
    1-20. Ground truth uses real COCO category ids, so every prediction is
    projected through :data:`VOC_INDEX_TO_COCO_ID`. Indices with no VOC->COCO
    mapping are skipped. Without this remap the PyTorch / ExecuTorch backends
    score ~0.087 mAP against the COCO-id ground truth.
    """
    score_threshold = float(decoder_cfg.get("score_threshold", 0.01))
    nms_iou_threshold = float(decoder_cfg.get("nms_iou_threshold", 0.45))
    max_detections = int(decoder_cfg.get("max_detections_per_image", 100))
    candidate_size = int(decoder_cfg.get("candidate_size", 200))
    center_variance = float(decoder_cfg.get("center_variance", 0.1))
    size_variance = float(decoder_cfg.get("size_variance", 0.2))

    confidences = confidences.detach().to("cpu")
    locations = locations.detach().to("cpu")
    scores = F.softmax(confidences, dim=-1)

    detections: list[dict[str, Any]] = []
    batch_size = scores.shape[0]
    for batch_index in range(batch_size):
        meta = metadata[batch_index]
        image_id = int(meta["image_id"])
        original_width = float(meta["width"])
        original_height = float(meta["height"])

        # Center-form -> corner-form boxes, normalized to [0, 1].
        boxes = box_utils.convert_locations_to_boxes(
            locations[batch_index], priors, center_variance, size_variance
        )
        boxes = box_utils.center_form_to_corner_form(boxes)
        image_scores = scores[batch_index]

        per_image: list[dict[str, Any]] = []
        # Channel 0 is background; classes are 1..num_classes-1 (VOC index).
        for class_index in range(1, num_classes):
            class_probs = image_scores[:, class_index]
            keep = class_probs > score_threshold
            if not bool(keep.any()):
                continue

            coco_category_id = VOC_INDEX_TO_COCO_ID.get(class_index)
            if coco_category_id is None:
                continue  # channel has no VOC->COCO mapping; skip.

            subset_boxes = boxes[keep, :]
            subset_probs = class_probs[keep]
            box_probs = torch.cat(
                [subset_boxes, subset_probs.reshape(-1, 1)], dim=1
            )
            box_probs = box_utils.hard_nms(
                box_probs,
                iou_threshold=nms_iou_threshold,
                top_k=max_detections,
                candidate_size=candidate_size,
            )

            for row in box_probs:
                x1 = float(row[0]) * original_width
                y1 = float(row[1]) * original_height
                x2 = float(row[2]) * original_width
                y2 = float(row[3]) * original_height
                width = max(x2 - x1, 0.0)
                height = max(y2 - y1, 0.0)
                per_image.append(
                    {
                        "image_id": image_id,
                        "category_id": int(coco_category_id),
                        "bbox": [x1, y1, width, height],
                        "score": float(row[4]),
                    }
                )

        per_image.sort(key=lambda det: det["score"], reverse=True)
        detections.extend(per_image[:max_detections])

    return detections


# =============================================================================
# Normalization
# =============================================================================


def _qfgaohao_normalize(
    tensor: torch.Tensor,
    mean: Sequence[float],
    std: Sequence[float],
) -> torch.Tensor:
    """Apply qfgaohao channel normalization to a ``[0, 1]`` NCHW tensor.

    Defaults of mean ``0.498`` / std ``0.502`` reproduce qfgaohao's
    ``(x - 127) / 128`` on ``[0, 255]`` images. This is correct and intentional.
    """
    mean_tensor = torch.tensor(mean, dtype=tensor.dtype).reshape(1, -1, 1, 1)
    std_tensor = torch.tensor(std, dtype=tensor.dtype).reshape(1, -1, 1, 1)
    return (tensor - mean_tensor) / std_tensor


# =============================================================================
# PyTorch / ExecuTorch evaluators
# =============================================================================


def _summarize_latency(latencies_ms: Sequence[float]) -> dict[str, float]:
    """Summarize a list of per-inference latencies (milliseconds)."""
    if not latencies_ms:
        return {"mean_ms": 0.0, "total_ms": 0.0, "samples": 0}
    array = np.asarray(latencies_ms, dtype=float)
    return {
        "mean_ms": round(float(array.mean()), 4),
        "total_ms": round(float(array.sum()), 4),
        "samples": int(array.size),
    }


def evaluate_pytorch_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    priors: torch.Tensor,
    box_utils: Any,
    decoder_cfg: dict[str, Any],
    norm_mean: Sequence[float],
    norm_std: Sequence[float],
    num_classes: int,
    max_samples: Optional[int] = None,
) -> dict[str, Any]:
    """Run the PyTorch model over ``dataloader`` and decode predictions.

    Returns ``{"predictions", "samples_processed", "latency",
    "latency_by_image"}``.
    """
    predictions: list[dict[str, Any]] = []
    latencies_ms: list[float] = []
    latency_by_image: dict[int, float] = {}
    samples_processed = 0

    model.eval()
    with torch.no_grad():
        for tensors, metadata in dataloader:
            if max_samples is not None and samples_processed >= max_samples:
                break
            tensors = _qfgaohao_normalize(tensors, norm_mean, norm_std)

            start = time.perf_counter()
            confidences, locations = model(tensors)
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            batch_predictions = decode_ssd_predictions(
                confidences, locations, priors, box_utils, metadata,
                decoder_cfg, num_classes,
            )
            predictions.extend(batch_predictions)

            per_image_ms = elapsed_ms / max(len(metadata), 1)
            for meta in metadata:
                latency_by_image[int(meta["image_id"])] = per_image_ms
                latencies_ms.append(per_image_ms)
            samples_processed += len(metadata)

    logger.info("PyTorch evaluation: %d image(s) processed.", samples_processed)
    return {
        "predictions": predictions,
        "samples_processed": samples_processed,
        "latency": _summarize_latency(latencies_ms),
        "latency_by_image": latency_by_image,
    }


def evaluate_executorch_model(
    program: Any,
    dataloader: DataLoader,
    priors: torch.Tensor,
    box_utils: Any,
    decoder_cfg: dict[str, Any],
    norm_mean: Sequence[float],
    norm_std: Sequence[float],
    num_classes: int,
    max_samples: Optional[int] = None,
) -> dict[str, Any]:
    """Run an ExecuTorch ``.pte`` program over ``dataloader`` and decode.

    # ── VERIFY ── the program invocation API (``program.forward([...])``) and
    # the output ordering against your runtime. Same return shape as
    # :func:`evaluate_pytorch_model`.
    """
    predictions: list[dict[str, Any]] = []
    latencies_ms: list[float] = []
    latency_by_image: dict[int, float] = {}
    samples_processed = 0

    for tensors, metadata in dataloader:
        if max_samples is not None and samples_processed >= max_samples:
            break
        tensors = _qfgaohao_normalize(tensors, norm_mean, norm_std)

        start = time.perf_counter()
        outputs = program.forward([tensors])  # ── VERIFY ── invocation API
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        confidences, locations = outputs[0], outputs[1]
        if not torch.is_tensor(confidences):
            confidences = torch.as_tensor(np.asarray(confidences))
        if not torch.is_tensor(locations):
            locations = torch.as_tensor(np.asarray(locations))

        batch_predictions = decode_ssd_predictions(
            confidences, locations, priors, box_utils, metadata,
            decoder_cfg, num_classes,
        )
        predictions.extend(batch_predictions)

        per_image_ms = elapsed_ms / max(len(metadata), 1)
        for meta in metadata:
            latency_by_image[int(meta["image_id"])] = per_image_ms
            latencies_ms.append(per_image_ms)
        samples_processed += len(metadata)

    logger.info(
        "ExecuTorch evaluation: %d image(s) processed.", samples_processed
    )
    return {
        "predictions": predictions,
        "samples_processed": samples_processed,
        "latency": _summarize_latency(latencies_ms),
        "latency_by_image": latency_by_image,
    }


# =============================================================================
# TFLite backend (ported inline from compare_tflite_pytorch_coco.py)
# =============================================================================


def _load_tflite_interpreter_cls() -> Any:
    """Lazily resolve a TFLite ``Interpreter`` class.

    Imported lazily so this module loads cleanly without TensorFlow installed;
    the TFLite backend is optional and config-gated.
    """
    try:
        from tflite_runtime.interpreter import Interpreter
        return Interpreter
    except ImportError:
        try:
            import tensorflow as tf  # type: ignore
            return tf.lite.Interpreter
        except ImportError as exc:
            raise ImportError(
                "TFLite backend requires tflite-runtime or tensorflow. "
                "Install one:  pip install tflite-runtime  "
                "(or)  pip install tensorflow"
            ) from exc


class TFLiteSSDEvaluator:
    """Run a TFLite SSD MobileNet V2 model and emit COCO-format detections.

    The model bakes ``TFLite_Detection_PostProcess`` into its graph, so the
    interpreter returns decoded boxes/classes/scores/count directly -- no
    Python-side prior decode or NMS is required. It emits real COCO category
    ids and therefore needs no remap.
    """

    _EXPECTED_NUM_OUTPUTS = 4  # boxes, classes, scores, num_detections

    def __init__(self, model_path: Path) -> None:
        self.model_path = Path(model_path)
        interpreter_cls = _load_tflite_interpreter_cls()
        self._interpreter = interpreter_cls(model_path=str(self.model_path))
        self._interpreter.allocate_tensors()

        self._input_details = self._interpreter.get_input_details()
        # Sort outputs by tensor index -> stable [boxes, classes, scores, count].
        self._output_details = sorted(
            self._interpreter.get_output_details(),
            key=lambda detail: detail["index"],
        )
        if len(self._output_details) != self._EXPECTED_NUM_OUTPUTS:
            raise RuntimeError(
                f"Expected {self._EXPECTED_NUM_OUTPUTS} output tensors "
                f"(boxes, classes, scores, num_detections); got "
                f"{len(self._output_details)}"
            )

        input_shape = self._input_details[0]["shape"]
        if len(input_shape) != 4 or input_shape[3] != 3:
            raise ValueError(
                f"Expected NHWC 3-channel input; got shape {input_shape}"
            )
        self._input_height = int(input_shape[1])
        self._input_width = int(input_shape[2])
        self._input_dtype = self._input_details[0]["dtype"]

        # Largest class index seen across the run; logged afterward to confirm
        # the model is genuinely 90-class (TF OD API) and not 80-class, which
        # would invalidate the identity+1 mapping.
        self._max_class_idx_observed: int = -1

        logger.info(
            "TFLite model loaded: %s | input: %s %s",
            self.model_path.name,
            tuple(input_shape),
            self._input_dtype.__name__,
        )

    def _preprocess(self, pil_image: Image.Image) -> np.ndarray:
        """Resize, cast to the model dtype, add a batch dim. NHWC layout."""
        resized = pil_image.convert("RGB").resize(
            (self._input_width, self._input_height), Image.BILINEAR
        )
        array = np.asarray(resized)
        if self._input_dtype == np.uint8:
            return array[np.newaxis, ...]
        # Float graphs expect [-1, 1] (standard MobileNet preprocessing). The
        # PTQ uint8 graph is the expected case; this branch is defensive only.
        return ((array.astype(np.float32) / 127.5) - 1.0)[np.newaxis, ...]

    def _invoke(
        self, pil_image: Image.Image
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Run one inference. Returns ``(boxes, classes, scores, count)``.

        ``boxes`` are ``[ymin, xmin, ymax, xmax]`` normalized to ``[0, 1]``;
        ``classes`` are 0-indexed into :data:`TFLITE_COCO90_INDEX_TO_COCO_ID`.
        """
        input_tensor = self._preprocess(pil_image)
        self._interpreter.set_tensor(
            self._input_details[0]["index"], input_tensor
        )
        self._interpreter.invoke()

        tensors = [
            self._interpreter.get_tensor(detail["index"])
            for detail in self._output_details
        ]
        boxes_t, classes_t, scores_t, count_t = tensors
        if boxes_t.ndim != 3 or boxes_t.shape[-1] != 4:
            raise RuntimeError(
                f"Unexpected boxes shape: {boxes_t.shape}; this likely means "
                f"the output tensors are in a non-standard order"
            )

        count = int(np.asarray(count_t).flat[0])
        boxes = boxes_t[0][:count]
        classes = classes_t[0][:count].astype(int)
        scores = scores_t[0][:count]
        return boxes, classes, scores, count

    def detections_for_image(
        self,
        pil_image: Image.Image,
        image_id: int,
        original_width: int,
        original_height: int,
        score_threshold: float,
    ) -> tuple[list[dict[str, Any]], float]:
        """Run inference and return ``(coco_detection_dicts, latency_ms)``."""
        start = time.perf_counter()
        boxes, classes, scores, count = self._invoke(pil_image)
        latency_ms = (time.perf_counter() - start) * 1000.0

        detections: list[dict[str, Any]] = []
        max_index = len(TFLITE_COCO90_INDEX_TO_COCO_ID)
        for i in range(count):
            score = float(scores[i])
            if score < score_threshold:
                continue
            class_index = int(classes[i])
            if not 0 <= class_index < max_index:
                continue
            if class_index > self._max_class_idx_observed:
                self._max_class_idx_observed = class_index

            category_id = TFLITE_COCO90_INDEX_TO_COCO_ID[class_index]
            ymin, xmin, ymax, xmax = (float(v) for v in boxes[i])
            x1 = xmin * original_width
            y1 = ymin * original_height
            x2 = xmax * original_width
            y2 = ymax * original_height
            detections.append(
                {
                    "image_id": int(image_id),
                    "category_id": int(category_id),
                    "bbox": [x1, y1, max(x2 - x1, 0.0), max(y2 - y1, 0.0)],
                    "score": score,
                }
            )
        return detections, latency_ms

    @property
    def max_class_index_observed(self) -> int:
        """Largest class index emitted across all invocations so far."""
        return self._max_class_idx_observed


def evaluate_tflite_model(
    model_path: Path,
    data_path: Path,
    gt_coco_json_path: Path,
    score_threshold: float = 0.01,
    max_samples: Optional[int] = None,
) -> dict[str, Any]:
    """Run the TFLite backend over the evaluation images.

    TFLite needs raw images (it does its own uint8 resize), so this iterates the
    ground-truth JSON image list directly rather than the normalized-tensor
    DataLoader. Returns the same shape as :func:`evaluate_pytorch_model`.
    """
    evaluator = TFLiteSSDEvaluator(model_path)

    with Path(gt_coco_json_path).open("r", encoding="utf-8") as handle:
        coco_json = json.load(handle)
    images = coco_json.get("images", [])
    if max_samples is not None:
        images = images[:max_samples]

    predictions: list[dict[str, Any]] = []
    latencies_ms: list[float] = []
    latency_by_image: dict[int, float] = {}
    samples_processed = 0
    skipped = 0

    for image in images:
        image_id = int(image["id"])
        resolved = _resolve_image_path(Path(data_path), image["file_name"])
        if resolved is None:
            skipped += 1
            continue

        with Image.open(resolved) as pil_image:
            rgb = pil_image.convert("RGB")
            original_width = int(image.get("width") or rgb.size[0])
            original_height = int(image.get("height") or rgb.size[1])
            detections, latency_ms = evaluator.detections_for_image(
                rgb, image_id, original_width, original_height, score_threshold
            )

        predictions.extend(detections)
        latencies_ms.append(latency_ms)
        latency_by_image[image_id] = latency_ms
        samples_processed += 1

    if skipped:
        logger.warning("TFLite evaluation: %d image(s) could not be located.", skipped)
    logger.info(
        "TFLite evaluation: %d image(s) processed | max class index observed: %d",
        samples_processed,
        evaluator.max_class_index_observed,
    )
    if 0 <= evaluator.max_class_index_observed <= 80:
        logger.warning(
            "TFLite max class index <= 80 -- model may not be 90-class; "
            "the identity+1 category mapping could be wrong."
        )

    return {
        "predictions": predictions,
        "samples_processed": samples_processed,
        "latency": _summarize_latency(latencies_ms),
        "latency_by_image": latency_by_image,
    }


# =============================================================================
# Results assembly & IO
# =============================================================================


def _build_backend_entry(
    name: str,
    backend: str,
    model_path: Optional[Path],
    metrics: dict[str, Any],
    latency: dict[str, Any],
    samples_processed: int,
    per_image_results: list[dict[str, Any]],
    *,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Assemble one backend's report entry.

    Used for the TFLite and PyTorch baselines (and reused for ExecuTorch
    variants) so every backend column shares an identical schema -- which lets
    the report generator render them all through one code path.

    The entry carries both the raw ``latency`` stats and a derived ``timing``
    block. reporting.py reads ``timing`` directly and only falls back to
    ``latency.mean_ms`` when ``timing`` is absent; emitting both keeps the
    report's timing columns populated whichever path the reporter takes.
    """
    mean_ms = float(latency.get("mean_ms", 0.0) or 0.0)
    total_ms = float(latency.get("total_ms", 0.0) or 0.0)
    entry: dict[str, Any] = {
        "name": name,
        # reporting.py reads ``model_name`` in _prepare_report_data and ``name``
        # in _build_comparison_table; emit both so neither falls back to
        # "Unknown".
        "model_name": name,
        "backend": backend,
        "model_path": str(model_path) if model_path is not None else None,
        "model_size_mb": _file_size_mb(model_path) if model_path else None,
        "samples_processed": samples_processed,
        "metrics": metrics,
        "latency": latency,
        "timing": {
            "avg_inference_time_ms": mean_ms,
            "total_time_s": round(total_ms / 1000.0, 4),
            "throughput_samples_per_sec": (
                round(1000.0 / mean_ms, 4) if mean_ms > 0.0 else 0.0
            ),
        },
        "per_image_results": per_image_results,
    }
    if extra:
        entry.update(extra)
    return entry


def _system_info() -> dict[str, Any]:
    """Collect a small environment fingerprint for the report."""
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
    }


def _json_default(value: Any) -> Any:
    """JSON fallback encoder for :func:`save_evaluation_results`.

    Converts NumPy scalars/arrays and ``Path`` objects to native types so the
    report pipeline receives real numbers (not strings). Anything genuinely
    non-serializable raises ``TypeError`` instead of being silently coerced --
    ``default=str`` would turn a stray ``np.float32`` metric into a string,
    which reporting.py's ``isinstance(v, (int, float))`` checks then drop.
    """
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(
        f"Object of type {type(value).__name__} is not JSON-serializable"
    )


def save_evaluation_results(results: dict[str, Any], output_file: Path) -> None:
    """Write the consolidated ``results`` dict to ``output_file`` as JSON."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, default=_json_default)
    logger.info("Consolidated results written: %s", output_file)


# =============================================================================
# Main
# =============================================================================


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate MobileNetV2-SSD-Lite across TFLite, PyTorch, "
        "and ExecuTorch backends."
    )
    parser.add_argument("--config", required=True, help="Path to the JSON config.")
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Limit the number of evaluation images.",
    )
    parser.add_argument("--skip-tflite", action="store_true")
    parser.add_argument("--skip-pytorch", action="store_true")
    parser.add_argument("--skip-executorch", action="store_true")
    parser.add_argument(
        "--generate-report", dest="generate_report",
        action="store_true", default=True,
    )
    parser.add_argument(
        "--no-report", dest="generate_report", action="store_false",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Entry point: run the configured backends and emit the report."""
    args = _parse_args(argv)

    config_path = Path(args.config).expanduser().resolve()
    config_base = config_path.parent
    config = _load_config(config_path)

    model_cfg = config["model"]
    eval_cfg = config["evaluation"]
    dataset_cfg = eval_cfg["dataset"]
    decoder_cfg = eval_cfg["decoder"]
    backends_cfg = eval_cfg["backends"]
    report_cfg = eval_cfg.get("report", {})
    paths_cfg = eval_cfg.get("paths", {})

    # --- resolve inputs -------------------------------------------------------
    class_names: list[str] = dataset_cfg["class_names"]
    num_classes = int(model_cfg.get("num_classes", len(class_names)))
    norm_mean = model_cfg.get("normalize_mean", [0.498, 0.498, 0.498])
    norm_std = model_cfg.get("normalize_std", [0.502, 0.502, 0.502])
    image_size = int(decoder_cfg.get("image_size", 300))
    max_samples = args.max_samples if args.max_samples is not None else \
        dataset_cfg.get("max_samples")

    data_path = _resolve_path(dataset_cfg["data_path"], config_base)
    gt_coco_json = _resolve_path(dataset_cfg["gt_coco_json"], config_base)
    results_dir = _resolve_path(
        paths_cfg.get("results_dir", "output/results"), config_base
    )
    repo_path = _resolve_path(
        model_cfg.get("model_sources_repo_path", model_cfg.get("source_path", ".")),
        config_base,
    )

    # --- shared evaluation assets --------------------------------------------
    dataloader = create_dataloader(
        data_path, gt_coco_json, image_size,
        batch_size=int(dataset_cfg.get("batch_size", 1)),
        num_workers=int(dataset_cfg.get("num_workers", 4)),
    )
    image_metadata = [
        {
            "image_id": record["image_id"],
            "file_name": record["file_name"],
            "width": record["width"],
            "height": record["height"],
        }
        for record in dataloader.dataset.records  # type: ignore[attr-defined]
    ]
    if max_samples is not None:
        image_metadata = image_metadata[:max_samples]

    # --- results skeleton -----------------------------------------------------
    # Insertion order places 'tflite_baseline' before 'pytorch_baseline'; the
    # report generator (reporting.py) must still render TFLite as the leftmost
    # column explicitly.
    results: dict[str, Any] = {
        "model_name": model_cfg.get("model_name", "mobile_net_v2_ssd"),
        "task": eval_cfg.get("task_type", "detection"),
        "task_type": eval_cfg.get("task_type", "detection"),
        "primary_metric": eval_cfg.get("primary_metric", "mAP_0.5_0.95"),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "num_samples": len(image_metadata),
        "dataset": {
            "name": data_path.name,
            "data_path": str(data_path),
            "gt_coco_json": str(gt_coco_json),
            "num_images": len(image_metadata),
            "num_samples": len(image_metadata),
            "samples_evaluated": len(image_metadata),
        },
        "system_info": _system_info(),
        "tflite_baseline": {},
        "pytorch_baseline": {},
        "executorch_models": [],
    }

    # ------------------------------------------------------------------ TFLite
    tflite_cfg = backends_cfg.get("tflite_baseline")
    if (
        not args.skip_tflite
        and backends_cfg.get("include_tflite_baseline")
        and tflite_cfg
    ):
        try:
            tflite_model_path = _resolve_path(tflite_cfg["model_path"], config_base)
            tflite_results = evaluate_tflite_model(
                tflite_model_path, data_path, gt_coco_json,
                score_threshold=float(tflite_cfg.get("score_threshold", 0.01)),
                max_samples=max_samples,
            )
            # No class_subset here: the TFLite baseline is a COCO-90 model, so
            # it is scored over the full ground-truth category space. Only the
            # qfgaohao VOC model (PyTorch / ExecuTorch) is restricted.
            tflite_metrics = calculate_detection_metrics(
                tflite_results["predictions"], gt_coco_json
            )
            results["tflite_baseline"] = _build_backend_entry(
                name=tflite_cfg.get("name", "tflite"),
                backend="TFLite",
                model_path=tflite_model_path,
                metrics=tflite_metrics,
                latency=tflite_results["latency"],
                samples_processed=tflite_results["samples_processed"],
                per_image_results=generate_per_image_detection_results(
                    tflite_results["predictions"], image_metadata, gt_coco_json,
                    tflite_results["latency_by_image"],
                ),
            )
            logger.info(
                "TFLite mAP@[.5:.95]=%.4f  mAP@0.5=%.4f",
                tflite_metrics["mAP_0.5_0.95"], tflite_metrics["mAP_0.5"],
            )
        except Exception as exc:  # optional backend: never abort the run
            logger.error("TFLite backend failed (%s); skipping.", exc)
    else:
        logger.info("TFLite backend disabled or skipped.")

    # ----------------------------------------------------------------- PyTorch
    priors: Optional[torch.Tensor] = None
    box_utils: Any = None
    if not args.skip_pytorch and backends_cfg.get("include_pytorch_baseline", True):
        try:
            priors, box_utils = _load_priors_and_boxutils(repo_path)
            pytorch_model_path = _resolve_path(model_cfg["model_path"], config_base)
            pytorch_model = load_pytorch_model(
                pytorch_model_path, repo_path, num_classes
            )
            pytorch_results = evaluate_pytorch_model(
                pytorch_model, dataloader, priors, box_utils, decoder_cfg,
                norm_mean, norm_std, num_classes, max_samples,
            )
            pytorch_metrics = calculate_detection_metrics(
                pytorch_results["predictions"], gt_coco_json,
                class_subset=MODEL_COCO_CATEGORY_IDS,
            )
            results["pytorch_baseline"] = _build_backend_entry(
                name="pytorch_baseline",
                backend="PyTorch",
                model_path=pytorch_model_path,
                metrics=pytorch_metrics,
                latency=pytorch_results["latency"],
                samples_processed=pytorch_results["samples_processed"],
                per_image_results=generate_per_image_detection_results(
                    pytorch_results["predictions"], image_metadata, gt_coco_json,
                    pytorch_results["latency_by_image"],
                ),
            )
            logger.info(
                "PyTorch mAP@[.5:.95]=%.4f  mAP@0.5=%.4f",
                pytorch_metrics["mAP_0.5_0.95"], pytorch_metrics["mAP_0.5"],
            )
        except Exception as exc:
            logger.error("PyTorch backend failed (%s); skipping.", exc)
    else:
        logger.info("PyTorch backend disabled or skipped.")

    # -------------------------------------------------------------- ExecuTorch
    executorch_variants = backends_cfg.get("executorch_models", [])
    if not args.skip_executorch and executorch_variants:
        if priors is None or box_utils is None:
            # PyTorch was skipped or failed, but ExecuTorch shares the SSD
            # decoder. A failure loading it here is non-fatal: skip ExecuTorch
            # like any other optional backend instead of aborting the run.
            try:
                priors, box_utils = _load_priors_and_boxutils(repo_path)
            except Exception as exc:
                logger.error(
                    "ExecuTorch backend cannot load the SSD decoder (%s); "
                    "skipping.", exc,
                )
                executorch_variants = []

        for variant in executorch_variants:
            name = variant.get("name", "executorch")
            try:
                pte_path = _resolve_path(variant["pte_path"], config_base)
                if not pte_path.is_file():
                    logger.warning("ExecuTorch variant '%s' missing: %s",
                                   name, pte_path)
                    continue

                program = load_executorch_model(pte_path)
                variant_results = evaluate_executorch_model(
                    program, dataloader, priors, box_utils, decoder_cfg,
                    norm_mean, norm_std, num_classes, max_samples,
                )
                variant_metrics = calculate_detection_metrics(
                    variant_results["predictions"], gt_coco_json,
                    class_subset=MODEL_COCO_CATEGORY_IDS,
                )

                ptd_path = pte_path.with_suffix(".ptd")
                extra: dict[str, Any] = {
                    "is_baseline": bool(variant.get("is_baseline", False)),
                    "pte_path": str(pte_path),
                    "pte_size_mb": _file_size_mb(pte_path),
                }
                if ptd_path.is_file():
                    extra["ptd_path"] = str(ptd_path)
                    extra["ptd_size_mb"] = _file_size_mb(ptd_path)

                results["executorch_models"].append(
                    _build_backend_entry(
                        name=name,
                        backend=variant.get("backend", "ExecuTorch"),
                        model_path=pte_path,
                        metrics=variant_metrics,
                        latency=variant_results["latency"],
                        samples_processed=variant_results["samples_processed"],
                        per_image_results=generate_per_image_detection_results(
                            variant_results["predictions"], image_metadata,
                            gt_coco_json, variant_results["latency_by_image"],
                        ),
                        extra=extra,
                    )
                )
                logger.info(
                    "ExecuTorch '%s' mAP@[.5:.95]=%.4f  mAP@0.5=%.4f",
                    name, variant_metrics["mAP_0.5_0.95"],
                    variant_metrics["mAP_0.5"],
                )
            except Exception as exc:
                logger.error(
                    "ExecuTorch variant '%s' failed (%s); skipping.", name, exc
                )
        if not results["executorch_models"]:
            logger.warning("No ExecuTorch variants evaluated successfully.")
    else:
        logger.info("ExecuTorch backends disabled or skipped.")

    # ------------------------------------------------------------------ output
    consolidated_name = report_cfg.get(
        "consolidated_json_name", f"{results['model_name']}_evaluation.json"
    )
    output_file = results_dir / consolidated_name
    save_evaluation_results(results, output_file)

    if args.generate_report and report_cfg.get("generate_html", True):
        try:
            # ── VERIFY ── report-module import path.
            from evaluation.common.reporting.generate_html_report import (  # type: ignore
                generate_html_for_json,
            )

            generate_html_for_json(output_file, output_dir=results_dir)
            logger.info("HTML report generated in: %s", results_dir)
        except Exception as exc:
            logger.error("HTML report generation failed (%s).", exc)


if __name__ == "__main__":
    main()
