#!/usr/bin/env python3
"""
Standalone Evaluation Script for MobileNetV2-SSD-Lite Detection Model
======================================================================

Evaluates three backends on Pascal VOC 2012 val:
  * PyTorch (qfgaohao MobileNetV2-SSD-Lite, is_test=False)         -> reference
  * TFLite  (qfgaohao quantized export; baked anchors+decode+NMS)  -> legacy baseline
  * ExecuTorch (.pte exported via team's executorch-toolkit;
                FP32 + INT8 per-tensor + INT8 per-channel variants) -> candidate

PyTorch is the reference (highest mAP in baseline run); ExecuTorch is the candidate.

Captured metrics
----------------
Accuracy:   mAP@0.5, mAP@0.5:0.95, mAP@0.75, size-bucket mAP (small/medium/large),
            AR@1/10/100, per-class AP@0.5, both class-agnostic and VOC-restricted.
Latency:    mean/std/min/max + p50/p90/p95/p99 + throughput FPS.
Size:       PTE, PTD, combined, compression ratio vs FP32 reference.
Parity:     PyTorch <-> ExecuTorch matched-detection IoU, score delta,
            match/extra/missed rate, class agreement.
Repro:      versions, hardware, config snapshot, dataset stats, seed.

Outputs
-------
  <results-dir>/evaluation_results.json   (machine-readable; fed to generate_report.py)
  Optional HTML via evaluation.mobilenetv2.generate_report.generate_html_for_json

Scope & Deviations
------------------
* Ground truth: Pascal VOC 2012 val converted to COCO format via dataset/scripts/voc_to_coco.py
  (5823 images, 13841 annotations, 20 categories, difficult objects excluded).
* PyTorch model loaded as the is_test=False pickleable artifact (mobile_net_v2_ssd.pth),
  decode + NMS performed in Python to match the export-time graph used for ExecuTorch.
* TFLite outputs are already decoded by the baked TFLite_Detection_PostProcess op;
  parsed directly without re-decode.
* Tensor layout is NCHW (toolkit standard); preprocessing follows qfgaohao convention:
  300x300, mean (127,127,127), std 128.0.
* Inspector C++ rebuild blocked (Qualcomm SDK 403 via Zscaler); ETDump preserved on disk
  for later analysis, not consumed here.

Usage
-----
    python evaluate.py \\
        --voc-root ~/.cache/.../VOCdevkit/VOC2012 \\
        --gt-coco-json ../../dataset/voc2012_val_coco.json \\
        --tflite-model ../../model_sources/MobileNetV2/weights/model.tflite \\
        --pytorch-model ../../model_sources/MobileNetV2/weights/mobile_net_v2_ssd.pth \\
        --model-dir ../../tests/integration/outputs/mobile_net_v2_ssd/basemodel_workflow/models \\
        --results-dir ../../output/eval_mobilenet_v2_ssd \\
        --generate-report

Author: Persistent Systems (Phase 2)
Date:   November 2025
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# ============================================================================
# Optional dependency shims (mirrors v1 pattern for mockable unit testing)
# ============================================================================

try:  # TFLite interpreter: prefer full TF, fall back to tflite-runtime.
    import tensorflow as _tf

    _TFLiteInterpreter = _tf.lite.Interpreter
except ImportError:
    try:
        import tflite_runtime.interpreter as _tflite

        _TFLiteInterpreter = _tflite.Interpreter
    except ImportError:
        _TFLiteInterpreter = None

try:  # pycocotools: inline fallback if the team's detection_metrics is unavailable.
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    _HAVE_PYCOCOTOOLS = True
except ImportError:
    COCO = None  # type: ignore[assignment]
    COCOeval = None  # type: ignore[assignment]
    _HAVE_PYCOCOTOOLS = False

# ============================================================================
# Project path resolution
#
# This file lives at:
#   <project_root>/executorch-toolkit/evaluation/mobilenetv2/evaluate.py
# So we need parents[2] for the toolkit root (for evaluation.common.* imports)
# and parents[3] for the project root (for the qfgaohao vision/ package).
# ============================================================================

_TOOLKIT_ROOT = Path(__file__).resolve().parents[2]
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_QFGAOHAO_VISION = (
    _PROJECT_ROOT / "model_sources" / "MobileNetV2" / "src" / "pytorch" / "pytorch-ssd"
)

if str(_TOOLKIT_ROOT) not in sys.path:
    sys.path.insert(0, str(_TOOLKIT_ROOT))
if _QFGAOHAO_VISION.exists() and str(_QFGAOHAO_VISION) not in sys.path:
    sys.path.insert(0, str(_QFGAOHAO_VISION))

# Toolkit's shared utilities. These are mandatory; failure to import is fatal.
from evaluation.common.utils.inference import InferenceTimer  # noqa: E402
from evaluation.common.utils.json_io import save_evaluation_results  # noqa: E402
from evaluation.common.utils.system_info import get_system_info  # noqa: E402

# qfgaohao vision package (anchor priors + box utilities). Required for decode.
try:
    from vision.ssd.config import mobilenetv1_ssd_config as _ssd_config  # noqa: E402
    from vision.ssd.mobilenet_v2_ssd_lite import (  # noqa: E402
        create_mobilenetv2_ssd_lite,
    )
    from vision.utils import box_utils as _box_utils  # noqa: E402

    _HAVE_QFGAOHAO = True
except ImportError as _qfg_exc:
    _ssd_config = None  # type: ignore[assignment]
    create_mobilenetv2_ssd_lite = None  # type: ignore[assignment]
    _box_utils = None  # type: ignore[assignment]
    _HAVE_QFGAOHAO = False
    _QFGAOHAO_IMPORT_ERROR = _qfg_exc

# Team's detection metrics (preferred). Falls back to inline COCO eval if signature
# differs or import fails.
try:
    from evaluation.mobilenetv2.detection_metrics import (  # noqa: E402
        calculate_detection_metrics as _team_calculate_detection_metrics,
    )

    _HAVE_TEAM_METRICS = True
except ImportError:
    _team_calculate_detection_metrics = None  # type: ignore[assignment]
    _HAVE_TEAM_METRICS = False


# ============================================================================
# Logging
# ============================================================================


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure root logging with a consistent format."""
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


logger = setup_logging()


# ============================================================================
# Configuration (frozen dataclasses for reproducibility)
# ============================================================================

# Pascal VOC 2012 class names (1-indexed; index 0 is the implicit background class).
VOC_CLASS_NAMES: Tuple[str, ...] = (
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)
NUM_VOC_CLASSES = len(VOC_CLASS_NAMES)  # 20 foreground classes


@dataclass(frozen=True)
class PreprocessingConfig:
    """qfgaohao MobileNetV2-SSD-Lite preprocessing parameters."""

    image_size: int = 300
    image_mean: Tuple[float, float, float] = (127.0, 127.0, 127.0)
    image_std: float = 128.0


@dataclass(frozen=True)
class DecodeConfig:
    """SSD post-processing parameters (anchor decode + per-class NMS).

    score_threshold is intentionally low (0.01) so the precision-recall curve
    used by mAP has a long tail; raising it would bias mAP downward.
    """

    iou_threshold: float = 0.45
    score_threshold: float = 0.01
    top_k: int = 200
    candidate_size: int = 200
    num_classes: int = NUM_VOC_CLASSES + 1  # +1 background


@dataclass(frozen=True)
class EvalConfig:
    """Top-level immutable configuration snapshot, serialized into results."""

    preprocess: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    decode: DecodeConfig = field(default_factory=DecodeConfig)
    layout: str = "NCHW"
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return {
            "preprocess": asdict(self.preprocess),
            "decode": asdict(self.decode),
            "layout": self.layout,
            "seed": self.seed,
        }


# ============================================================================
# Detection record (immutable; safe to pass between backends)
# ============================================================================


@dataclass(frozen=True)
class Detection:
    """One detection in COCO conventions: bbox_xywh = [x, y, w, h] in image pixels."""

    image_id: int
    bbox_xywh: Tuple[float, float, float, float]
    score: float
    category_id: int  # 1..NUM_VOC_CLASSES (background excluded)

    def to_coco(self) -> Dict[str, Any]:
        return {
            "image_id": int(self.image_id),
            "category_id": int(self.category_id),
            "bbox": [float(v) for v in self.bbox_xywh],
            "score": float(self.score),
        }


# ============================================================================
# Reproducibility
# ============================================================================


def seed_everything(seed: int) -> None:
    """Best-effort determinism for the evaluation pass."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - CPU eval expected
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# Dataset: VOC images keyed by COCO-format ground truth
# ============================================================================


class VOCDetectionDataset(Dataset):
    """Pascal VOC images paired with a COCO-format ground-truth JSON.

    The ground-truth JSON is produced by dataset/scripts/voc_to_coco.py and is the
    single source of truth for image-id <-> file mapping, image dimensions, and
    annotations. This dataset only emits images + identifiers; it never touches
    annotations directly (those flow into pycocotools).
    """

    def __init__(
        self,
        voc_root: Path,
        gt_coco_json: Path,
        preprocess: PreprocessingConfig,
        layout: str = "NCHW",
    ) -> None:
        self.voc_root = Path(voc_root)
        self.preprocess = preprocess
        self.layout = layout
        self.images_dir = self.voc_root / "JPEGImages"

        if not self.images_dir.is_dir():
            raise FileNotFoundError(
                f"VOC images directory not found: {self.images_dir}"
            )

        with open(gt_coco_json, "r", encoding="utf-8") as f:
            coco = json.load(f)

        self.images: List[Dict[str, Any]] = list(coco.get("images", []))
        if not self.images:
            raise ValueError(f"No images in {gt_coco_json}")

        self.categories: Dict[int, str] = {
            int(c["id"]): c["name"] for c in coco.get("categories", [])
        }
        if not self.categories:
            raise ValueError(f"No categories in {gt_coco_json}")

        logger.info(
            "VOCDetectionDataset: %d images, %d categories from %s",
            len(self.images),
            len(self.categories),
            gt_coco_json,
        )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        meta = self.images[idx]
        img_path = self.images_dir / meta["file_name"]
        pil_img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = pil_img.size  # PIL uses (width, height)

        input_tensor = preprocess_image(pil_img, self.preprocess, layout=self.layout)

        return {
            "input": input_tensor,
            "image_id": int(meta["id"]),
            "orig_size": (int(orig_w), int(orig_h)),
            "file_name": str(meta["file_name"]),
        }


def _identity_collate(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Bypass auto-collation for batch_size=1 detection eval.

    Detection has variable-count outputs per image; auto-collation of metadata
    tuples like `orig_size` produces awkward stacked tensors. Hard-asserting
    batch=1 lets us preserve the dict shape end-to-end.
    """
    if len(batch) != 1:
        raise RuntimeError(
            f"Detection eval requires batch_size=1, got {len(batch)}"
        )
    return batch[0]


# ============================================================================
# Preprocessing
# ============================================================================


def preprocess_image(
    pil_img: Image.Image,
    config: PreprocessingConfig,
    layout: str = "NCHW",
) -> torch.Tensor:
    """Resize -> normalize -> layout-convert.

    Mirrors qfgaohao.vision.ssd.predictor preprocessing exactly:
        x = (img - mean) / std,   mean=(127,127,127), std=128
    Returns a 4-D batched tensor ready for any backend.
    """
    if layout not in ("NCHW", "NHWC"):
        raise ValueError(f"Unsupported layout {layout!r}")

    resized = pil_img.resize(
        (config.image_size, config.image_size), Image.BILINEAR
    )
    arr = np.asarray(resized, dtype=np.float32)  # HxWxC, RGB
    mean = np.array(config.image_mean, dtype=np.float32)
    arr = (arr - mean) / float(config.image_std)

    tensor = torch.from_numpy(arr)  # HxWxC
    if layout == "NCHW":
        tensor = tensor.permute(2, 0, 1).contiguous()  # CxHxW
    tensor = tensor.unsqueeze(0)  # add batch
    return tensor


# ============================================================================
# Model loaders
# ============================================================================


def get_file_size_mb(path: os.PathLike[str] | str) -> float:
    """File size in MB; returns 0.0 if the file does not exist."""
    p = Path(path)
    return p.stat().st_size / (1024.0 * 1024.0) if p.exists() else 0.0


def load_pytorch_model(model_path: Path) -> Optional[torch.nn.Module]:
    """Load the qfgaohao MobileNetV2-SSD-Lite model in is_test=False mode.

    Two paths in priority order:
      1) Direct unpickle of a complete nn.Module checkpoint (mobile_net_v2_ssd.pth
         saved with `torch.save(model, ...)`). This is the recommended path because
         it preserves the exact graph used at export time.
      2) Factory + state_dict load from mb2-ssd-lite-mp-0_686.pth (the qfgaohao
         release weights). Fall-back only.

    Both paths return a model with is_test=False (or coerce it) so the forward
    returns raw (confidences, locations) - matching the .pte export contract.
    """
    if not _HAVE_QFGAOHAO:
        logger.error(
            "qfgaohao vision package unavailable (%s). "
            "Set PYTHONPATH or ensure %s exists.",
            _QFGAOHAO_IMPORT_ERROR,
            _QFGAOHAO_VISION,
        )
        return None

    logger.info("Loading PyTorch model from %s", model_path)

    # Path 1: full module unpickle.
    try:
        try:
            model = torch.load(
                model_path, map_location="cpu", weights_only=False
            )
        except TypeError:  # older torch without weights_only kwarg
            model = torch.load(model_path, map_location="cpu")

        if isinstance(model, torch.nn.Module):
            if hasattr(model, "is_test"):
                model.is_test = False  # enforce raw-output contract
            model.eval()
            logger.info("Loaded as full nn.Module (is_test=False enforced)")
            return model

        # Path 2: state_dict fallback.
        if isinstance(model, dict):
            logger.info("Checkpoint is a state_dict; reconstructing via factory")
            net = create_mobilenetv2_ssd_lite(
                num_classes=NUM_VOC_CLASSES + 1, is_test=False
            )
            state_dict = model.get("state_dict", model)
            net.load_state_dict(state_dict, strict=False)
            net.eval()
            return net

        logger.error(
            "Unsupported PyTorch checkpoint format: %s", type(model).__name__
        )
        return None
    except Exception as exc:
        logger.error("Failed to load PyTorch model: %s", exc)
        return None


def load_tflite_model(model_path: Path) -> Optional[Any]:
    """Load a TFLite interpreter; returns None if TFLite is unavailable."""
    if _TFLiteInterpreter is None:
        logger.error(
            "TFLite Interpreter unavailable (install tensorflow or tflite-runtime)"
        )
        return None
    logger.info("Loading TFLite model from %s", model_path)
    interp = _TFLiteInterpreter(model_path=str(model_path))
    interp.allocate_tensors()
    return interp


def load_executorch_model(model_path: Path) -> Optional[Any]:
    """Load a .pte and pair it with sidecar `<stem>_constants.ptd` if present.

    PTD pairing is generic (task-agnostic); mirrors v1 behavior.
    """
    try:
        from executorch.extension.pybindings.portable_lib import (
            _load_for_executorch,
        )
    except ImportError:
        logger.error("ExecuTorch runtime not available; cannot load .pte")
        return None

    ptd_path = model_path.parent / f"{model_path.stem}_constants.ptd"
    if ptd_path.exists():
        logger.info("Found PTD sidecar: %s", ptd_path.name)
        program = _load_for_executorch(str(model_path), data_path=str(ptd_path))
    else:
        logger.info("No PTD sidecar; loading PTE only")
        program = _load_for_executorch(str(model_path))
    return program


# ============================================================================
# Postprocessing (anchor decode + per-class NMS)
#
# Used by both PyTorch and ExecuTorch paths. Single source of truth -> byte-
# identical detections between the two, so parity reflects model differences
# only, not post-processing differences.
# ============================================================================


def decode_raw_predictions(
    confidences_raw: torch.Tensor,
    locations_raw: torch.Tensor,
    decode_cfg: DecodeConfig,
) -> List[Tuple[Tuple[float, float, float, float], float, int]]:
    """Decode raw SSD heads to (xyxy_normalized, score, class_index).

    Inputs
    ------
    confidences_raw : [1, N, num_classes] logits
    locations_raw   : [1, N, 4] center-form raw locations

    Returns
    -------
    List of ((x1, y1, x2, y2), score, class_index) with coords in [0, 1]
    normalized image space, score in [0, 1], class_index in 1..NUM_VOC_CLASSES.
    """
    if _box_utils is None or _ssd_config is None:
        raise RuntimeError(
            "qfgaohao vision package required for decode; not importable"
        )

    scores = F.softmax(confidences_raw, dim=2)  # [1, N, C]
    boxes_center = _box_utils.convert_locations_to_boxes(
        locations_raw,
        _ssd_config.priors,
        _ssd_config.center_variance,
        _ssd_config.size_variance,
    )
    boxes_corner = _box_utils.center_form_to_corner_form(boxes_center)

    scores = scores[0]  # [N, C]
    boxes = boxes_corner[0]  # [N, 4] xyxy normalized

    out: List[Tuple[Tuple[float, float, float, float], float, int]] = []
    for class_idx in range(1, decode_cfg.num_classes):  # skip background
        class_scores = scores[:, class_idx]
        mask = class_scores > decode_cfg.score_threshold
        if mask.sum().item() == 0:
            continue
        class_boxes = boxes[mask]
        class_scores = class_scores[mask]
        box_scores = torch.cat(
            [class_boxes, class_scores.reshape(-1, 1)], dim=1
        )
        kept = _box_utils.hard_nms(
            box_scores,
            iou_threshold=decode_cfg.iou_threshold,
            top_k=decode_cfg.top_k,
            candidate_size=decode_cfg.candidate_size,
        )
        for row in kept:
            x1, y1, x2, y2, score = (float(v) for v in row.tolist())
            out.append(((x1, y1, x2, y2), score, class_idx))
    return out


def parse_tflite_outputs(
    interp: Any,
    output_details: Sequence[Mapping[str, Any]],
    score_threshold: float,
) -> List[Tuple[Tuple[float, float, float, float], float, int]]:
    """Parse the 4-tensor output of TFLite_Detection_PostProcess.

    Standard TFLite SSD output:
        boxes   [1, N, 4]  normalized yxyx (note: y-first!)
        classes [1, N]     0-indexed class ids (background excluded)
        scores  [1, N]
        count   [1]        valid detection count

    We probe shapes to avoid hard-coding output indices, since order can vary
    by exporter version.
    """
    raw = [interp.get_tensor(d["index"]) for d in output_details]

    boxes_arr = classes_arr = scores_arr = None
    count_val: Optional[int] = None

    for arr in raw:
        if arr.ndim == 3 and arr.shape[-1] == 4:
            boxes_arr = arr
        elif arr.ndim == 1 and arr.size == 1:
            count_val = int(arr[0])
        elif arr.ndim == 2:
            # Either classes or scores; distinguish by dtype + value range.
            flat = arr.reshape(-1).astype(np.float32)
            if flat.size == 0:
                continue
            if (flat == flat.astype(np.int32)).all() and float(flat.max()) <= NUM_VOC_CLASSES:
                classes_arr = arr
            else:
                scores_arr = arr

    if boxes_arr is None or classes_arr is None or scores_arr is None:
        raise RuntimeError(
            "Could not identify TFLite output tensors "
            f"(shapes={[a.shape for a in raw]})"
        )

    num = (
        count_val
        if count_val is not None
        else int(min(boxes_arr.shape[1], classes_arr.shape[1], scores_arr.shape[1]))
    )

    out: List[Tuple[Tuple[float, float, float, float], float, int]] = []
    for i in range(num):
        score = float(scores_arr[0, i])
        if score < score_threshold:
            continue
        y1, x1, y2, x2 = (float(v) for v in boxes_arr[0, i])
        # TFLite class ids are 0-indexed over foreground; convert to 1..NUM_VOC_CLASSES.
        cls_id = int(classes_arr[0, i]) + 1
        out.append(((x1, y1, x2, y2), score, cls_id))
    return out


def detections_to_image_space(
    decoded: Iterable[Tuple[Tuple[float, float, float, float], float, int]],
    image_id: int,
    orig_w: int,
    orig_h: int,
) -> List[Detection]:
    """Convert normalized xyxy detections to image-space xywh Detection records."""
    out: List[Detection] = []
    for (x1, y1, x2, y2), score, cls in decoded:
        x1_p = max(0.0, x1 * orig_w)
        y1_p = max(0.0, y1 * orig_h)
        x2_p = min(float(orig_w), x2 * orig_w)
        y2_p = min(float(orig_h), y2 * orig_h)
        w = max(0.0, x2_p - x1_p)
        h = max(0.0, y2_p - y1_p)
        if w <= 0.0 or h <= 0.0:
            continue
        out.append(
            Detection(
                image_id=image_id,
                bbox_xywh=(x1_p, y1_p, w, h),
                score=score,
                category_id=cls,
            )
        )
    return out


# ============================================================================
# Per-backend evaluation
#
# All three return the same shape:
#   {
#     "detections_coco":      List[Dict]               (for pycocotools)
#     "detections_by_image":  Dict[int, List[Detection]] (for parity analysis)
#     "latency_ms_per_image": List[float]
#     "samples_processed":    int
#   }
# ============================================================================


_EvalResult = Dict[str, Any]


def _empty_result() -> _EvalResult:
    return {
        "detections_coco": [],
        "detections_by_image": {},
        "latency_ms_per_image": [],
        "samples_processed": 0,
    }


def _should_stop(max_samples: Optional[int], processed: int) -> bool:
    return max_samples is not None and processed >= max_samples


def evaluate_pytorch_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    decode_cfg: DecodeConfig,
    max_samples: Optional[int] = None,
) -> _EvalResult:
    """Run PyTorch forward + Python decode + NMS, capturing per-image latency.

    Latency wraps the model forward only (preprocessing already done by dataset,
    decode done after timer release). Matches v1's per-image timing convention.
    """
    logger.info("Starting PyTorch model evaluation")
    result = _empty_result()
    timer = InferenceTimer()

    with torch.no_grad():
        for batch in dataloader:
            if _should_stop(max_samples, result["samples_processed"]):
                break

            input_tensor = batch["input"]
            image_id = int(batch["image_id"])
            orig_w, orig_h = batch["orig_size"]

            start = time.perf_counter()
            with timer:
                confidences, locations = model(input_tensor)
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            decoded = decode_raw_predictions(confidences, locations, decode_cfg)
            dets = detections_to_image_space(decoded, image_id, orig_w, orig_h)

            result["detections_by_image"][image_id] = dets
            result["detections_coco"].extend(d.to_coco() for d in dets)
            result["latency_ms_per_image"].append(elapsed_ms)
            result["samples_processed"] += 1

            if result["samples_processed"] % 100 == 0:
                logger.info(
                    "PyTorch: %d images processed", result["samples_processed"]
                )

    logger.info(
        "PyTorch evaluation complete: %d images, %d total detections",
        result["samples_processed"],
        len(result["detections_coco"]),
    )
    result["timer_stats"] = timer.get_stats()
    return result


def evaluate_tflite_model(
    interp: Any,
    dataloader: DataLoader,
    preprocess_cfg: PreprocessingConfig,
    decode_cfg: DecodeConfig,
    max_samples: Optional[int] = None,
) -> _EvalResult:
    """Run TFLite forward and parse pre-decoded outputs.

    TFLite SSD models bake anchors+decode+NMS into the graph (the reason the
    .tflite is ~23 MB vs the PyTorch weights-only ~13 MB). We therefore do not
    re-decode; we just parse the 4 output tensors with shape probing.
    """
    logger.info("Starting TFLite model evaluation")
    result = _empty_result()
    timer = InferenceTimer()

    input_details = interp.get_input_details()
    output_details = interp.get_output_details()
    inp = input_details[0]
    inp_shape = inp["shape"]
    inp_dtype = inp.get("dtype", np.float32)

    # TFLite SSD typically expects NHWC.
    expect_nhwc = (len(inp_shape) == 4 and inp_shape[-1] == 3)

    for batch in dataloader:
        if _should_stop(max_samples, result["samples_processed"]):
            break

        input_tensor = batch["input"]  # NCHW float32 by default
        image_id = int(batch["image_id"])
        orig_w, orig_h = batch["orig_size"]

        np_input = input_tensor.numpy()
        if expect_nhwc and np_input.shape[1] == 3:
            np_input = np.transpose(np_input, (0, 2, 3, 1))
        np_input = np_input.astype(inp_dtype, copy=False)

        start = time.perf_counter()
        with timer:
            interp.set_tensor(inp["index"], np_input)
            interp.invoke()
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        decoded = parse_tflite_outputs(
            interp, output_details, score_threshold=decode_cfg.score_threshold
        )
        dets = detections_to_image_space(decoded, image_id, orig_w, orig_h)

        result["detections_by_image"][image_id] = dets
        result["detections_coco"].extend(d.to_coco() for d in dets)
        result["latency_ms_per_image"].append(elapsed_ms)
        result["samples_processed"] += 1

        if result["samples_processed"] % 100 == 0:
            logger.info(
                "TFLite: %d images processed", result["samples_processed"]
            )

    logger.info(
        "TFLite evaluation complete: %d images, %d total detections",
        result["samples_processed"],
        len(result["detections_coco"]),
    )
    result["timer_stats"] = timer.get_stats()
    return result


def evaluate_executorch_model(
    program: Any,
    dataloader: DataLoader,
    decode_cfg: DecodeConfig,
    max_samples: Optional[int] = None,
) -> _EvalResult:
    """Run ExecuTorch forward + Python decode + NMS.

    The .pte was exported with is_test=False, so program.forward returns raw
    (confidences, locations); we decode in Python using the SAME function as
    the PyTorch path, guaranteeing identical post-processing.

    forward signature follows ExecuTorch pybindings convention: input is wrapped
    in a tuple, output is a list (see v1 evaluate.py L504).
    """
    logger.info("Starting ExecuTorch model evaluation")
    result = _empty_result()
    timer = InferenceTimer()

    for batch in dataloader:
        if _should_stop(max_samples, result["samples_processed"]):
            break

        input_tensor = batch["input"]
        image_id = int(batch["image_id"])
        orig_w, orig_h = batch["orig_size"]

        start = time.perf_counter()
        with timer:
            outputs = program.forward((input_tensor,))
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        # qfgaohao SSD returns (confidences, locations) when is_test=False.
        confidences, locations = outputs[0], outputs[1]
        if confidences.dim() != 3 or locations.dim() != 3:
            raise RuntimeError(
                f"Unexpected ExecuTorch output shapes: "
                f"confidences={tuple(confidences.shape)} "
                f"locations={tuple(locations.shape)}"
            )

        decoded = decode_raw_predictions(confidences, locations, decode_cfg)
        dets = detections_to_image_space(decoded, image_id, orig_w, orig_h)

        result["detections_by_image"][image_id] = dets
        result["detections_coco"].extend(d.to_coco() for d in dets)
        result["latency_ms_per_image"].append(elapsed_ms)
        result["samples_processed"] += 1

        if result["samples_processed"] % 100 == 0:
            logger.info(
                "ExecuTorch: %d images processed", result["samples_processed"]
            )

    logger.info(
        "ExecuTorch evaluation complete: %d images, %d total detections",
        result["samples_processed"],
        len(result["detections_coco"]),
    )
    result["timer_stats"] = timer.get_stats()
    return result


# ============================================================================
# Metrics: latency
# ============================================================================


def compute_latency_stats(times_ms: Sequence[float]) -> Dict[str, float]:
    """Comprehensive latency statistics over a per-image timing series.

    Returns mean/std/min/max plus p50/p90/p95/p99, total wall time, and FPS
    (computed against the mean of the per-image distribution, the standard
    convention for steady-state throughput estimates).
    """
    if not times_ms:
        return {
            "count": 0, "mean_ms": 0.0, "std_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0,
            "p50_ms": 0.0, "p90_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0,
            "total_s": 0.0, "throughput_fps": 0.0,
        }
    arr = np.asarray(list(times_ms), dtype=np.float64)
    mean = float(arr.mean())
    return {
        "count": int(arr.size),
        "mean_ms": mean,
        "std_ms": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "total_s": float(arr.sum() / 1000.0),
        "throughput_fps": (1000.0 / mean) if mean > 0 else 0.0,
    }


# ============================================================================
# Metrics: detection (mAP via pycocotools, with team's wrapper preferred)
# ============================================================================


def _coco_eval_stats_to_dict(stats: Sequence[float], suffix: str = "") -> Dict[str, float]:
    """Materialize the 12-element COCOeval.stats array into named fields."""
    keys = [
        "mAP_50_95", "mAP_50", "mAP_75",
        "mAP_small", "mAP_medium", "mAP_large",
        "AR_1", "AR_10", "AR_100",
        "AR_small", "AR_medium", "AR_large",
    ]
    return {f"{k}{suffix}": float(stats[i]) for i, k in enumerate(keys)}


def _run_coco_eval_inline(
    detections_coco: Sequence[Mapping[str, Any]],
    gt_coco_path: Path,
    class_subset: Optional[Sequence[int]] = None,
) -> Dict[str, float]:
    """Run pycocotools COCOeval directly when the team's wrapper is unavailable.

    Optionally restricts evaluation to a subset of categories (e.g. VOC-only).
    """
    if not _HAVE_PYCOCOTOOLS:
        logger.warning("pycocotools not installed; skipping inline mAP computation")
        return {}

    coco_gt = COCO(str(gt_coco_path))
    if not detections_coco:
        logger.warning("No detections to evaluate")
        return {}

    coco_dt = coco_gt.loadRes(list(detections_coco))
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    if class_subset is not None:
        coco_eval.params.catIds = list(class_subset)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    out = _coco_eval_stats_to_dict(coco_eval.stats)

    # Per-class AP@0.5 (helpful for diagnosing which classes degrade under quant).
    # precision shape: [T, R, K, A, M] where T=iou thresholds, K=classes.
    try:
        precision = coco_eval.eval["precision"]  # [10, 101, K, 4, 3]
        # IoU=0.5 -> index 0; area=all -> index 0; max_dets=100 -> index 2.
        ap_50_per_class: Dict[str, float] = {}
        cat_ids = coco_eval.params.catIds
        for k, cat_id in enumerate(cat_ids):
            p = precision[0, :, k, 0, 2]
            p = p[p > -1]
            ap = float(p.mean()) if p.size else float("nan")
            cat_name = coco_gt.loadCats([cat_id])[0]["name"]
            ap_50_per_class[cat_name] = ap
        out["per_class_AP_50"] = ap_50_per_class  # type: ignore[assignment]
    except Exception as exc:  # pragma: no cover - per-class is informational
        logger.warning("Per-class AP extraction failed: %s", exc)

    return out


def compute_detection_metrics(
    detections_coco: Sequence[Mapping[str, Any]],
    gt_coco_path: Path,
) -> Dict[str, Any]:
    """Top-level mAP wrapper. Prefers the team's detection_metrics module.

    Returns a dict with class-agnostic and VOC-restricted sections so the report
    template can pick either view.
    """
    if not detections_coco:
        logger.warning("No detections; skipping mAP computation")
        return {"class_agnostic": {}, "voc_restricted": {}}

    # Prefer the team's wrapper. We try the most-likely signatures defensively;
    # if all fail we fall back to inline pycocotools so we never silently drop
    # the primary metric.
    if _HAVE_TEAM_METRICS and _team_calculate_detection_metrics is not None:
        for kwargs in (
            {"predictions": list(detections_coco), "gt_coco_path": str(gt_coco_path)},
            {"detections": list(detections_coco), "ground_truth": str(gt_coco_path)},
            {"predictions": list(detections_coco), "ground_truth_path": str(gt_coco_path)},
        ):
            try:
                metrics = _team_calculate_detection_metrics(**kwargs)  # type: ignore[misc]
                if isinstance(metrics, dict) and metrics:
                    return {"class_agnostic": metrics, "voc_restricted": {}}
            except TypeError:
                continue
            except Exception as exc:
                logger.warning(
                    "Team detection_metrics raised %s; falling back to inline",
                    exc,
                )
                break

    class_agnostic = _run_coco_eval_inline(detections_coco, gt_coco_path)
    voc_restricted = _run_coco_eval_inline(
        detections_coco,
        gt_coco_path,
        class_subset=list(range(1, NUM_VOC_CLASSES + 1)),
    )
    return {"class_agnostic": class_agnostic, "voc_restricted": voc_restricted}


# ============================================================================
# Metrics: parity (PyTorch reference vs ExecuTorch candidate)
# ============================================================================


def _bbox_iou_xywh(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    """IoU between two boxes in xywh (image-space pixels)."""
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0.0 else 0.0


def compute_parity_metrics(
    ref_by_image: Mapping[int, Sequence[Detection]],
    cand_by_image: Mapping[int, Sequence[Detection]],
    iou_match_threshold: float = 0.5,
) -> Dict[str, float]:
    """Detection-level parity between reference (PyTorch) and candidate (ExecuTorch).

    For each image, greedily match candidate detections to reference detections
    by IoU (same class, IoU >= threshold), highest-confidence first. Aggregates:
      * match_rate          fraction of ref dets with a matched candidate
      * extra_rate          fraction of cand dets with no matching ref
      * missed_rate         fraction of ref dets unmatched (= 1 - match_rate)
      * class_agreement     fraction of matched pairs with identical class
                            (== 1.0 by construction here, but kept for sanity)
      * mean_matched_iou    mean IoU across matched pairs
      * mean_abs_score_delta mean |score_ref - score_cand| across matched pairs
      * mean_dets_per_image_ref / _cand
    """
    total_ref = 0
    total_cand = 0
    matched_pairs: List[Tuple[Detection, Detection, float]] = []
    matched_ref_count = 0

    all_image_ids = set(ref_by_image) | set(cand_by_image)
    for image_id in all_image_ids:
        ref_dets = list(ref_by_image.get(image_id, []))
        cand_dets = sorted(
            cand_by_image.get(image_id, []), key=lambda d: -d.score
        )
        total_ref += len(ref_dets)
        total_cand += len(cand_dets)

        ref_used = [False] * len(ref_dets)
        for cd in cand_dets:
            best_idx = -1
            best_iou = iou_match_threshold
            for i, rd in enumerate(ref_dets):
                if ref_used[i] or rd.category_id != cd.category_id:
                    continue
                iou = _bbox_iou_xywh(rd.bbox_xywh, cd.bbox_xywh)
                if iou >= best_iou:
                    best_iou = iou
                    best_idx = i
            if best_idx >= 0:
                ref_used[best_idx] = True
                matched_pairs.append((ref_dets[best_idx], cd, best_iou))
                matched_ref_count += 1

    n_matched = len(matched_pairs)
    n_images = max(1, len(all_image_ids))

    if n_matched > 0:
        mean_iou = sum(p[2] for p in matched_pairs) / n_matched
        mean_score_delta = sum(
            abs(p[0].score - p[1].score) for p in matched_pairs
        ) / n_matched
        class_agreement = sum(
            1 for p in matched_pairs if p[0].category_id == p[1].category_id
        ) / n_matched
    else:
        mean_iou = 0.0
        mean_score_delta = 0.0
        class_agreement = 0.0

    return {
        "iou_match_threshold": float(iou_match_threshold),
        "total_ref_detections": int(total_ref),
        "total_cand_detections": int(total_cand),
        "matched_pairs": int(n_matched),
        "match_rate": (matched_ref_count / total_ref) if total_ref else 0.0,
        "extra_rate": ((total_cand - n_matched) / total_cand) if total_cand else 0.0,
        "missed_rate": ((total_ref - matched_ref_count) / total_ref) if total_ref else 0.0,
        "class_agreement": float(class_agreement),
        "mean_matched_iou": float(mean_iou),
        "mean_abs_score_delta": float(mean_score_delta),
        "mean_dets_per_image_ref": total_ref / n_images,
        "mean_dets_per_image_cand": total_cand / n_images,
    }


# ============================================================================
# Results assembly
# ============================================================================


def _per_image_detections_payload(
    detections_by_image: Mapping[int, Sequence[Detection]],
) -> List[Dict[str, Any]]:
    """Serialize per-image detection lists for the report payload."""
    return [
        {
            "image_id": int(image_id),
            "num_detections": len(dets),
            "detections": [d.to_coco() for d in dets],
        }
        for image_id, dets in sorted(detections_by_image.items())
    ]


def build_baseline_entry(
    backend_name: str,
    model_path: Path,
    eval_result: _EvalResult,
    detection_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "backend": backend_name,
        "model_path": str(model_path),
        "model_size_mb": round(get_file_size_mb(model_path), 4),
        "samples_processed": int(eval_result["samples_processed"]),
        "total_detections": int(len(eval_result["detections_coco"])),
        "metrics": detection_metrics,
        "latency": compute_latency_stats(eval_result["latency_ms_per_image"]),
        "per_image_detections": _per_image_detections_payload(
            eval_result["detections_by_image"]
        ),
    }


def build_executorch_entry(
    pte_path: Path,
    eval_result: _EvalResult,
    detection_metrics: Dict[str, Any],
    parity_metrics: Optional[Dict[str, float]],
    fp32_reference_size_mb: Optional[float],
) -> Dict[str, Any]:
    pte_size = get_file_size_mb(pte_path)
    ptd_path = pte_path.parent / f"{pte_path.stem}_constants.ptd"
    ptd_size = get_file_size_mb(ptd_path) if ptd_path.exists() else None
    combined = pte_size + (ptd_size or 0.0)
    compression_ratio = (
        (fp32_reference_size_mb / combined)
        if (fp32_reference_size_mb and combined > 0)
        else None
    )
    return {
        "model_path": str(pte_path),
        "ptd_path": str(ptd_path) if ptd_path.exists() else None,
        "pte_size_mb": round(pte_size, 4),
        "ptd_size_mb": round(ptd_size, 4) if ptd_size is not None else None,
        "model_size_mb": round(combined, 4),
        "compression_ratio_vs_fp32": (
            round(compression_ratio, 4) if compression_ratio is not None else None
        ),
        "samples_processed": int(eval_result["samples_processed"]),
        "total_detections": int(len(eval_result["detections_coco"])),
        "metrics": detection_metrics,
        "latency": compute_latency_stats(eval_result["latency_ms_per_image"]),
        "parity_vs_pytorch": parity_metrics,
        "per_image_detections": _per_image_detections_payload(
            eval_result["detections_by_image"]
        ),
    }


def initialize_results_dict(
    dataset: VOCDetectionDataset,
    args: argparse.Namespace,
    config: EvalConfig,
) -> Dict[str, Any]:
    """Create the top-level results document. Schema matches what the report expects."""
    return {
        "model_name": "MobileNetV2-SSD-Lite",
        "task": "detection",
        "primary_metric": "mAP_50",
        "timestamp": datetime.now().isoformat(),
        "dataset": {
            "name": "PascalVOC2012-val",
            "voc_root": str(args.voc_root),
            "gt_coco_json": str(args.gt_coco_json),
            "num_images": len(dataset),
            "num_classes": NUM_VOC_CLASSES,
            "class_names": list(VOC_CLASS_NAMES),
            "samples_evaluated": 0,
        },
        "config": config.to_dict(),
        "system_info": get_system_info(),
        "run_args": {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in vars(args).items()
        },
        "tflite_baseline": {},
        "pytorch_baseline": {},
        "executorch_models": [],
    }


# ============================================================================
# Argument parsing
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate MobileNetV2-SSD-Lite across PyTorch / TFLite / ExecuTorch"
    )

    # Data
    parser.add_argument(
        "--voc-root", type=Path, required=True,
        help="Path to VOCdevkit/VOC2012 (contains JPEGImages/, Annotations/)",
    )
    parser.add_argument(
        "--gt-coco-json", type=Path, required=True,
        help="COCO-format VOC val ground-truth JSON (from voc_to_coco.py)",
    )

    # Models
    parser.add_argument(
        "--pytorch-model", type=Path,
        default=Path("../model_sources/MobileNetV2/weights/mobile_net_v2_ssd.pth"),
        help="PyTorch checkpoint (is_test=False pickle preferred; state_dict accepted)",
    )
    parser.add_argument(
        "--tflite-model", type=Path,
        default=Path("../model_sources/MobileNetV2/weights/model.tflite"),
        help="Baseline TFLite model",
    )
    parser.add_argument(
        "--model-dir", type=Path, required=True,
        help="Directory containing exported ExecuTorch .pte files",
    )
    parser.add_argument(
        "--pte-glob", type=str, default="mobile_net_v2_ssd*.pte",
        help="Glob pattern for ExecuTorch artifacts inside --model-dir",
    )

    # Output
    parser.add_argument(
        "--results-dir", type=Path, required=True,
        help="Where to write evaluation_results.json (and optional HTML report)",
    )
    parser.add_argument(
        "--generate-report", action="store_true", default=False,
        help="Generate HTML report after evaluation",
    )

    # Eval control
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Cap samples per backend (default: all)",
    )
    parser.add_argument(
        "--num-workers", type=int, default=4,
        help="DataLoader workers (default: 4)",
    )
    parser.add_argument(
        "--skip-pytorch", action="store_true",
        help="Skip PyTorch baseline (parity will be unavailable)",
    )
    parser.add_argument(
        "--skip-tflite", action="store_true",
        help="Skip TFLite baseline",
    )
    parser.add_argument(
        "--skip-executorch", action="store_true",
        help="Skip ExecuTorch variants",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed (default: 42)",
    )

    # Optional decode overrides (rarely needed; preserved for reviewer reproducibility)
    parser.add_argument(
        "--score-threshold", type=float, default=0.01,
        help="NMS score threshold (default: 0.01 for mAP correctness)",
    )
    parser.add_argument(
        "--iou-threshold", type=float, default=0.45,
        help="NMS IoU threshold (default: 0.45)",
    )
    parser.add_argument(
        "--top-k", type=int, default=200,
        help="NMS top-k per class (default: 200)",
    )

    return parser.parse_args()


# ============================================================================
# Main pipeline
# ============================================================================


def _log_section(title: str) -> None:
    """Print a heavy divider so log files are visually scannable."""
    logger.info("=" * 80)
    logger.info(title)
    logger.info("=" * 80)


def _build_eval_config(args: argparse.Namespace) -> EvalConfig:
    return EvalConfig(
        preprocess=PreprocessingConfig(),
        decode=DecodeConfig(
            iou_threshold=float(args.iou_threshold),
            score_threshold=float(args.score_threshold),
            top_k=int(args.top_k),
        ),
        layout="NCHW",
        seed=int(args.seed),
    )


def main() -> int:
    args = parse_args()

    config = _build_eval_config(args)
    seed_everything(config.seed)

    results_dir: Path = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    model_dir: Path = args.model_dir

    if not _HAVE_QFGAOHAO:
        logger.error(
            "qfgaohao vision package not importable; PyTorch + ExecuTorch eval require it. "
            "Set PYTHONPATH=%s",
            _QFGAOHAO_VISION,
        )
        # TFLite-only mode is still permitted.

    # ------------------------------------------------------------------
    # Dataset & dataloader
    # ------------------------------------------------------------------
    _log_section("LOADING DATASET")
    dataset = VOCDetectionDataset(
        voc_root=args.voc_root,
        gt_coco_json=args.gt_coco_json,
        preprocess=config.preprocess,
        layout=config.layout,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # detection: hardcoded; variable-count outputs preclude batching
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        collate_fn=_identity_collate,
    )

    # ------------------------------------------------------------------
    # Results scaffold
    # ------------------------------------------------------------------
    results = initialize_results_dict(dataset, args, config)

    # We track PyTorch detections for parity analysis against each ExecuTorch variant.
    pytorch_detections_by_image: Optional[Mapping[int, Sequence[Detection]]] = None
    fp32_reference_size_mb: Optional[float] = None

    # ------------------------------------------------------------------
    # PyTorch baseline
    # ------------------------------------------------------------------
    if not args.skip_pytorch:
        _log_section("EVALUATING PYTORCH BASELINE")
        pt_path: Path = args.pytorch_model
        if pt_path.exists():
            logger.info(
                "PyTorch model size: %.2f MB", get_file_size_mb(pt_path)
            )
            try:
                pt_model = load_pytorch_model(pt_path)
                if pt_model is None:
                    raise RuntimeError("load_pytorch_model returned None")

                pt_eval = evaluate_pytorch_model(
                    pt_model, dataloader, config.decode, max_samples=args.max_samples
                )
                pt_metrics = compute_detection_metrics(
                    pt_eval["detections_coco"], args.gt_coco_json
                )
                results["pytorch_baseline"] = build_baseline_entry(
                    "pytorch", pt_path, pt_eval, pt_metrics
                )
                pytorch_detections_by_image = pt_eval["detections_by_image"]

                if results["dataset"]["samples_evaluated"] == 0:
                    results["dataset"]["samples_evaluated"] = pt_eval["samples_processed"]

                ca = pt_metrics.get("class_agnostic", {})
                logger.info(
                    "PyTorch mAP@0.5=%.4f  mAP@0.5:0.95=%.4f",
                    ca.get("mAP_50", float("nan")),
                    ca.get("mAP_50_95", float("nan")),
                )
            except Exception as exc:
                logger.error("PyTorch evaluation failed: %s", exc, exc_info=True)
        else:
            logger.warning("PyTorch model not found at %s", pt_path)

    # ------------------------------------------------------------------
    # TFLite baseline
    # ------------------------------------------------------------------
    if not args.skip_tflite:
        _log_section("EVALUATING TFLITE BASELINE")
        tfl_path: Path = args.tflite_model
        if tfl_path.exists():
            logger.info("TFLite model size: %.2f MB", get_file_size_mb(tfl_path))
            try:
                interp = load_tflite_model(tfl_path)
                if interp is None:
                    raise RuntimeError("load_tflite_model returned None")

                tfl_eval = evaluate_tflite_model(
                    interp, dataloader, config.preprocess, config.decode,
                    max_samples=args.max_samples,
                )
                tfl_metrics = compute_detection_metrics(
                    tfl_eval["detections_coco"], args.gt_coco_json
                )
                results["tflite_baseline"] = build_baseline_entry(
                    "tflite", tfl_path, tfl_eval, tfl_metrics
                )

                if results["dataset"]["samples_evaluated"] == 0:
                    results["dataset"]["samples_evaluated"] = tfl_eval["samples_processed"]

                ca = tfl_metrics.get("class_agnostic", {})
                logger.info(
                    "TFLite mAP@0.5=%.4f  mAP@0.5:0.95=%.4f",
                    ca.get("mAP_50", float("nan")),
                    ca.get("mAP_50_95", float("nan")),
                )
            except Exception as exc:
                logger.error("TFLite evaluation failed: %s", exc, exc_info=True)
        else:
            logger.warning("TFLite model not found at %s", tfl_path)

    # ------------------------------------------------------------------
    # ExecuTorch variants
    # ------------------------------------------------------------------
    if not args.skip_executorch:
        _log_section("EVALUATING EXECUTORCH MODELS")
        if not model_dir.is_dir():
            logger.warning("ExecuTorch model dir not found: %s", model_dir)
        else:
            pte_files = sorted(model_dir.glob(args.pte_glob))
            if not pte_files:
                logger.warning(
                    "No PTE files matching %r in %s; falling back to *.pte",
                    args.pte_glob, model_dir,
                )
                pte_files = sorted(model_dir.glob("*.pte"))

            if not pte_files:
                logger.warning("No PTE files found in %s", model_dir)

            # Identify the FP32 variant as the compression reference. Heuristic:
            # filename contains "fp32" or has no quant suffix.
            for pte_path in pte_files:
                name = pte_path.stem.lower()
                if "fp32" in name or all(
                    tag not in name for tag in ("int8", "_8a", "_pt", "_pc")
                ):
                    fp32_reference_size_mb = (
                        get_file_size_mb(pte_path)
                        + get_file_size_mb(pte_path.parent / f"{pte_path.stem}_constants.ptd")
                    )
                    logger.info(
                        "FP32 reference: %s (%.2f MB combined)",
                        pte_path.name, fp32_reference_size_mb,
                    )
                    break

            for pte_path in pte_files:
                logger.info("Evaluating: %s", pte_path.name)
                try:
                    program = load_executorch_model(pte_path)
                    if program is None:
                        logger.warning("Could not load %s; skipping", pte_path.name)
                        continue

                    et_eval = evaluate_executorch_model(
                        program, dataloader, config.decode,
                        max_samples=args.max_samples,
                    )
                    et_metrics = compute_detection_metrics(
                        et_eval["detections_coco"], args.gt_coco_json
                    )
                    parity = None
                    if pytorch_detections_by_image is not None:
                        parity = compute_parity_metrics(
                            ref_by_image=pytorch_detections_by_image,
                            cand_by_image=et_eval["detections_by_image"],
                            iou_match_threshold=0.5,
                        )
                        logger.info(
                            "Parity vs PyTorch: match_rate=%.3f mean_iou=%.3f "
                            "mean_abs_score_delta=%.4f",
                            parity["match_rate"],
                            parity["mean_matched_iou"],
                            parity["mean_abs_score_delta"],
                        )

                    entry = build_executorch_entry(
                        pte_path=pte_path,
                        eval_result=et_eval,
                        detection_metrics=et_metrics,
                        parity_metrics=parity,
                        fp32_reference_size_mb=fp32_reference_size_mb,
                    )
                    results["executorch_models"].append(entry)

                    if results["dataset"]["samples_evaluated"] == 0:
                        results["dataset"]["samples_evaluated"] = et_eval["samples_processed"]

                    ca = et_metrics.get("class_agnostic", {})
                    logger.info(
                        "%s  mAP@0.5=%.4f  mAP@0.5:0.95=%.4f  mean_latency=%.2f ms",
                        pte_path.name,
                        ca.get("mAP_50", float("nan")),
                        ca.get("mAP_50_95", float("nan")),
                        entry["latency"]["mean_ms"],
                    )
                except Exception as exc:
                    logger.error(
                        "ExecuTorch eval failed for %s: %s",
                        pte_path.name, exc, exc_info=True,
                    )

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    _log_section("SAVING RESULTS")
    output_file = results_dir / "evaluation_results.json"
    save_evaluation_results(results, output_file)
    logger.info("Results saved to %s", output_file)

    # ------------------------------------------------------------------
    # Optional HTML report
    # ------------------------------------------------------------------
    if args.generate_report:
        _log_section("GENERATING HTML REPORT")
        try:
            from evaluation.mobilenetv2.generate_report import (  # noqa: WPS433
                generate_html_for_json,
            )

            report_path = generate_html_for_json(
                output_file, output_dir=results_dir
            )
            if report_path:
                logger.info("HTML report: %s", report_path)
            else:
                logger.warning("HTML report generation returned no path")
        except Exception as exc:
            logger.error("HTML report generation failed: %s", exc, exc_info=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
