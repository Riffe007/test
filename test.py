"""ExecuTorch evaluation script for MobileNetV2-SSD-Lite (object detection).

Mirrors evaluation/mobile_net_v1/evaluate.py (Phase 1 classification template)
but adapted for COCO-style detection evaluation against VOC2012-as-COCO ground
truth. Detection metric functions are embedded module-level rather than living
in evaluation/common/vision_metrics.py — see Scope & Deviations in the
submission README for the rationale and the "future cleanup: lift detection
helpers into vision_metrics.py once a second detection model exists" note.

Inputs are driven by the JSON file alongside this module
(config_mobile_net_v2_ssd.json). CLI flags are limited to runtime overrides
(--max-samples, --skip-pytorch, --skip-executorch, --generate-report) so the
evaluation surface area stays declarative; this is a documented deviation from
V1's argparse-heavy main(), justified by detection requiring 21 class names,
7 decoder params, and named per-quant pte_paths that don't fit CLI flags.

Per-model pipeline:
    qfgaohao MobileNetV2-SSD-Lite (is_test=False)
      -> raw (confidences, locations)
      -> softmax + prior-based decode (vision.utils.box_utils)
      -> per-class NMS (hard_nms, candidate_size=200, iou=0.45)
      -> top-K detections (default 100) in COCO xywh-pixel format
      -> pycocotools COCOeval against the VOC2012-as-COCO ground truth.

The PyTorch baseline and each ExecuTorch .pte share the same Python-side
decoder, which keeps PyTorch-vs-ExecuTorch logits parity meaningful and
ensures all quantized variants are scored on the same post-processing stack.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# Add project root to path — mirrors mobile_net_v1/evaluate.py line 71.
sys.path.append(str(Path(__file__).resolve().parents[2]))

from evaluation.common.utils.inference import InferenceTimer
from evaluation.common.utils.json_io import save_evaluation_results
from evaluation.common.utils.system_info import get_system_info


# ============================================================
# Logging Setup
# ============================================================

def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure logging (matches V1's setup_logging shape)."""
    logging.basicConfig(
        level=level,
        format='[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    return logging.getLogger(__name__)


logger = setup_logging()


# ============================================================
# Detection Metrics (inline; future cleanup: lift to vision_metrics.py
#                    once a second detection model exists)
# ============================================================

# Contract pinned to evaluation/common/metrics_definitions.json
# (vision_metrics.detection block). Any drift here will desync the
# HTML report's metric interpretation thresholds — see
# metrics_loader._INTERPRETATION_PRESETS for the resolved presets.
_DETECTION_METRIC_KEYS: tuple[str, ...] = (
    'mean_precision',
    'mean_recall',
    'mean_f1',
    'mean_iou',
    'mAP_0.5',
    'mAP_0.65',
    'mAP_0.75',
    'mAP_0.5_0.95',
    'num_samples',
)


def calculate_detection_metrics(
    predictions: Sequence[dict],
    labels: Any,
    class_subset: Optional[Iterable[int]] = None,
) -> dict:
    """Compute COCO-style detection metrics.

    Args:
        predictions: List of detection dicts. Each must contain
            ``{'image_id': int, 'category_id': int, 'bbox': [x,y,w,h], 'score': float}``.
            ``bbox`` is in pixel xywh format (COCO convention).
        labels: Path to a COCO-format GT JSON, a dict in that format, or an
            already-loaded ``pycocotools.coco.COCO`` instance.
        class_subset: Optional iterable of category_ids to restrict eval to.

    Returns:
        Dict with the 9 keys defined in ``_DETECTION_METRIC_KEYS``. On error
        or no-predictions fallback, the metric keys are returned as zeros so
        the HTML report renders consistently across variants.
    """
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        logger.warning(
            "pycocotools not available; returning num_samples-only result"
        )
        return {'num_samples': _count_gt_images(labels)}

    gt_coco = _coerce_to_coco(labels, COCO)
    num_samples = len(gt_coco.getImgIds())

    if not predictions:
        logger.warning("No predictions provided; returning zeroed metrics")
        return _empty_detection_result(num_samples)

    try:
        coco_dt = gt_coco.loadRes(list(predictions))
    except Exception as e:
        logger.error(f"Failed to load detections into pycocotools: {e}")
        return _empty_detection_result(num_samples)

    coco_eval = COCOeval(gt_coco, coco_dt, 'bbox')
    if class_subset is not None:
        coco_eval.params.catIds = list(class_subset)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    precision_05 = _mean_precision_at_iou_05(coco_eval)
    recall_avg = _mean_recall_across_ious(coco_eval)
    f1 = _harmonic_mean(precision_05, recall_avg)
    mean_iou = _mean_iou_from_cocoeval(coco_eval, gt_coco)

    result = {
        'mean_precision': float(precision_05),
        'mean_recall': float(recall_avg),
        'mean_f1': float(f1),
        'mean_iou': float(mean_iou),
        'mAP_0.5': float(_ap_at_iou(coco_eval, 0.5)),
        'mAP_0.65': float(_ap_at_iou(coco_eval, 0.65)),
        'mAP_0.75': float(_ap_at_iou(coco_eval, 0.75)),
        'mAP_0.5_0.95': float(coco_eval.stats[0]),
        'num_samples': num_samples,
    }

    missing = set(_DETECTION_METRIC_KEYS) - set(result.keys())
    if missing:
        raise RuntimeError(
            f"Detection metrics contract violation: missing keys {missing}"
        )
    return result


def generate_per_image_detection_results(
    predictions: Sequence[dict],
    gt_coco_or_path: Any,
    latencies: Optional[Sequence[float]] = None,
) -> list[dict]:
    """Build per-image rows for the HTML report.

    Returns rows ordered by ascending image_id, each row starting with
    ``image_index`` (0-based dense index) for stable HTML rendering.
    """
    try:
        from pycocotools.coco import COCO
    except ImportError:
        return []

    gt_coco = _coerce_to_coco(gt_coco_or_path, COCO)

    # Group predictions by image_id.
    preds_by_img: dict[int, list[dict]] = {}
    for det in predictions:
        img_id = det.get('image_id')
        if img_id is None:
            continue
        preds_by_img.setdefault(int(img_id), []).append(det)

    rows: list[dict] = []
    image_ids = sorted(gt_coco.getImgIds())
    for idx, image_id in enumerate(image_ids):
        preds = preds_by_img.get(image_id, [])
        gt_ann_ids = gt_coco.getAnnIds(imgIds=[image_id])
        row = {
            'image_index': idx,
            'image_id': image_id,
            'num_predictions': len(preds),
            'num_gt': len(gt_ann_ids),
            'max_score': float(
                max((p.get('score', 0.0) for p in preds), default=0.0)
            ),
        }
        if latencies is not None and idx < len(latencies):
            row['latency_ms'] = float(latencies[idx])
        rows.append(row)
    return rows


# ---------- Detection helpers (underscore-prefixed; not part of public API) ----------

def _coerce_to_coco(labels: Any, COCO) -> Any:
    """Accept a path, dict, or COCO instance; return a COCO instance."""
    if hasattr(labels, 'getImgIds') and hasattr(labels, 'getAnnIds'):
        return labels
    if isinstance(labels, (str, Path)):
        return COCO(str(labels))
    if isinstance(labels, dict):
        import tempfile
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        ) as tmp:
            json.dump(labels, tmp)
            tmp_path = tmp.name
        return COCO(tmp_path)
    raise TypeError(f"Cannot coerce {type(labels)} to pycocotools.COCO")


def _count_gt_images(labels: Any) -> int:
    """Best-effort GT image count without requiring pycocotools."""
    try:
        if hasattr(labels, 'getImgIds'):
            return len(labels.getImgIds())
        if isinstance(labels, (str, Path)):
            with open(labels) as f:
                data = json.load(f)
            return len(data.get('images', []))
        if isinstance(labels, dict):
            return len(labels.get('images', []))
    except Exception:
        pass
    return 0


def _empty_detection_result(num_samples: int) -> dict:
    """Zeroed result used on no-predictions/error fallback."""
    return {
        'mean_precision': 0.0,
        'mean_recall': 0.0,
        'mean_f1': 0.0,
        'mean_iou': 0.0,
        'mAP_0.5': 0.0,
        'mAP_0.65': 0.0,
        'mAP_0.75': 0.0,
        'mAP_0.5_0.95': 0.0,
        'num_samples': num_samples,
    }


def _ap_at_iou(coco_eval, iou_threshold: float) -> float:
    """Mean AP at a specific IoU threshold (mean over recall thresholds and
    classes), pulled from COCOeval.eval['precision'] of shape
    [T, R, K, A, M] where T=iou_thrs, R=recall_thrs, K=classes, A=areas
    (0=all), M=max_dets (-1=last). argmin against the iouThrs grid handles
    non-standard thresholds like 0.65.
    """
    if not hasattr(coco_eval, 'eval') or 'precision' not in coco_eval.eval:
        return 0.0
    iou_thrs = coco_eval.params.iouThrs
    t_idx = int(np.argmin(np.abs(iou_thrs - iou_threshold)))
    precision = coco_eval.eval['precision'][t_idx, :, :, 0, -1]  # [R, K]
    valid = precision[precision > -1]
    if valid.size == 0:
        return 0.0
    return float(np.mean(valid))


def _mean_precision_at_iou_05(coco_eval) -> float:
    """Mean precision averaged over recall thresholds, at IoU=0.5
    (binary-preset convention from metrics_definitions.json)."""
    return _ap_at_iou(coco_eval, 0.5)


def _mean_recall_across_ious(coco_eval) -> float:
    """Mean recall averaged across the full IoU sweep [0.5:0.95].

    Asymmetry with mean_precision (which is at IoU=0.5 only) is intentional:
    recall@0.5 saturates quickly on quantized models and obscures
    localization degradation, so we use the sweep as the more stable summary.
    """
    if not hasattr(coco_eval, 'eval') or 'recall' not in coco_eval.eval:
        return 0.0
    recall = coco_eval.eval['recall'][:, :, 0, -1]  # [T, K]
    valid = recall[recall > -1]
    if valid.size == 0:
        return 0.0
    return float(np.mean(valid))


def _harmonic_mean(a: float, b: float) -> float:
    """Harmonic mean with zero-safety (returns 0 when both inputs are 0)."""
    if a + b <= 0:
        return 0.0
    return 2.0 * a * b / (a + b)


def _mean_iou_from_cocoeval(coco_eval, gt_coco) -> float:
    """Average best-match IoU per GT annotation across all (img_id, cat_id)
    cells in ``coco_eval.ious``. GTs with no detection at their cell count
    in the denominator but contribute 0 to the numerator.
    """
    if not hasattr(coco_eval, 'ious') or not coco_eval.ious:
        return 0.0

    total_iou = 0.0
    total_gt = 0
    img_ids = coco_eval.params.imgIds
    cat_ids = coco_eval.params.catIds

    for img_id in img_ids:
        for cat_id in cat_ids:
            ann_ids = gt_coco.getAnnIds(imgIds=[img_id], catIds=[cat_id])
            num_gt_here = len(ann_ids)
            if num_gt_here == 0:
                continue
            ious = coco_eval.ious.get((img_id, cat_id), [])
            if len(ious) == 0:
                total_gt += num_gt_here
                continue
            ious_arr = np.asarray(ious)
            if ious_arr.size == 0:
                total_gt += num_gt_here
                continue
            # rows=detections, cols=GTs; best per GT = max along axis=0.
            if ious_arr.ndim == 2:
                best_per_gt = ious_arr.max(axis=0)
            else:
                best_per_gt = ious_arr
            total_iou += float(best_per_gt.sum())
            total_gt += num_gt_here

    if total_gt == 0:
        return 0.0
    return total_iou / total_gt


# ============================================================
# Custom Dataset for VOC Detection (flat directory structure)
# ============================================================

class VOCDetectionDataset(Dataset):
    """VOC-as-COCO detection dataset.

    Reads images keyed off ``file_name`` in the GT COCO JSON, looking under
    both the data_path root and ``data_path/JPEGImages/`` (the standard VOC
    layout). Returns ``(image_tensor [3, H, W] float32 in [0,1], image_id,
    original_width, original_height)``. Normalization is deferred to the
    evaluation functions so the same dataset can feed both the PyTorch and
    ExecuTorch paths without diverging tensor pipelines.
    """

    def __init__(self, data_path: Any, gt_coco, image_size: int = 300):
        self.data_path = Path(data_path)
        self.gt_coco = gt_coco
        self.image_size = image_size
        self.image_ids = sorted(gt_coco.getImgIds())
        self.image_info = {
            img_id: gt_coco.loadImgs([img_id])[0]
            for img_id in self.image_ids
        }

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        info = self.image_info[image_id]
        file_name = info['file_name']

        # VOC2012-as-COCO emits bare filenames (e.g. "2008_000008.jpg") that
        # live under JPEGImages/. Try the root first in case the adapter
        # included the subdir.
        candidates = [
            self.data_path / file_name,
            self.data_path / 'JPEGImages' / file_name,
        ]
        img_path = next((p for p in candidates if p.exists()), None)
        if img_path is None:
            raise FileNotFoundError(
                f"Image {file_name} not found in {self.data_path} or "
                f"{self.data_path / 'JPEGImages'}"
            )

        pil_image = Image.open(img_path).convert('RGB')
        original_width, original_height = pil_image.size

        # Resize to the model's expected input. Done here (not in a
        # torchvision transform) so the resized dimensions are deterministic
        # and aligned with the priors' implicit coordinate space.
        pil_image = pil_image.resize(
            (self.image_size, self.image_size), Image.BILINEAR
        )

        arr = np.asarray(pil_image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()

        return tensor, image_id, original_width, original_height


# ============================================================
# Dataset Loading
# ============================================================

def _detection_collate_fn(batch):
    """Stack image tensors; keep image_ids and original sizes as lists.

    DataLoader's default collate would try to tensorize image_ids and sizes,
    which is unnecessary and brittle when batch_size=1.
    """
    images = torch.stack([item[0] for item in batch], dim=0)
    image_ids = [item[1] for item in batch]
    widths = [item[2] for item in batch]
    heights = [item[3] for item in batch]
    return images, image_ids, widths, heights


def create_dataloader(
    data_path: Any,
    gt_coco_json_path: Any,
    image_size: int = 300,
    batch_size: int = 1,
    num_workers: int = 4,
):
    """Build the VOC-as-COCO dataloader.

    Signature deviates from mobile_net_v1's ``create_dataloader`` (which is
    classification-shaped: ``data_path, batch_size, num_workers``) because
    detection requires GT enumeration up front to establish stable
    image_id ordering for COCOeval. This deviation is documented in the
    submission README under Scope & Deviations.
    """
    try:
        from pycocotools.coco import COCO
    except ImportError as e:
        raise ImportError(
            "pycocotools is required for detection evaluation. "
            "Install with: pip install pycocotools"
        ) from e

    logger.info(f"Loading GT COCO annotations from {gt_coco_json_path}")
    gt_coco = COCO(str(gt_coco_json_path))

    logger.info(f"Loading VOC images from {data_path}")
    dataset = VOCDetectionDataset(
        data_path, gt_coco, image_size=image_size
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=_detection_collate_fn,
    )

    logger.info(
        f"Dataset loaded: {len(dataset)} images, "
        f"{len(gt_coco.getCatIds())} categories"
    )
    return dataloader, dataset, gt_coco


# ============================================================
# Model Loading
# ============================================================

def get_file_size_mb(path: Any) -> float:
    """File size in megabytes; returns 0 for missing files (mirrors V1)."""
    path_obj = Path(path)
    if path_obj.exists():
        return path_obj.stat().st_size / (1024 * 1024)
    return 0


def _ensure_qfgaohao_on_path(qfgaohao_repo_path: Any) -> None:
    """Idempotently prepend the qfgaohao repo to sys.path."""
    repo_path = Path(qfgaohao_repo_path).resolve()
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))


def load_pytorch_model(
    model_path: Any,
    qfgaohao_repo_path: Any,
    num_classes: int = 21,
    device: str = 'cpu',
):
    """Load a qfgaohao MobileNetV2-SSD-Lite checkpoint.

    Handles the two common pickle shapes:
      - Full ``nn.Module`` (what the user has at
        ``model_sources/MobileNetV2/weights/mobile_net_v2_ssd.pth``): returned
        as-is after ``.eval()``.
      - state_dict: load into a freshly constructed module.

    ``is_test`` stays False so ``forward()`` returns raw
    ``(confidences, locations)``; decoding is done Python-side in
    ``decode_ssd_predictions`` for symmetry with the ExecuTorch path.
    """
    logger.info(f"Loading PyTorch model from {model_path}")
    _ensure_qfgaohao_on_path(qfgaohao_repo_path)

    try:
        try:
            ckpt = torch.load(
                str(model_path), map_location=device, weights_only=False
            )
        except TypeError:
            # Older torch without weights_only kwarg.
            ckpt = torch.load(str(model_path), map_location=device)

        if isinstance(ckpt, torch.nn.Module):
            model = ckpt
        elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
            from vision.ssd.mobilenetv2_ssd_lite import (
                create_mobilenetv2_ssd_lite,
            )
            model = create_mobilenetv2_ssd_lite(num_classes, is_test=False)
            model.load_state_dict(ckpt['state_dict'])
        else:
            logger.error(
                "Unsupported checkpoint format — expected full nn.Module "
                "or dict with 'state_dict'."
            )
            return None

        model.eval()
        return model
    except Exception as e:
        logger.error(f"Failed to load PyTorch model: {e}")
        return None


def load_executorch_model(model_path: Any):
    """Load an ExecuTorch .pte; mirrors mobile_net_v1/evaluate.py line 300."""
    logger.info(f"Loading ExecuTorch model from {model_path}")
    try:
        from executorch.extension.pybindings.portable_lib import (
            _load_for_executorch,
        )
    except ImportError:
        logger.error(
            "ExecuTorch not available. Install executorch to evaluate "
            ".pte models."
        )
        return None

    model_path = Path(model_path)
    ptd_path = model_path.with_suffix('.pte').parent / (
        f"{model_path.stem}_constants.ptd"
    )

    try:
        if ptd_path.exists():
            logger.info(f"Found PTD file: {ptd_path.name}")
            program = _load_for_executorch(
                str(model_path), data_path=str(ptd_path)
            )
        else:
            logger.info("No PTD file found, loading PTE only")
            program = _load_for_executorch(str(model_path))
        logger.info(f"ExecuTorch model loaded: {model_path.name}")
        return program
    except Exception as e:
        logger.error(f"Failed to load ExecuTorch model: {e}")
        return None


# ============================================================
# SSD Post-processing (priors-based decode + per-class NMS)
# ============================================================

def _load_priors_and_boxutils(qfgaohao_repo_path: Any):
    """Import qfgaohao's MobileNetV1-SSD prior config and box_utils.

    MobileNetV2-SSD-Lite in qfgaohao's repo shares the same prior
    configuration as MobileNetV1-SSD (300x300 input, 6 feature maps,
    matching aspect ratios), so importing ``mobilenetv1_ssd_config`` for
    the priors is correct. If this assumption ever breaks (e.g., the repo
    grows a dedicated mv2 config), this is the single point of failure.
    """
    _ensure_qfgaohao_on_path(qfgaohao_repo_path)
    from vision.ssd.config import mobilenetv1_ssd_config
    from vision.utils import box_utils
    return mobilenetv1_ssd_config.priors, box_utils


def decode_ssd_predictions(
    confidences: torch.Tensor,
    locations: torch.Tensor,
    priors: torch.Tensor,
    box_utils,
    original_width: int,
    original_height: int,
    decoder_config: dict,
    num_classes: int = 21,
    image_id: Optional[int] = None,
) -> list[dict]:
    """Apply softmax + prior-based decode + per-class NMS to one image's
    raw SSD output. Returns COCO-format detection dicts in pixel xywh.

    Args:
        confidences: ``[num_priors, num_classes]`` raw logits (no batch dim).
        locations: ``[num_priors, 4]`` center-form deltas (no batch dim).
        priors: ``[num_priors, 4]`` center-form prior boxes, normalized.
        box_utils: imported ``vision.utils.box_utils`` module.
        original_width / original_height: ints — for denormalizing predictions
            into the same pixel coordinate space as the GT bboxes.
        decoder_config: dict with keys ``score_threshold``,
            ``nms_iou_threshold``, ``max_detections_per_image``,
            ``candidate_size``, ``center_variance``, ``size_variance``.
        num_classes: total class count including background (21 for VOC).
        image_id: image_id to attach to every output detection.
    """
    score_threshold = decoder_config.get('score_threshold', 0.01)
    iou_threshold = decoder_config.get('nms_iou_threshold', 0.45)
    candidate_size = decoder_config.get('candidate_size', 200)
    max_detections = decoder_config.get('max_detections_per_image', 100)
    center_variance = decoder_config.get('center_variance', 0.1)
    size_variance = decoder_config.get('size_variance', 0.2)

    confidences = confidences.detach().cpu()
    locations = locations.detach().cpu()

    scores = F.softmax(confidences, dim=1)

    # Prior-based decode: center-form normalized boxes.
    boxes_center = box_utils.convert_locations_to_boxes(
        locations, priors, center_variance, size_variance
    )
    boxes_corner = box_utils.center_form_to_corner_form(boxes_center)

    detections: list[dict] = []
    img_id_int = int(image_id) if image_id is not None else 0

    # Class 0 is background — skip it.
    for class_id in range(1, num_classes):
        class_scores = scores[:, class_id]
        mask = class_scores > score_threshold
        if not mask.any():
            continue

        class_boxes = boxes_corner[mask]
        class_scores_filtered = class_scores[mask]

        # Stack [boxes | score] for hard_nms (qfgaohao's expected shape).
        box_probs = torch.cat(
            [class_boxes, class_scores_filtered.unsqueeze(1)], dim=1
        )
        box_probs = box_utils.hard_nms(
            box_probs,
            iou_threshold=iou_threshold,
            top_k=-1,
            candidate_size=candidate_size,
        )

        for det in box_probs:
            x1 = float(det[0]) * original_width
            y1 = float(det[1]) * original_height
            x2 = float(det[2]) * original_width
            y2 = float(det[3]) * original_height
            score = float(det[4])
            # COCO bbox: [x, y, width, height] in pixels.
            width = max(x2 - x1, 0.0)
            height = max(y2 - y1, 0.0)
            detections.append({
                'image_id': img_id_int,
                'category_id': int(class_id),
                'bbox': [x1, y1, width, height],
                'score': score,
            })

    detections.sort(key=lambda d: d['score'], reverse=True)
    return detections[:max_detections]


# ============================================================
# Evaluation Functions
# ============================================================

def _qfgaohao_normalize(
    image_tensor: torch.Tensor,
    normalize_mean: Sequence[float],
    normalize_std: Sequence[float],
) -> torch.Tensor:
    """Apply qfgaohao's ``(x - 127) / 128`` normalization.

    With ``image_tensor`` in [0, 1] float, this is equivalent to
    ``(255*x - 127) / 128``, reproduced via ToTensor + Normalize(mean=127/255,
    std=128/255). The config supplies mean=0.4980 (=127/255) and
    std=0.5020 (=128/255), reproducing qfgaohao's original within ~4e-5.
    """
    mean = torch.tensor(
        list(normalize_mean), dtype=image_tensor.dtype
    ).view(1, 3, 1, 1)
    std = torch.tensor(
        list(normalize_std), dtype=image_tensor.dtype
    ).view(1, 3, 1, 1)
    return (image_tensor - mean) / std


def evaluate_pytorch_model(
    model,
    dataloader,
    priors: torch.Tensor,
    box_utils,
    decoder_config: dict,
    normalize_mean: Sequence[float],
    normalize_std: Sequence[float],
    num_classes: int = 21,
    max_samples: Optional[int] = None,
) -> dict:
    """Evaluate the qfgaohao PyTorch model with Python-side decoding."""
    logger.info("Starting PyTorch model evaluation...")

    all_predictions: list[dict] = []
    timer = InferenceTimer()
    samples_processed = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_samples is not None and samples_processed >= max_samples:
                break

            images, image_ids, widths, heights = batch
            images_norm = _qfgaohao_normalize(
                images, normalize_mean, normalize_std
            )

            with timer:
                outputs = model(images_norm)

            # qfgaohao with is_test=False returns (confidences, locations).
            if isinstance(outputs, (tuple, list)) and len(outputs) == 2:
                confidences_batch, locations_batch = outputs
            else:
                logger.error(
                    f"Unexpected PyTorch output shape: {type(outputs).__name__}; "
                    f"expected (confidences, locations) tuple"
                )
                continue

            for i in range(images.shape[0]):
                if (max_samples is not None
                        and samples_processed >= max_samples):
                    break
                dets = decode_ssd_predictions(
                    confidences_batch[i],
                    locations_batch[i],
                    priors,
                    box_utils,
                    widths[i],
                    heights[i],
                    decoder_config,
                    num_classes=num_classes,
                    image_id=image_ids[i],
                )
                all_predictions.extend(dets)
                samples_processed += 1

            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Processed {samples_processed} samples...")

    logger.info(
        f"PyTorch evaluation complete: {samples_processed} samples, "
        f"{len(all_predictions)} total detections"
    )
    return {
        'predictions': all_predictions,
        'samples_processed': samples_processed,
        'latency': timer.get_stats(),
    }


def evaluate_executorch_model(
    program,
    dataloader,
    priors: torch.Tensor,
    box_utils,
    decoder_config: dict,
    normalize_mean: Sequence[float],
    normalize_std: Sequence[float],
    num_classes: int = 21,
    max_samples: Optional[int] = None,
) -> dict:
    """Evaluate an ExecuTorch .pte with the same decoder as the PyTorch path."""
    logger.info("Starting ExecuTorch model evaluation...")

    all_predictions: list[dict] = []
    timer = InferenceTimer()
    samples_processed = 0

    for batch_idx, batch in enumerate(dataloader):
        if max_samples is not None and samples_processed >= max_samples:
            break

        images, image_ids, widths, heights = batch
        images_norm = _qfgaohao_normalize(
            images, normalize_mean, normalize_std
        )

        for i in range(images.shape[0]):
            if max_samples is not None and samples_processed >= max_samples:
                break

            img_input = images_norm[i:i + 1]

            with timer:
                outputs = program.forward((img_input,))

            # is_test=False export emits (confidences, locations); both
            # paths must agree for parity to be meaningful.
            if isinstance(outputs, (tuple, list)) and len(outputs) >= 2:
                confidences = outputs[0]
                locations = outputs[1]
            else:
                logger.error(
                    f"Unexpected ExecuTorch output shape: "
                    f"{type(outputs).__name__}, "
                    f"len="
                    f"{len(outputs) if hasattr(outputs, '__len__') else '?'}"
                )
                samples_processed += 1
                continue

            # Strip batch dim if the PTE preserved it.
            if confidences.dim() == 3:
                confidences = confidences[0]
            if locations.dim() == 3:
                locations = locations[0]

            dets = decode_ssd_predictions(
                confidences,
                locations,
                priors,
                box_utils,
                widths[i],
                heights[i],
                decoder_config,
                num_classes=num_classes,
                image_id=image_ids[i],
            )
            all_predictions.extend(dets)
            samples_processed += 1

        if (batch_idx + 1) % 10 == 0:
            logger.info(f"Processed {samples_processed} samples...")

    logger.info(
        f"ExecuTorch evaluation complete: {samples_processed} samples, "
        f"{len(all_predictions)} total detections"
    )
    return {
        'predictions': all_predictions,
        'samples_processed': samples_processed,
        'latency': timer.get_stats(),
    }


# ============================================================
# Config helpers
# ============================================================

def _resolve_path(path_str: Any, config_dir: Path) -> Path:
    """Resolve a config path relative to the config file's directory.

    Absolute paths and ``~``-paths pass through; relative paths anchor at
    the config file's parent so the toolkit works regardless of CWD.
    """
    p = Path(str(path_str)).expanduser()
    if p.is_absolute():
        return p
    return (config_dir / p).resolve()


def _load_config(config_path: Path) -> dict:
    """Read and minimally validate the V2-SSD config file."""
    with open(config_path) as f:
        config = json.load(f)

    required_top = ['model', 'evaluation']
    missing = [k for k in required_top if k not in config]
    if missing:
        raise ValueError(f"Config missing required top-level keys: {missing}")

    eval_block = config['evaluation']
    for required_sub in ('dataset', 'decoder', 'output', 'backends'):
        if required_sub not in eval_block:
            raise ValueError(
                f"Config 'evaluation' block missing required key: {required_sub}"
            )

    if not eval_block.get('enabled', True):
        logger.warning(
            "evaluation.enabled is False — running anyway because the "
            "evaluator was invoked explicitly"
        )
    return config


# ============================================================
# Main Evaluation Pipeline
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Evaluate MobileNetV2-SSD-Lite Detection Model'
    )
    parser.add_argument(
        '--config', type=str,
        default=str(Path(__file__).parent / 'config_mobile_net_v2_ssd.json'),
        help='Path to the configuration JSON file',
    )
    parser.add_argument(
        '--max-samples', type=int, default=None,
        help='Maximum number of samples to evaluate (default: all)',
    )
    parser.add_argument(
        '--skip-pytorch', action='store_true',
        help='Skip PyTorch baseline evaluation',
    )
    parser.add_argument(
        '--skip-executorch', action='store_true',
        help='Skip ExecuTorch model evaluation',
    )
    parser.add_argument(
        '--generate-report', action='store_true', default=True,
        help='Generate HTML report after evaluation',
    )
    args = parser.parse_args()

    # ----- Load config -----
    config_path = Path(args.config).resolve()
    config_dir = config_path.parent
    logger.info(f"Loading config from {config_path}")
    config = _load_config(config_path)

    model_cfg = config['model']
    eval_cfg = config['evaluation']
    dataset_cfg = eval_cfg['dataset']
    decoder_cfg = eval_cfg['decoder']
    output_cfg = eval_cfg['output']
    backends_cfg = eval_cfg['backends']
    report_cfg = eval_cfg.get('report', {})

    # ----- Resolve paths -----
    data_path = _resolve_path(dataset_cfg['data_path'], config_dir)
    gt_coco_json = _resolve_path(dataset_cfg['gt_coco_json'], config_dir)
    pytorch_model_path = _resolve_path(
        model_cfg['model_path'], config_dir
    )
    qfgaohao_repo_path = _resolve_path(
        model_cfg['model_sources_repo_path'], config_dir
    )
    results_dir = _resolve_path(output_cfg['results_dir'], config_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ----- Detection params -----
    num_classes = int(model_cfg.get('num_classes', 21))
    class_names = list(dataset_cfg.get('class_names', []))
    normalize_mean = list(
        model_cfg.get('normalize_mean', [0.4980, 0.4980, 0.4980])
    )
    normalize_std = list(
        model_cfg.get('normalize_std', [0.5020, 0.5020, 0.5020])
    )
    image_size = int(decoder_cfg.get('image_size', 300))
    batch_size = int(dataset_cfg.get('batch_size', 1))
    num_workers = int(dataset_cfg.get('num_workers', 4))
    primary_metric = eval_cfg.get('primary_metric', 'mAP_0.5_0.95')

    # ----- Build dataloader -----
    dataloader, dataset, gt_coco = create_dataloader(
        data_path,
        gt_coco_json,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # ----- Load priors + box_utils once -----
    priors, box_utils = _load_priors_and_boxutils(qfgaohao_repo_path)

    # ----- System info & results dict -----
    logger.info("Collecting system information...")
    system_info = get_system_info()

    # Top-level primary_metric threads through metrics_loader's
    # get_result_primary_metric() priority chain so the HTML report
    # interprets mAP_0.5_0.95 (not task_config's default of mean_f1).
    results: dict = {
        'model_name': 'MobileNetV2-SSD-Lite',
        'task': 'detection',
        'primary_metric': primary_metric,
        'timestamp': datetime.now().isoformat(),
        'dataset': {
            'path': str(data_path),
            'gt_coco_json': str(gt_coco_json),
            'num_samples': len(dataset),
            'num_classes': num_classes,
            'class_names': class_names,
            'samples_evaluated': 0,
        },
        'system_info': system_info,
        'pytorch_baseline': {},
        'executorch_models': [],
    }

    # ============================================================
    # Evaluate PyTorch Baseline
    # ============================================================
    if (not args.skip_pytorch
            and backends_cfg.get('include_pytorch_baseline', True)):
        logger.info("=" * 80)
        logger.info("EVALUATING PYTORCH BASELINE")
        logger.info("=" * 80)

        if pytorch_model_path.exists():
            pytorch_model_size_mb = get_file_size_mb(str(pytorch_model_path))
            logger.info(
                f"PyTorch Model Size: {pytorch_model_size_mb:.2f} MB"
            )
            try:
                pt_model = load_pytorch_model(
                    pytorch_model_path,
                    qfgaohao_repo_path,
                    num_classes=num_classes,
                )
                if pt_model is not None:
                    pt_results = evaluate_pytorch_model(
                        pt_model,
                        dataloader,
                        priors,
                        box_utils,
                        decoder_cfg,
                        normalize_mean,
                        normalize_std,
                        num_classes=num_classes,
                        max_samples=args.max_samples,
                    )
                    pt_metrics = calculate_detection_metrics(
                        pt_results['predictions'], gt_coco
                    )
                    per_image_results = (
                        generate_per_image_detection_results(
                            pt_results['predictions'], gt_coco
                        )
                    )
                    results['pytorch_baseline'] = {
                        'model_path': str(pytorch_model_path),
                        'model_size_mb': round(pytorch_model_size_mb, 2),
                        'metrics': pt_metrics,
                        'latency': pt_results['latency'],
                        'per_image_results': per_image_results,
                    }
                    samples_evaluated = pt_results['samples_processed']
                    if results['dataset']['samples_evaluated'] == 0:
                        results['dataset']['samples_evaluated'] = (
                            samples_evaluated
                        )
                    logger.info(
                        f"PyTorch {primary_metric}: "
                        f"{pt_metrics.get(primary_metric, 0.0):.4f}"
                    )
                    logger.info(
                        f"PyTorch mAP@0.5: "
                        f"{pt_metrics.get('mAP_0.5', 0.0):.4f}"
                    )
            except Exception as e:
                logger.error(
                    f"Failed to evaluate PyTorch model: {e}",
                    exc_info=True,
                )
        else:
            logger.warning(
                f"PyTorch model not found at {pytorch_model_path}"
            )

    # ============================================================
    # Evaluate ExecuTorch Models
    # ============================================================
    if not args.skip_executorch:
        et_model_entries = backends_cfg.get('executorch_models', [])
        valid_et_entries = []
        for entry in et_model_entries:
            if 'pte_path' not in entry:
                logger.warning(
                    f"Skipping malformed executorch_models entry "
                    f"(missing pte_path): {entry}"
                )
                continue
            pte_path = _resolve_path(entry['pte_path'], config_dir)
            if not pte_path.exists():
                logger.warning(
                    f"PTE not found for '{entry.get('name', '?')}': "
                    f"{pte_path}; skipping this variant"
                )
                continue
            valid_et_entries.append((entry, pte_path))

        if valid_et_entries:
            logger.info("=" * 80)
            logger.info(
                f"EVALUATING {len(valid_et_entries)} EXECUTORCH MODEL(S)"
            )
            logger.info("=" * 80)

            for entry, pte_file in valid_et_entries:
                variant_name = entry.get('name', pte_file.stem)
                logger.info(f"\nEvaluating: {variant_name} ({pte_file.name})")

                program = load_executorch_model(pte_file)
                if program is None:
                    continue

                try:
                    et_results = evaluate_executorch_model(
                        program,
                        dataloader,
                        priors,
                        box_utils,
                        decoder_cfg,
                        normalize_mean,
                        normalize_std,
                        num_classes=num_classes,
                        max_samples=args.max_samples,
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to evaluate {variant_name}: {e}",
                        exc_info=True,
                    )
                    continue

                et_metrics = calculate_detection_metrics(
                    et_results['predictions'], gt_coco
                )

                pte_size_mb = get_file_size_mb(str(pte_file))
                ptd_file = pte_file.parent / (
                    f"{pte_file.stem}_constants.ptd"
                )
                ptd_size_mb = (
                    get_file_size_mb(str(ptd_file))
                    if ptd_file.exists() else None
                )
                model_size_mb = pte_size_mb + (ptd_size_mb or 0)

                samples_evaluated = et_results['samples_processed']
                if results['dataset']['samples_evaluated'] == 0:
                    results['dataset']['samples_evaluated'] = (
                        samples_evaluated
                    )

                model_entry = {
                    'name': variant_name,
                    'is_baseline': bool(entry.get('is_baseline', False)),
                    'model_path': str(pte_file),
                    'model_size_mb': round(model_size_mb, 2),
                    'pte_size_mb': round(pte_size_mb, 2),
                    'metrics': et_metrics,
                    'latency': et_results['latency'],
                    'per_image_results': (
                        generate_per_image_detection_results(
                            et_results['predictions'], gt_coco
                        )
                    ),
                }
                if ptd_size_mb is not None:
                    model_entry['ptd_size_mb'] = round(ptd_size_mb, 2)

                results['executorch_models'].append(model_entry)

                logger.info(
                    f"{variant_name} {primary_metric}: "
                    f"{et_metrics.get(primary_metric, 0.0):.4f}"
                )
                logger.info(
                    f"{variant_name} mAP@0.5: "
                    f"{et_metrics.get('mAP_0.5', 0.0):.4f}"
                )
                latency_stats = et_results['latency'] or {}
                logger.info(
                    f"{variant_name} Avg Latency: "
                    f"{latency_stats.get('mean_ms', 0.0):.2f} ms"
                )
        else:
            logger.warning(
                "No valid ExecuTorch .pte variants found from config; "
                "skipping ExecuTorch section"
            )

    # ============================================================
    # Save Results
    # ============================================================
    consolidated_name = output_cfg.get(
        'consolidated_json_name', 'mobile_net_v2_ssd_evaluation.json'
    )
    output_file = results_dir / consolidated_name
    save_evaluation_results(results, output_file)

    logger.info("=" * 80)
    logger.info(f"Evaluation complete! Results saved to: {output_file}")
    logger.info("=" * 80)

    # ============================================================
    # Generate HTML Report
    # ============================================================
    if args.generate_report and report_cfg.get('generate_html', True):
        logger.info("\nGenerating HTML report...")
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        try:
            from evaluation.common.reporting.generate_html_report import (
                generate_html_for_json,
            )
            if generate_html_for_json(output_file, output_dir=results_dir):
                report_name = report_cfg.get(
                    'html_name',
                    'MobileNetV2-SSD-Lite_evaluation_report.html',
                )
                report_path = results_dir / report_name
                logger.info(f"✓ HTML report generated: {report_path}")
            else:
                logger.warning("⚠ Failed to generate HTML report")
        except ImportError as e:
            logger.error(f"Failed to import HTML report generator: {e}")


if __name__ == '__main__':
    main()
