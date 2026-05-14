#!/usr/bin/env python3
"""
compare_tflite_vs_pytorch.py
============================

Side-by-side end-to-end evaluation of the original (non-quantized) PyTorch
(.pth) and TFLite (.tflite) checkpoints for qfgaohao's MobileNetV2-SSD-Lite
on COCO-style VOC2012.

Design goals
------------
* Both backends share **identical input preprocessing**, decode
  hyper-parameters, and ground-truth, so the comparison isolates backend
  behavior from data-pipeline noise.
* **Single source of truth** for every threshold and path lives in `Config`
  — no magic numbers sprinkled through the code.
* **Latency is timed around the model forward only** (preprocessing and
  decode excluded); the first `warmup` images are discarded so JIT/XNNPACK
  warmup doesn't pollute the distribution.
* **COCO-style mAP** via `pycocotools.cocoeval.COCOeval` — the canonical
  metric used in every comparable paper.
* **Per-image matched-IoU** is computed as a secondary metric for the
  XLSX per-image sheet (mean IoU of greedy GT↔prediction matches at the
  default IoU threshold, score-thresholded predictions only).
* TFLite output format is **auto-detected**: post-NMS (TFLite SSD with
  detection-postprocess op, 4 outputs) or raw (2 outputs: scores + boxes,
  decoded with the same priors as the PyTorch path).

Outputs (default `--output-dir ./output/comparison`)
----------------------------------------------------
* `comparison_results.json` — full metric tree + per-image rows + system info
* `comparison_results.xlsx` — Summary, Per-Class AP, Per-Image, System Info

Usage
-----
    python compare_tflite_vs_pytorch.py \\
        --pt-weights ../model_sources/MobileNetV2/weights/mb2-ssd-lite-mp-0_686.pth \\
        --tflite    ../model_sources/MobileNetV2/weights/mb2-ssd-lite.tflite \\
        --source-path ../model_sources/MobileNetV2/src/pytorch/pytorch-ssd \\
        --gt-json   ../dataset/voc2012_as_coco/instances_voc2012_val.json \\
        --images-dir ~/.cache/kagglehub/datasets/watanabe2362/voctrainval-11may2012/versions/1/VOCdevkit/VOC2012/JPEGImages
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import statistics
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd  # noqa: F401 — used implicitly via openpyxl ExcelWriter
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm


# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True)
class Config:
    """Single source of truth for paths, hyper-params, output locations."""

    # --- Backends -----------------------------------------------------------
    pt_weights: Path
    tflite_path: Path
    source_path: Path             # qfgaohao `vision` module parent dir
    device: str = "cpu"           # PyTorch device

    # --- Data ---------------------------------------------------------------
    gt_json: Path = Path("instances_voc2012_val.json")
    images_dir: Path = Path("JPEGImages")
    limit: int | None = None      # None = full set

    # --- Model / decode -----------------------------------------------------
    num_classes: int = 21                                # incl. BACKGROUND
    input_size: int = 300
    score_threshold: float = 0.01
    nms_iou_threshold: float = 0.45
    max_detections_per_image: int = 100
    candidate_size: int = 200
    iou_match_threshold: float = 0.5                     # for per-image IoU

    # --- Latency ------------------------------------------------------------
    warmup_images: int = 5

    # --- Output -------------------------------------------------------------
    output_dir: Path = Path("output/comparison")
    json_filename: str = "comparison_results.json"
    xlsx_filename: str = "comparison_results.xlsx"

    # --- Misc ---------------------------------------------------------------
    log_level: str = "INFO"

    def validate(self) -> None:
        problems: list[str] = []
        for label, path, must_be_file in [
            ("pt_weights", self.pt_weights, True),
            ("tflite_path", self.tflite_path, True),
            ("source_path", self.source_path, False),
            ("gt_json", self.gt_json, True),
            ("images_dir", self.images_dir, False),
        ]:
            if must_be_file and not path.is_file():
                problems.append(f"  {label}: file not found: {path}")
            if not must_be_file and not path.is_dir():
                problems.append(f"  {label}: directory not found: {path}")
        if problems:
            raise FileNotFoundError(
                "Config validation failed:\n" + "\n".join(problems))


# =============================================================================
# Lightweight result types
# =============================================================================

@dataclass(frozen=True)
class Detection:
    """A single post-NMS detection, in original-image pixel coordinates."""
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    category_id: int  # 1-indexed COCO category (matches GT JSON)

    @property
    def xywh(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)


@dataclass
class PerImageResult:
    image_id: int
    file_name: str
    width: int
    height: int
    num_detections: int
    inference_time_ms: float
    matched_mean_iou: float | None        # None when no GT/predictions overlap


@dataclass
class BackendResult:
    name: str
    model_path: str
    model_size_mb: float
    per_image: list[PerImageResult] = field(default_factory=list)
    coco_predictions: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Geometry helpers
# =============================================================================

def iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two [N,4] and [M,4] xyxy arrays. Returns [N,M]."""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    a_x1, a_y1, a_x2, a_y2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    b_x1, b_y1, b_x2, b_y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    inter_x1 = np.maximum(a_x1, b_x1)
    inter_y1 = np.maximum(a_y1, b_y1)
    inter_x2 = np.minimum(a_x2, b_x2)
    inter_y2 = np.minimum(a_y2, b_y2)
    inter_w = np.clip(inter_x2 - inter_x1, 0, None)
    inter_h = np.clip(inter_y2 - inter_y1, 0, None)
    inter = inter_w * inter_h
    area_a = (a_x2 - a_x1) * (a_y2 - a_y1)
    area_b = (b_x2 - b_x1) * (b_y2 - b_y1)
    union = area_a + area_b - inter
    return (inter / np.clip(union, 1e-12, None)).astype(np.float32)


def matched_mean_iou(preds_xyxy: np.ndarray,
                     gt_xyxy: np.ndarray,
                     iou_threshold: float) -> float | None:
    """Greedy 1-1 matching, return mean IoU of matched pairs (None if 0)."""
    if preds_xyxy.size == 0 or gt_xyxy.size == 0:
        return None
    iou = iou_xyxy(preds_xyxy, gt_xyxy)
    matched_ious: list[float] = []
    used_gt: set[int] = set()
    pred_order = np.argsort(-iou.max(axis=1))   # best-matching preds first
    for p in pred_order:
        candidates = [
            (iou[p, g], g) for g in range(gt_xyxy.shape[0])
            if g not in used_gt and iou[p, g] >= iou_threshold
        ]
        if not candidates:
            continue
        best_iou, best_g = max(candidates, key=lambda t: t[0])
        used_gt.add(best_g)
        matched_ious.append(float(best_iou))
    if not matched_ious:
        return None
    return float(np.mean(matched_ious))


def hard_nms(boxes: np.ndarray, scores: np.ndarray,
             iou_threshold: float, top_k: int) -> np.ndarray:
    """Classic single-class hard NMS. Returns indices kept (sorted by score)."""
    if boxes.size == 0:
        return np.empty((0,), dtype=np.int64)
    order = np.argsort(-scores)
    keep: list[int] = []
    while order.size > 0 and len(keep) < top_k:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        ious = iou_xyxy(boxes[i:i + 1], boxes[order[1:]])[0]
        order = order[1:][ious < iou_threshold]
    return np.asarray(keep, dtype=np.int64)


# =============================================================================
# Image / GT loading
# =============================================================================

def load_coco_index(gt_json: Path) -> tuple[COCO, list[dict[str, Any]]]:
    log.info("Loading COCO ground truth: %s", gt_json)
    coco = COCO(str(gt_json))
    img_ids = coco.getImgIds()
    images = coco.loadImgs(img_ids)
    images = sorted(images, key=lambda im: im["id"])
    return coco, images


def load_image_rgb(images_dir: Path, file_name: str) -> Image.Image:
    path = images_dir / file_name
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")


def gt_boxes_xyxy(coco: COCO, image_id: int) -> np.ndarray:
    """Return GT boxes for an image as xyxy float32 [N,4]."""
    ann_ids = coco.getAnnIds(imgIds=[image_id], iscrowd=False)
    anns = coco.loadAnns(ann_ids)
    if not anns:
        return np.zeros((0, 4), dtype=np.float32)
    xywh = np.asarray([a["bbox"] for a in anns], dtype=np.float32)
    xyxy = np.empty_like(xywh)
    xyxy[:, 0] = xywh[:, 0]
    xyxy[:, 1] = xywh[:, 1]
    xyxy[:, 2] = xywh[:, 0] + xywh[:, 2]
    xyxy[:, 3] = xywh[:, 1] + xywh[:, 3]
    return xyxy


# =============================================================================
# Backends
# =============================================================================

class _BackendBase:
    """Common contract: load(), infer_and_decode(pil_image) -> (Detections, ms)."""

    name: str

    def model_size_mb(self) -> float:
        raise NotImplementedError

    def infer_and_decode(self, image: Image.Image
                         ) -> tuple[list[Detection], float]:
        raise NotImplementedError


# ----------------------------- PyTorch backend ------------------------------

class PyTorchBackend(_BackendBase):
    """Loads qfgaohao MobileNetV2-SSD-Lite .pth + uses its Predictor."""

    name = "PyTorch"

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._predictor = None  # set in load()
        self._weights_path = cfg.pt_weights

    def load(self) -> None:
        # qfgaohao's `vision` package must be importable; the user can pass
        # --source-path; we insert it into sys.path here.
        if str(self.cfg.source_path) not in sys.path:
            sys.path.insert(0, str(self.cfg.source_path))
        try:
            from vision.ssd.mobilenet_v2_ssd_lite import (
                create_mobilenetv2_ssd_lite,
                create_mobilenetv2_ssd_lite_predictor,
            )
        except ImportError as exc:
            raise RuntimeError(
                f"Cannot import qfgaohao 'vision' module from "
                f"{self.cfg.source_path}. Pass --source-path pointing at the "
                f"qfgaohao pytorch-ssd checkout."
            ) from exc

        log.info("[PyTorch] Building network …")
        net = create_mobilenetv2_ssd_lite(self.cfg.num_classes, is_test=True)
        log.info("[PyTorch] Loading weights: %s", self._weights_path)
        net.load(str(self._weights_path))
        net.eval()

        self._predictor = create_mobilenetv2_ssd_lite_predictor(
            net,
            candidate_size=self.cfg.candidate_size,
            nms_method=None,            # default = hard NMS in qfgaohao
            device=self.cfg.device,
        )
        # Override decode hyper-params to match Config.
        self._predictor.iou_threshold = self.cfg.nms_iou_threshold
        self._predictor.filter_threshold = self.cfg.score_threshold

    def model_size_mb(self) -> float:
        return self._weights_path.stat().st_size / (1024.0 * 1024.0)

    def infer_and_decode(self, image: Image.Image
                         ) -> tuple[list[Detection], float]:
        if self._predictor is None:
            raise RuntimeError("PyTorchBackend not loaded; call load() first.")

        np_image = np.asarray(image)  # RGB uint8 HWC
        t0 = time.perf_counter()
        # predictor.predict applies transform → forward → decode → NMS
        boxes, labels, probs = self._predictor.predict(
            np_image,
            top_k=self.cfg.max_detections_per_image,
            prob_threshold=self.cfg.score_threshold,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if boxes is None or len(boxes) == 0:
            return [], elapsed_ms

        boxes_np = boxes.detach().cpu().numpy().astype(np.float32)
        labels_np = labels.detach().cpu().numpy().astype(np.int64)
        probs_np = probs.detach().cpu().numpy().astype(np.float32)

        detections = [
            Detection(
                x1=float(b[0]), y1=float(b[1]),
                x2=float(b[2]), y2=float(b[3]),
                score=float(p),
                category_id=int(l),
            )
            for b, l, p in zip(boxes_np, labels_np, probs_np)
        ]
        return detections, elapsed_ms


# ----------------------------- TFLite backend -------------------------------

class TFLiteBackend(_BackendBase):
    """Loads a .tflite SSD model; auto-detects post-NMS vs raw output."""

    name = "TFLite"

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._interpreter = None
        self._input_details: list[dict[str, Any]] = []
        self._output_details: list[dict[str, Any]] = []
        self._is_post_nms: bool = False
        self._priors: np.ndarray | None = None
        self._mean = np.array([127.0, 127.0, 127.0], dtype=np.float32)
        self._std = 128.0

    # -- private helpers ----------------------------------------------------

    def _import_interpreter(self):
        try:
            from tflite_runtime.interpreter import Interpreter
            return Interpreter
        except ImportError:
            pass
        try:
            from tensorflow.lite.python.interpreter import Interpreter
            return Interpreter
        except ImportError as exc:
            raise RuntimeError(
                "Neither tflite_runtime nor tensorflow is installed. "
                "Install one: `pip install tflite-runtime`  or  "
                "`pip install tensorflow`."
            ) from exc

    def _build_priors(self) -> np.ndarray:
        """qfgaohao MobileNetV2-SSD-Lite priors at 300×300."""
        if str(self.cfg.source_path) not in sys.path:
            sys.path.insert(0, str(self.cfg.source_path))
        # qfgaohao stores priors in mobilenetv1_ssd_config.priors;
        # mb2-ssd-lite reuses them.
        from vision.ssd.config import mobilenetv1_ssd_config as ssd_cfg
        return ssd_cfg.priors.numpy().astype(np.float32)  # [num_priors,4] cxcywh

    # -- API ----------------------------------------------------------------

    def load(self) -> None:
        Interpreter = self._import_interpreter()
        log.info("[TFLite] Loading model: %s", self.cfg.tflite_path)
        self._interpreter = Interpreter(model_path=str(self.cfg.tflite_path))
        self._interpreter.allocate_tensors()
        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()

        # Auto-detect format.
        # Post-NMS: 4 outputs (boxes, classes, scores, num) with rank 3,2,2,1.
        # Raw qfgaohao: 2 outputs (scores [1,P,C], boxes [1,P,4]).
        n_out = len(self._output_details)
        if n_out == 4:
            self._is_post_nms = True
            log.info("[TFLite] Detected post-NMS output format (4 tensors)")
        elif n_out == 2:
            self._is_post_nms = False
            self._priors = self._build_priors()
            log.info("[TFLite] Detected raw output format (2 tensors); "
                     "applying decode + NMS in-process")
        else:
            raise RuntimeError(
                f"Unexpected TFLite output count: {n_out}. "
                f"Expected 4 (post-NMS) or 2 (raw)."
            )

    def model_size_mb(self) -> float:
        return self.cfg.tflite_path.stat().st_size / (1024.0 * 1024.0)

    # -- preprocessing -----------------------------------------------------

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        """Resize → normalize → batch-pack to the input tensor's dtype/layout."""
        in_det = self._input_details[0]
        target_h, target_w = in_det["shape"][1], in_det["shape"][2]
        resized = image.resize(
            (target_w, target_h), Image.Resampling.BILINEAR)
        arr = np.asarray(resized, dtype=np.float32)        # HWC RGB
        arr = (arr - self._mean) / self._std
        # Most qfgaohao TFLite exports take float32 NHWC.
        if in_det["dtype"] == np.uint8:
            # Quantized input — shouldn't happen for FP32 model, but just in case.
            scale, zero = in_det["quantization"]
            arr = np.clip(arr / scale + zero, 0, 255).astype(np.uint8)
        else:
            arr = arr.astype(np.float32)
        return np.expand_dims(arr, axis=0)

    # -- inference --------------------------------------------------------

    def infer_and_decode(self, image: Image.Image
                         ) -> tuple[list[Detection], float]:
        if self._interpreter is None:
            raise RuntimeError("TFLiteBackend not loaded; call load() first.")

        orig_w, orig_h = image.size
        x = self._preprocess(image)

        t0 = time.perf_counter()
        self._interpreter.set_tensor(self._input_details[0]["index"], x)
        self._interpreter.invoke()
        outs = [
            self._interpreter.get_tensor(d["index"])
            for d in self._output_details
        ]
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if self._is_post_nms:
            detections = self._decode_post_nms(outs, orig_w, orig_h)
        else:
            detections = self._decode_raw(outs, orig_w, orig_h)
        return detections, elapsed_ms

    # -- decode paths -----------------------------------------------------

    def _decode_post_nms(self, outs: Sequence[np.ndarray],
                         orig_w: int, orig_h: int) -> list[Detection]:
        # Standard TF SSD postprocess output order: boxes, classes, scores, num.
        # Identify by shape since order isn't guaranteed across exports.
        boxes_t = None
        classes_t = None
        scores_t = None
        num_t = None
        for o in outs:
            if o.ndim == 3 and o.shape[-1] == 4:
                boxes_t = o[0]
            elif o.ndim == 2:
                # classes or scores; classes are integer-ish floats, scores ∈ [0,1].
                if scores_t is None and 0.0 <= o.max() <= 1.0:
                    scores_t = o[0]
                else:
                    classes_t = o[0]
            elif o.ndim == 1:
                num_t = int(o[0])

        if boxes_t is None or scores_t is None or classes_t is None:
            raise RuntimeError(
                "Could not identify post-NMS outputs by shape.")

        if num_t is None:
            num_t = int(scores_t.shape[0])

        dets: list[Detection] = []
        for i in range(num_t):
            score = float(scores_t[i])
            if score < self.cfg.score_threshold:
                continue
            # boxes come normalized ymin, xmin, ymax, xmax
            ymin, xmin, ymax, xmax = boxes_t[i]
            x1 = float(xmin) * orig_w
            y1 = float(ymin) * orig_h
            x2 = float(xmax) * orig_w
            y2 = float(ymax) * orig_h
            cat = int(classes_t[i]) + 1   # TF labels are 0-indexed → COCO 1-idx
            dets.append(Detection(x1, y1, x2, y2, score, cat))
        return dets[: self.cfg.max_detections_per_image]

    def _decode_raw(self, outs: Sequence[np.ndarray],
                    orig_w: int, orig_h: int) -> list[Detection]:
        # qfgaohao SSD raw outputs:
        #   scores [1, num_priors, num_classes]   logits already softmaxed
        #   boxes  [1, num_priors, 4]             encoded relative to priors
        # The order between outputs[0] and outputs[1] isn't guaranteed —
        # identify by last-dim size.
        assert self._priors is not None
        if outs[0].shape[-1] == 4:
            boxes_raw, scores_raw = outs[0][0], outs[1][0]
        else:
            scores_raw, boxes_raw = outs[0][0], outs[1][0]

        decoded = self._decode_locations(
            boxes_raw, self._priors,
            center_variance=0.1, size_variance=0.2)
        # convert center cxcywh → xyxy
        decoded_xyxy = np.empty_like(decoded)
        decoded_xyxy[:, 0] = decoded[:, 0] - decoded[:, 2] / 2
        decoded_xyxy[:, 1] = decoded[:, 1] - decoded[:, 3] / 2
        decoded_xyxy[:, 2] = decoded[:, 0] + decoded[:, 2] / 2
        decoded_xyxy[:, 3] = decoded[:, 1] + decoded[:, 3] / 2
        # boxes are normalized [0,1] → scale to orig image
        decoded_xyxy[:, [0, 2]] *= orig_w
        decoded_xyxy[:, [1, 3]] *= orig_h

        dets: list[Detection] = []
        # Per-class NMS, skipping background (class 0).
        for c in range(1, scores_raw.shape[1]):
            cls_scores = scores_raw[:, c]
            keep_mask = cls_scores > self.cfg.score_threshold
            if not keep_mask.any():
                continue
            cls_boxes = decoded_xyxy[keep_mask]
            cls_s = cls_scores[keep_mask]
            keep_idx = hard_nms(
                cls_boxes, cls_s,
                self.cfg.nms_iou_threshold,
                top_k=self.cfg.candidate_size,
            )
            for i in keep_idx:
                b = cls_boxes[i]
                dets.append(Detection(
                    x1=float(b[0]), y1=float(b[1]),
                    x2=float(b[2]), y2=float(b[3]),
                    score=float(cls_s[i]),
                    category_id=c,
                ))
        # Top-k across all classes.
        dets.sort(key=lambda d: d.score, reverse=True)
        return dets[: self.cfg.max_detections_per_image]

    @staticmethod
    def _decode_locations(boxes_raw: np.ndarray, priors: np.ndarray,
                          center_variance: float, size_variance: float
                          ) -> np.ndarray:
        """SSD location decoding (cxcywh)."""
        cxcy = boxes_raw[:, :2] * center_variance * priors[:, 2:] + priors[:, :2]
        wh = np.exp(boxes_raw[:, 2:] * size_variance) * priors[:, 2:]
        return np.concatenate([cxcy, wh], axis=1)


# =============================================================================
# Evaluation orchestration
# =============================================================================

def evaluate_backend(backend: _BackendBase,
                     cfg: Config,
                     coco: COCO,
                     images: Sequence[Mapping[str, Any]]
                     ) -> BackendResult:
    log.info("=== Evaluating backend: %s ===", backend.name)
    backend.load()
    result = BackendResult(
        name=backend.name,
        model_path=str(
            cfg.pt_weights if backend.name == "PyTorch" else cfg.tflite_path),
        model_size_mb=backend.model_size_mb(),
    )

    for idx, img_meta in enumerate(tqdm(images, desc=backend.name, unit="img")):
        image_id = int(img_meta["id"])
        file_name = str(img_meta["file_name"])
        try:
            image = load_image_rgb(cfg.images_dir, file_name)
        except FileNotFoundError as exc:
            log.warning("Skipping missing image: %s (%s)", file_name, exc)
            continue

        try:
            detections, elapsed_ms = backend.infer_and_decode(image)
        except Exception:
            log.error("Inference failed on %s:\n%s",
                      file_name, traceback.format_exc())
            continue

        # COCO predictions list.
        for d in detections:
            x, y, w, h = d.xywh
            result.coco_predictions.append({
                "image_id": image_id,
                "category_id": d.category_id,
                "bbox": [float(x), float(y), float(w), float(h)],
                "score": float(d.score),
            })

        # Per-image matched IoU.
        pred_xyxy = (
            np.asarray([[d.x1, d.y1, d.x2, d.y2] for d in detections],
                       dtype=np.float32)
            if detections else np.zeros((0, 4), dtype=np.float32)
        )
        gt_xyxy = gt_boxes_xyxy(coco, image_id)
        mean_iou = matched_mean_iou(
            pred_xyxy, gt_xyxy, cfg.iou_match_threshold)

        result.per_image.append(PerImageResult(
            image_id=image_id,
            file_name=file_name,
            width=int(img_meta.get("width", image.width)),
            height=int(img_meta.get("height", image.height)),
            num_detections=len(detections),
            inference_time_ms=elapsed_ms,
            matched_mean_iou=mean_iou,
        ))

    return result


# =============================================================================
# Metrics
# =============================================================================

def _latency_stats(per_image: Sequence[PerImageResult],
                   warmup: int) -> dict[str, float]:
    """Return {mean, std, p50, p90, p95, p99} in ms over warm samples."""
    samples = [p.inference_time_ms for p in per_image[warmup:]]
    if not samples:
        return {"mean_ms": 0.0, "std_ms": 0.0, "p50_ms": 0.0,
                "p90_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0}
    arr = np.asarray(samples, dtype=np.float64)
    return {
        "mean_ms": float(arr.mean()),
        "std_ms":  float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "p50_ms":  float(np.percentile(arr, 50)),
        "p90_ms":  float(np.percentile(arr, 90)),
        "p95_ms":  float(np.percentile(arr, 95)),
        "p99_ms":  float(np.percentile(arr, 99)),
        "warmup_excluded": float(warmup),
        "n_samples": float(arr.size),
    }


def compute_metrics(result: BackendResult,
                    coco_gt: COCO,
                    cfg: Config) -> None:
    """Populate result.metrics in place."""
    log.info("Computing COCO mAP for %s …", result.name)
    if not result.coco_predictions:
        log.warning("%s produced zero detections — mAP will be 0.", result.name)
        coco_stats: dict[str, float] = {k: 0.0 for k in _COCO_STAT_KEYS}
        per_class_ap: dict[str, float] = {}
    else:
        coco_dt = coco_gt.loadRes(result.coco_predictions)
        coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
        coco_eval.params.imgIds = [p.image_id for p in result.per_image]
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        coco_stats = dict(zip(_COCO_STAT_KEYS, [float(s) for s in coco_eval.stats]))
        per_class_ap = _per_class_ap(coco_eval, coco_gt)

    # Per-image matched-IoU summary.
    matched_ious = [
        p.matched_mean_iou for p in result.per_image
        if p.matched_mean_iou is not None
    ]
    mean_matched_iou = float(np.mean(matched_ious)) if matched_ious else 0.0

    result.metrics = {
        "model_size_mb": result.model_size_mb,
        "num_images": len(result.per_image),
        "num_predictions": len(result.coco_predictions),
        "coco": coco_stats,
        "per_class_AP_at_IoU_0.5_0.95": per_class_ap,
        "mean_matched_iou_at_0.5": mean_matched_iou,
        "n_images_with_matches": float(len(matched_ious)),
        "latency": _latency_stats(result.per_image, cfg.warmup_images),
    }


# pycocotools.COCOeval.summarize order:
_COCO_STAT_KEYS: Final[tuple[str, ...]] = (
    "mAP_0.5_0.95",
    "mAP_0.5",
    "mAP_0.75",
    "mAP_small",
    "mAP_medium",
    "mAP_large",
    "AR_max1",
    "AR_max10",
    "AR_max100",
    "AR_small",
    "AR_medium",
    "AR_large",
)


def _per_class_ap(coco_eval: COCOeval, coco_gt: COCO) -> dict[str, float]:
    """Return AP @[.5:.95] per category, keyed by category name."""
    # precision shape: [T, R, K, A, M] = [10 IoUs, 101 recalls, K classes, 4 areas, 3 maxDets]
    precision = coco_eval.eval.get("precision")
    if precision is None:
        return {}
    cat_ids = coco_eval.params.catIds
    cats = {c["id"]: c["name"] for c in coco_gt.loadCats(cat_ids)}
    out: dict[str, float] = {}
    for k, cat_id in enumerate(cat_ids):
        # all areas (idx 0), maxDets=100 (idx 2)
        p = precision[:, :, k, 0, 2]
        p = p[p > -1]
        ap = float(p.mean()) if p.size else 0.0
        out[cats[cat_id]] = ap
    return out


# =============================================================================
# Report writers
# =============================================================================

def write_json(out_path: Path,
               cfg: Config,
               results: Sequence[BackendResult],
               system_info: Mapping[str, Any]) -> None:
    payload = {
        "config": {k: (str(v) if isinstance(v, Path) else v)
                   for k, v in asdict(cfg).items()},
        "system_info": dict(system_info),
        "backends": {
            r.name: {
                "model_path": r.model_path,
                "metrics": r.metrics,
                "per_image": [asdict(p) for p in r.per_image],
            }
            for r in results
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, default=_json_default)
    log.info("Wrote JSON: %s", out_path)


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Unserializable: {type(o)}")


def write_xlsx(out_path: Path,
               cfg: Config,
               results: Sequence[BackendResult],
               system_info: Mapping[str, Any]) -> None:
    import pandas as pd  # local — keeps top-level import cost low

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        _summary_sheet(writer, results)
        _per_class_sheet(writer, results)
        _per_image_sheet(writer, results)
        _system_info_sheet(writer, cfg, system_info)
    log.info("Wrote XLSX: %s", out_path)


def _summary_sheet(writer, results: Sequence[BackendResult]) -> None:
    import pandas as pd
    rows: list[dict[str, Any]] = []
    for r in results:
        m = r.metrics
        coco = m.get("coco", {})
        lat = m.get("latency", {})
        rows.append({
            "Backend": r.name,
            "Model Path": r.model_path,
            "Model Size (MB)": round(m.get("model_size_mb", 0.0), 3),
            "# Images": int(m.get("num_images", 0)),
            "# Predictions": int(m.get("num_predictions", 0)),
            "mAP @[.5:.95]": round(coco.get("mAP_0.5_0.95", 0.0), 4),
            "mAP @.5":       round(coco.get("mAP_0.5",       0.0), 4),
            "mAP @.75":      round(coco.get("mAP_0.75",      0.0), 4),
            "AR @100":       round(coco.get("AR_max100",     0.0), 4),
            "Matched-IoU mean (≥0.5)":
                round(m.get("mean_matched_iou_at_0.5", 0.0), 4),
            "# images matched":
                int(m.get("n_images_with_matches", 0)),
            "Latency mean (ms)": round(lat.get("mean_ms", 0.0), 3),
            "Latency p50 (ms)":  round(lat.get("p50_ms",  0.0), 3),
            "Latency p95 (ms)":  round(lat.get("p95_ms",  0.0), 3),
            "Latency p99 (ms)":  round(lat.get("p99_ms",  0.0), 3),
            "Latency std (ms)":  round(lat.get("std_ms",  0.0), 3),
        })
    pd.DataFrame(rows).to_excel(writer, sheet_name="Summary", index=False)


def _per_class_sheet(writer, results: Sequence[BackendResult]) -> None:
    import pandas as pd
    # union of class names across backends, in the order seen on first backend
    if not results:
        return
    base_order = list(
        results[0].metrics.get("per_class_AP_at_IoU_0.5_0.95", {}).keys())
    extra = {
        name
        for r in results
        for name in r.metrics.get("per_class_AP_at_IoU_0.5_0.95", {}).keys()
    } - set(base_order)
    class_order = base_order + sorted(extra)

    table: dict[str, list[Any]] = {"Class": class_order}
    for r in results:
        d = r.metrics.get("per_class_AP_at_IoU_0.5_0.95", {})
        table[f"{r.name} AP @[.5:.95]"] = [round(d.get(c, 0.0), 4)
                                            for c in class_order]
    pd.DataFrame(table).to_excel(
        writer, sheet_name="Per-Class AP", index=False)


def _per_image_sheet(writer, results: Sequence[BackendResult]) -> None:
    import pandas as pd
    if not results:
        return
    # Align rows across backends by (image_id, file_name).
    index_map: dict[int, dict[str, Any]] = {}
    for r in results:
        for p in r.per_image:
            row = index_map.setdefault(p.image_id, {
                "image_id": p.image_id,
                "file_name": p.file_name,
                "width": p.width,
                "height": p.height,
            })
            row[f"{r.name} latency_ms"] = round(p.inference_time_ms, 3)
            row[f"{r.name} #det"] = p.num_detections
            row[f"{r.name} matched_IoU"] = (
                round(p.matched_mean_iou, 4)
                if p.matched_mean_iou is not None else None
            )
    rows = sorted(index_map.values(), key=lambda r: r["image_id"])
    pd.DataFrame(rows).to_excel(writer, sheet_name="Per-Image", index=False)


def _system_info_sheet(writer, cfg: Config,
                       system_info: Mapping[str, Any]) -> None:
    import pandas as pd
    info_rows: list[dict[str, Any]] = []
    for k, v in system_info.items():
        info_rows.append({"Key": k, "Value": str(v)})
    info_rows.append({"Key": "—" * 8, "Value": "—" * 8})
    for k, v in asdict(cfg).items():
        info_rows.append({"Key": f"config.{k}", "Value": str(v)})
    pd.DataFrame(info_rows).to_excel(
        writer, sheet_name="System Info", index=False)


# =============================================================================
# System info
# =============================================================================

def collect_system_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform":       platform.platform(),
        "python":         sys.version.split()[0],
        "processor":      platform.processor() or "unknown",
        "machine":        platform.machine(),
    }
    try:
        import torch  # noqa: WPS433
        info["torch"] = torch.__version__
        info["torch_cuda_available"] = bool(torch.cuda.is_available())
    except ImportError:
        info["torch"] = "not installed"
    try:
        import tflite_runtime  # type: ignore  # noqa: WPS433
        info["tflite_runtime"] = tflite_runtime.__version__  # type: ignore[attr-defined]
    except (ImportError, AttributeError):
        try:
            import tensorflow as tf  # noqa: WPS433
            info["tensorflow"] = tf.__version__
        except ImportError:
            info["tflite"] = "not installed"
    try:
        import pycocotools as pct  # noqa: WPS433
        info["pycocotools"] = getattr(pct, "__version__", "unknown")
    except ImportError:
        info["pycocotools"] = "not installed"
    info["timestamp_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return info


# =============================================================================
# CLI / main
# =============================================================================

log: Final[logging.Logger] = logging.getLogger("compare")


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--pt-weights", type=Path, required=True,
                   help="Path to PyTorch .pth checkpoint (qfgaohao format)")
    p.add_argument("--tflite", type=Path, required=True,
                   help="Path to .tflite model (non-quantized)")
    p.add_argument("--source-path", type=Path, required=True,
                   help="Path to qfgaohao pytorch-ssd checkout "
                        "(contains the 'vision' package)")
    p.add_argument("--gt-json", type=Path, required=True,
                   help="COCO-format ground truth JSON")
    p.add_argument("--images-dir", type=Path, required=True,
                   help="Directory containing image files referenced "
                        "by gt-json file_names")
    p.add_argument("--output-dir", type=Path,
                   default=Path("output/comparison"),
                   help="Output directory for JSON + XLSX (default: %(default)s)")
    p.add_argument("--limit", type=int, default=None,
                   help="Evaluate only the first N images (debug)")
    p.add_argument("--device", type=str, default="cpu",
                   choices=("cpu", "cuda"),
                   help="PyTorch device (default: cpu)")
    p.add_argument("--warmup", type=int, default=5,
                   help="Discard first N images from latency stats (default 5)")
    p.add_argument("--score-threshold", type=float, default=0.01)
    p.add_argument("--nms-iou-threshold", type=float, default=0.45)
    p.add_argument("--max-detections", type=int, default=100)
    p.add_argument("--candidate-size", type=int, default=200)
    p.add_argument("--log-level", default="INFO",
                   choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return p


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    _setup_logging(args.log_level)

    cfg = Config(
        pt_weights=args.pt_weights.expanduser().resolve(),
        tflite_path=args.tflite.expanduser().resolve(),
        source_path=args.source_path.expanduser().resolve(),
        gt_json=args.gt_json.expanduser().resolve(),
        images_dir=args.images_dir.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        device=args.device,
        limit=args.limit,
        warmup_images=args.warmup,
        score_threshold=args.score_threshold,
        nms_iou_threshold=args.nms_iou_threshold,
        max_detections_per_image=args.max_detections,
        candidate_size=args.candidate_size,
        log_level=args.log_level,
    )
    cfg.validate()
    log.info("Resolved config:")
    for k, v in asdict(cfg).items():
        log.info("  %s = %s", k, v)

    coco, images = load_coco_index(cfg.gt_json)
    if cfg.limit is not None:
        images = images[: cfg.limit]
    log.info("Evaluating on %d image(s)", len(images))

    backends: list[_BackendBase] = [
        PyTorchBackend(cfg),
        TFLiteBackend(cfg),
    ]
    results: list[BackendResult] = []
    for b in backends:
        try:
            r = evaluate_backend(b, cfg, coco, images)
            compute_metrics(r, coco, cfg)
            results.append(r)
        except Exception:
            log.error("Backend %s failed:\n%s",
                      b.name, traceback.format_exc())

    system_info = collect_system_info()
    write_json(cfg.output_dir / cfg.json_filename, cfg, results, system_info)
    write_xlsx(cfg.output_dir / cfg.xlsx_filename, cfg, results, system_info)

    # Console summary
    log.info("\n%s\nCOMPARISON SUMMARY\n%s", "=" * 64, "=" * 64)
    for r in results:
        m = r.metrics
        coco_m = m.get("coco", {})
        lat = m.get("latency", {})
        log.info(
            "%-8s | size=%6.2f MB | mAP@.5:.95=%.4f | mAP@.5=%.4f "
            "| mean=%6.2f ms | p95=%6.2f ms",
            r.name,
            m.get("model_size_mb", 0.0),
            coco_m.get("mAP_0.5_0.95", 0.0),
            coco_m.get("mAP_0.5", 0.0),
            lat.get("mean_ms", 0.0),
            lat.get("p95_ms", 0.0),
        )
    log.info("%s", "=" * 64)
    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
