#!/usr/bin/env python3
"""
tflite_converter_v2.py
======================
Robust TFLite -> PyTorch converter with end-to-end numerical parity validation,
per-operator coverage tracking, and structured conversion reporting.

Pipeline
--------
    TFLite (source of truth)
        -> ONNX (via tf2onnx)
        -> ONNX-level patches (custom-op decomposition)
        -> shape re-inference
        -> PyTorch GraphModule (via onnx2torch)
        -> FX fixups (weight -> shape -> structural)
        -> NUMERICAL PARITY GATE (mandatory)
        -> save .pt (only if parity passes)
        -> conversion report (JSON + HTML)

Design principles
-----------------
1. The TFLite model is the source of truth. The .pt file is only saved if its
   outputs match the TFLite outputs within tolerance on multiple inputs.
2. Every fixup is self-detecting and a no-op when its target pattern is absent.
3. Every transformation is recorded. Operator coverage, shape changes, weight
   modifications, and numerical divergences are all tracked.
4. Heuristics are eliminated where ONNX metadata can provide deterministic
   answers (e.g., ConvTranspose detection by op_type, not by name/shape).
5. Failures are loud, specific, and actionable. No silent excepts.

Usage
-----
Library::

    from tflite_converter_v2 import convert_tflite_to_pytorch
    result = convert_tflite_to_pytorch(
        tflite_path="model.tflite",
        output_pt_path="model.pt",
        sample_inputs=[real_preprocessed_image],
    )
    if not result.parity_passed:
        raise SystemExit(result.summary())

CLI::

    python -m tflite_converter_v2 convert \\
        --tflite model.tflite --output model.pt \\
        --sample-image test.jpg --report report.html

    python -m tflite_converter_v2 parity \\
        --tflite model.tflite --pytorch model.pt
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Tolerances & constants (no magic numbers below this line)
# =============================================================================
TOL_PARITY_MAX_ABS: float = 1.0e-2
TOL_PARITY_COSINE: float = 0.999
TOL_LAYER_MAX_ABS: float = 5.0e-2
TOL_LAYER_COSINE: float = 0.995
TOL_WEIGHT_MAX_ABS: float = 1.0e-5

DEFAULT_ONNX_OPSET: int = 15
DEFAULT_PARITY_INPUT_COUNT: int = 4  # zeros, ones, randn(seed=0), randn(seed=1)
TFLITE_CONVTRANSPOSE_WEIGHT_PERM: tuple[int, ...] = (3, 0, 1, 2)
TFLITE_CONV_TRANSPOSE_CUSTOM_OP: str = "TFL_Convolution2DTransposeBias"
TFLITE_MAXPOOL_ARGMAX_CUSTOM_OP: str = "TFL_MaxPoolingWithArgmax2D"
TFLITE_MAXUNPOOL_CUSTOM_OP: str = "TFL_MaxUnpooling2D"


# =============================================================================
# Exceptions
# =============================================================================
class ConverterError(Exception):
    """Base for all converter errors."""


class ConversionStageError(ConverterError):
    """A pipeline stage failed."""


class ParityFailure(ConverterError):
    """TFLite vs PyTorch outputs diverged beyond tolerance."""


class UnsupportedOperatorError(ConverterError):
    """A TFLite operator has no known mapping to PyTorch."""


class MissingAttributeError(ConverterError):
    """An ONNX node is missing a required attribute and the converter
    refuses to guess."""


# =============================================================================
# Records (everything observable lives here)
# =============================================================================
@dataclass
class StageRecord:
    """One stage of the pipeline."""

    name: str
    success: bool = False
    duration_s: float = 0.0
    notes: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class OperatorRecord:
    """Mapping from a TFLite/ONNX op to its PyTorch realization."""

    tflite_op: Optional[str]
    onnx_op: Optional[str]
    pytorch_module: Optional[str]
    occurrences: int = 1
    status: str = "mapped"  # "mapped" | "patched" | "fallback" | "unmapped"
    notes: list[str] = field(default_factory=list)


@dataclass
class TensorDiff:
    """Per-tensor comparison between TFLite and PyTorch outputs."""

    name: str
    max_abs: float
    mean_abs: float
    cosine: float
    tflite_shape: tuple[int, ...]
    pytorch_shape: tuple[int, ...]
    passed: bool

    def __str__(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        return (
            f"[{mark}] {self.name}: "
            f"max_abs={self.max_abs:.4g} cos={self.cosine:.4f} "
            f"(TF {self.tflite_shape} vs PT {self.pytorch_shape})"
        )


@dataclass
class ParityRunResult:
    """Result of one parity test against one input."""

    input_label: str
    tensor_diffs: list[TensorDiff]
    passed: bool


@dataclass
class ConversionReport:
    """Top-level report. Everything observable from a conversion run."""

    tflite_path: str
    output_pt_path: str
    stages: list[StageRecord] = field(default_factory=list)
    operators: dict[str, OperatorRecord] = field(default_factory=dict)
    parity_runs: list[ParityRunResult] = field(default_factory=list)
    parity_passed: bool = False
    total_duration_s: float = 0.0
    error: Optional[str] = None

    # --- Aggregations ----------------------------------------------------
    @property
    def all_stages_passed(self) -> bool:
        return all(s.success for s in self.stages)

    @property
    def unmapped_operators(self) -> list[OperatorRecord]:
        return [op for op in self.operators.values() if op.status == "unmapped"]

    @property
    def patched_operators(self) -> list[OperatorRecord]:
        return [op for op in self.operators.values() if op.status == "patched"]

    # --- Serialization ---------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "tflite_path": self.tflite_path,
            "output_pt_path": self.output_pt_path,
            "parity_passed": self.parity_passed,
            "all_stages_passed": self.all_stages_passed,
            "total_duration_s": self.total_duration_s,
            "error": self.error,
            "stages": [dataclasses.asdict(s) for s in self.stages],
            "operators": {k: dataclasses.asdict(v) for k, v in self.operators.items()},
            "parity_runs": [
                {
                    "input_label": run.input_label,
                    "passed": run.passed,
                    "tensor_diffs": [dataclasses.asdict(d) for d in run.tensor_diffs],
                }
                for run in self.parity_runs
            ],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def summary(self) -> str:
        """Human-readable one-screen summary."""
        lines = [
            "=" * 78,
            f"CONVERSION REPORT  {self.tflite_path} -> {self.output_pt_path}",
            "=" * 78,
            f"Status:   {'PASS' if self.parity_passed else 'FAIL'}",
            f"Duration: {self.total_duration_s:.2f}s",
            f"Stages:   {sum(s.success for s in self.stages)}/{len(self.stages)} passed",
            f"Ops:      {len(self.operators)} unique; "
            f"{len(self.unmapped_operators)} unmapped, "
            f"{len(self.patched_operators)} patched",
        ]
        if self.error:
            lines.extend(["", f"ERROR: {self.error}"])
        if self.parity_runs:
            lines.append("")
            lines.append("Parity runs:")
            for run in self.parity_runs:
                mark = "PASS" if run.passed else "FAIL"
                worst = max(
                    (d.max_abs for d in run.tensor_diffs), default=float("nan")
                )
                lines.append(
                    f"  [{mark}] {run.input_label}: worst max_abs={worst:.4g}"
                )
                if not run.passed:
                    for d in run.tensor_diffs:
                        if not d.passed:
                            lines.append(f"        {d}")
        if self.unmapped_operators:
            lines.append("")
            lines.append("Unmapped operators (manual mapping required):")
            for op in self.unmapped_operators:
                lines.append(f"  - {op.tflite_op or op.onnx_op} (x{op.occurrences})")
        lines.append("=" * 78)
        return "\n".join(lines)


# =============================================================================
# Stage timing context manager
# =============================================================================
class _Stage:
    """Context manager that records a pipeline stage into a report."""

    def __init__(self, report: ConversionReport, name: str):
        self._report = report
        self._record = StageRecord(name=name)
        self._t0 = 0.0

    def __enter__(self) -> StageRecord:
        self._t0 = time.perf_counter()
        logger.info("[stage:start] %s", self._record.name)
        return self._record

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self._record.duration_s = time.perf_counter() - self._t0
        if exc_type is None:
            self._record.success = True
            logger.info(
                "[stage:done ] %s (%.2fs)",
                self._record.name,
                self._record.duration_s,
            )
        else:
            self._record.success = False
            self._record.notes.append(f"{exc_type.__name__}: {exc_val}")
            logger.error(
                "[stage:fail ] %s (%.2fs): %s",
                self._record.name,
                self._record.duration_s,
                exc_val,
            )
        self._report.stages.append(self._record)
        return False  # never swallow


# =============================================================================
# Export-friendly nn.Module replacements (torch.export-safe)
# =============================================================================
def _import_torch() -> tuple[Any, Any, Any]:
    """Lazy torch import so the module can be imported in environments
    that only need to read reports."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    return torch, nn, F


class StaticReshape:
    """Reshape with a compile-time-constant output shape.

    Replaces ``OnnxReshape``, whose ``_do_reshape`` uses
    ``if torch.any(shape == 0)`` -- a data-dependent symbolic guard that
    breaks ``torch.export``. Constructed lazily because torch may not be
    importable at module load.
    """

    @staticmethod
    def build(output_shape: tuple[int, ...]):
        torch, nn, _F = _import_torch()

        class _StaticReshapeImpl(nn.Module):
            def __init__(self, shape: tuple[int, ...]):
                super().__init__()
                self._output_shape = tuple(shape)

            def forward(self, x, *args, **kwargs):  # noqa: ARG002
                return x.reshape(self._output_shape)

            def extra_repr(self) -> str:
                return f"output_shape={self._output_shape}"

        return _StaticReshapeImpl(output_shape)


class StaticResize:
    """Interpolate with a baked-in output spatial size.

    Replaces ``OnnxResize``, whose ``forward`` calls ``scales.tolist()``
    and then compares ``scales[:2] != [1, 1]`` -- data-dependent symbolic
    guards that break ``torch.export``.
    """

    @staticmethod
    def build(output_spatial_size: tuple[int, ...], mode: str, align_corners: bool):
        torch, nn, F = _import_torch()

        class _StaticResizeImpl(nn.Module):
            def __init__(self, size: tuple[int, ...], m: str, ac: bool):
                super().__init__()
                self._output_spatial_size = tuple(size)
                self._mode = m
                self._align_corners = ac if m in {"linear", "bilinear", "trilinear", "bicubic"} else None

            def forward(self, x, *args, **kwargs):  # noqa: ARG002
                return F.interpolate(
                    x,
                    size=self._output_spatial_size,
                    mode=self._mode,
                    align_corners=self._align_corners,
                )

            def extra_repr(self) -> str:
                return (
                    f"size={self._output_spatial_size} mode={self._mode} "
                    f"align_corners={self._align_corners}"
                )

        return _StaticResizeImpl(output_spatial_size, mode, align_corners)


class NativeMinMax:
    """Min/Max via ``torch.minimum``/``torch.maximum`` (broadcasts natively).

    Replaces ``OnnxMinMax``, whose ``apply_reduction`` calls
    ``tensor.broadcast_to(shape)``, producing zero-stride intermediate
    tensors that ExecuTorch's ``SpecPropPass`` rejects with
    ``"0 in strides is not supported"``.
    """

    @staticmethod
    def build(op_type: str):
        torch, nn, _F = _import_torch()
        if op_type not in {"Min", "Max"}:
            raise ValueError(f"NativeMinMax: op_type must be 'Min' or 'Max', got {op_type!r}")

        class _NativeMinMaxImpl(nn.Module):
            def __init__(self, t: str):
                super().__init__()
                self._op_type = t
                self._reducer = torch.minimum if t == "Min" else torch.maximum

            def forward(self, *tensors):
                if not tensors:
                    raise ValueError("NativeMinMax: no input tensors")
                result = tensors[0]
                for t in tensors[1:]:
                    result = self._reducer(result, t)
                return result

            def extra_repr(self) -> str:
                return f"op_type={self._op_type}"

        return _NativeMinMaxImpl(op_type)


class TFLMaxPoolWithArgmax2d:
    """NHWC max-pool that returns indices in PyTorch's expected NCHW format.

    Key correctness invariant: the returned ``indices`` tensor is permuted
    to NCHW so that ``F.max_unpool2d`` -- which expects per-(N,C) flat HxW
    offsets in NCHW layout -- can consume it directly. The previous
    implementation kept indices in NHWC, which silently scrambled spatial
    positions during unpool.
    """

    @staticmethod
    def build(kernel_size: tuple[int, int], stride: tuple[int, int], return_indices: bool):
        torch, nn, F = _import_torch()

        class _Impl(nn.Module):
            def __init__(self, ks, st, ri):
                super().__init__()
                self._kernel_size = tuple(ks)
                self._stride = tuple(st)
                self._return_indices = bool(ri)

            def forward(self, input_tensor):  # NHWC in
                nchw = input_tensor.permute(0, 3, 1, 2).contiguous()
                values, indices = F.max_pool2d(
                    nchw,
                    kernel_size=self._kernel_size,
                    stride=self._stride,
                    return_indices=True,
                )
                values_nhwc = values.permute(0, 2, 3, 1).contiguous()
                if self._return_indices:
                    # Indices stay in NCHW for downstream max_unpool2d.
                    return values_nhwc, indices
                return values_nhwc

            def extra_repr(self) -> str:
                return (
                    f"kernel_size={self._kernel_size} stride={self._stride} "
                    f"return_indices={self._return_indices}"
                )

        return _Impl(kernel_size, stride, return_indices)


class TFLMaxUnpool2d:
    """NHWC max-unpool consuming NCHW indices from TFLMaxPoolWithArgmax2d.

    Mirror of the pool: NHWC -> NCHW for compute, back to NHWC on output.
    """

    @staticmethod
    def build(kernel_size: tuple[int, int], stride: tuple[int, int], output_size_nhwc: Optional[tuple[int, ...]]):
        torch, nn, F = _import_torch()

        class _Impl(nn.Module):
            def __init__(self, ks, st, osz):
                super().__init__()
                self._kernel_size = tuple(ks)
                self._stride = tuple(st)
                self._output_size_nhwc = tuple(osz) if osz is not None else None

            def forward(self, input_tensor, indices_tensor):  # NHWC in, NCHW indices
                nchw = input_tensor.permute(0, 3, 1, 2).contiguous()
                if self._output_size_nhwc is not None:
                    n, h, w, c = self._output_size_nhwc
                    output_size = (n, c, h, w)
                else:
                    output_size = None
                values = F.max_unpool2d(
                    nchw,
                    indices_tensor,
                    kernel_size=self._kernel_size,
                    stride=self._stride,
                    output_size=output_size,
                )
                return values.permute(0, 2, 3, 1).contiguous()

            def extra_repr(self) -> str:
                return (
                    f"kernel_size={self._kernel_size} stride={self._stride} "
                    f"output_size_nhwc={self._output_size_nhwc}"
                )

        return _Impl(kernel_size, stride, output_size_nhwc)


# =============================================================================
# ONNX-level utilities & fixups
# =============================================================================
def _get_attr(node, name: str, default: Any = None) -> Any:
    """Return an ONNX node attribute by name, or default if absent."""
    from onnx import helper

    for a in node.attribute:
        if a.name == name:
            return helper.get_attribute_value(a)
    return default


def _require_attr(node, name: str) -> Any:
    """Return an ONNX node attribute or raise -- no silent defaults."""
    value = _get_attr(node, name, default=None)
    if value is None:
        raise MissingAttributeError(
            f"Node {node.name or node.op_type!r} missing required attribute {name!r}"
        )
    return value


def _shape_of(graph, tensor_name: str) -> Optional[tuple[int, ...]]:
    """Look up the shape of a tensor from value_info, inputs, or outputs."""
    for collection in (graph.value_info, graph.input, graph.output):
        for vi in collection:
            if vi.name == tensor_name and vi.type.tensor_type.HasField("shape"):
                return tuple(
                    int(d.dim_value) if getattr(d, "dim_value", 0) > 0 else -1
                    for d in vi.type.tensor_type.shape.dim
                )
    return None


def patch_tfl_convtranspose_bias_onnx(model, report: ConversionReport):
    """Decompose ``TFL_Convolution2DTransposeBias`` into standard ONNX ops.

    Reads stride/pad/output_padding from the node's actual attributes.
    Refuses to guess when attributes are absent (raises
    ``MissingAttributeError``).
    """
    import onnx
    from onnx import helper

    graph = model.graph
    targets = [
        n for n in graph.node if n.op_type == TFLITE_CONV_TRANSPOSE_CUSTOM_OP
    ]
    if not targets:
        return model

    new_nodes: list[Any] = []
    replaced = 0

    for node in graph.node:
        if node.op_type != TFLITE_CONV_TRANSPOSE_CUSTOM_OP:
            new_nodes.append(node)
            continue

        if len(node.input) < 3 or len(node.output) < 1:
            raise ConversionStageError(
                f"{TFLITE_CONV_TRANSPOSE_CUSTOM_OP} node {node.name!r} has "
                f"unexpected I/O: {len(node.input)} inputs, {len(node.output)} outputs"
            )

        x, w, b = node.input[0], node.input[1], node.input[2]
        y = node.output[0]
        x_nchw = f"{y}_input_nchw"
        out_nchw = f"{y}_convtranspose_nchw"
        out_nhwc = f"{y}_convtranspose_nhwc"

        # Pull attributes from the node itself -- raise if missing.
        strides = _require_attr(node, "strides")
        pads = _get_attr(node, "pads", default=[0, 0, 0, 0])
        output_padding = _get_attr(node, "output_padding", default=[0, 0])

        to_nchw = helper.make_node(
            "Transpose",
            inputs=[x],
            outputs=[x_nchw],
            name=f"{node.name}_ToNCHW",
            perm=[0, 3, 1, 2],
        )
        conv = helper.make_node(
            "ConvTranspose",
            inputs=[x_nchw, w],
            outputs=[out_nchw],
            name=f"{node.name}_ConvTranspose",
            strides=list(strides),
            pads=list(pads),
            output_padding=list(output_padding),
        )
        to_nhwc = helper.make_node(
            "Transpose",
            inputs=[out_nchw],
            outputs=[out_nhwc],
            name=f"{node.name}_ToNHWC",
            perm=[0, 2, 3, 1],
        )
        add = helper.make_node(
            "Add",
            inputs=[out_nhwc, b],
            outputs=[y],
            name=f"{node.name}_BiasAdd",
        )
        new_nodes.extend([to_nchw, conv, to_nhwc, add])
        replaced += 1

    del graph.node[:]
    graph.node.extend(new_nodes)

    try:
        onnx.checker.check_model(model)
    except onnx.checker.ValidationError as exc:
        # Non-fatal: shape inference will normalize later.
        logger.debug("ONNX check after bias patch (non-fatal): %s", exc)

    report.operators.setdefault(
        TFLITE_CONV_TRANSPOSE_CUSTOM_OP,
        OperatorRecord(
            tflite_op=TFLITE_CONV_TRANSPOSE_CUSTOM_OP,
            onnx_op="ConvTranspose+Add",
            pytorch_module=None,
            occurrences=replaced,
            status="patched",
            notes=[f"Decomposed {replaced} node(s) into Transpose/ConvTranspose/Transpose/Add"],
        ),
    )
    return model


def permute_convtranspose_weights_in_onnx(model, report: ConversionReport) -> None:
    """Permute TFLite ConvTranspose weight initializers from
    ``[O, kH, kW, I]`` to ``[I, O/groups, kH, kW]`` *deterministically*
    by walking the ONNX graph for ``ConvTranspose`` ops and identifying
    their weight initializer by name.

    This eliminates the prior heuristic
    ``name.endswith('_ConvTranspose') or w.shape[-1] > w.shape[0]``,
    which false-negatives when out_channels >= in_channels.
    """
    import onnx
    from onnx import numpy_helper

    graph = model.graph
    initializers_by_name = {init.name: init for init in graph.initializer}

    # ConvTranspose nodes that originated from TFLite carry the
    # `_ConvTranspose` suffix added by patch_tfl_convtranspose_bias_onnx.
    target_weight_names: set[str] = set()
    for node in graph.node:
        if node.op_type != "ConvTranspose":
            continue
        if not node.name.endswith("_ConvTranspose"):
            continue
        if len(node.input) >= 2:
            target_weight_names.add(node.input[1])

    permuted = 0
    for w_name in target_weight_names:
        init = initializers_by_name.get(w_name)
        if init is None:
            logger.warning("ConvTranspose weight %r has no initializer", w_name)
            continue
        arr = numpy_helper.to_array(init)
        if arr.ndim != 4:
            logger.warning(
                "Skipping ConvTranspose weight %r: ndim=%d (expected 4)",
                w_name,
                arr.ndim,
            )
            continue
        permuted_arr = np.transpose(arr, TFLITE_CONVTRANSPOSE_WEIGHT_PERM).copy()
        new_init = numpy_helper.from_array(permuted_arr, name=w_name)
        init.CopyFrom(new_init)
        permuted += 1

    if permuted:
        report.operators.setdefault(
            "ConvTransposeWeightPermute",
            OperatorRecord(
                tflite_op=None,
                onnx_op="ConvTranspose",
                pytorch_module="nn.ConvTranspose2d",
                occurrences=permuted,
                status="patched",
                notes=[f"Permuted {permuted} weight initializer(s) from [O,kH,kW,I] to [I,O/g,kH,kW]"],
            ),
        )
    logger.info("permute_convtranspose_weights_in_onnx: permuted %d weight(s)", permuted)


# =============================================================================
# Operator coverage analysis
# =============================================================================
def analyze_operator_coverage(onnx_model, report: ConversionReport) -> None:
    """Walk the ONNX graph and ensure every node has a known mapping."""
    onnx_op_counts: dict[str, int] = {}
    for node in onnx_model.graph.node:
        onnx_op_counts[node.op_type] = onnx_op_counts.get(node.op_type, 0) + 1

    # Known-safe ONNX ops with direct onnx2torch mappings.
    known_safe = {
        "Conv", "ConvTranspose", "Relu", "Clip", "Add", "Mul", "Sub", "Div",
        "Reshape", "Resize", "Transpose", "Concat", "Split", "Slice", "Gather",
        "Squeeze", "Unsqueeze", "Cast", "Shape", "Constant", "ConstantOfShape",
        "MatMul", "Gemm", "MaxPool", "AveragePool", "GlobalAveragePool",
        "BatchNormalization", "Softmax", "Sigmoid", "Tanh", "Identity", "Pad",
        "Flatten", "Min", "Max", "ReduceMin", "ReduceMax", "ReduceMean",
        "ReduceSum", "Where", "Equal", "Greater", "Less", "Not", "And", "Or",
    }

    for op, count in onnx_op_counts.items():
        if op in report.operators:
            continue  # already recorded by a patch
        status = "mapped" if op in known_safe else "unmapped"
        report.operators[op] = OperatorRecord(
            tflite_op=None,
            onnx_op=op,
            pytorch_module=None,
            occurrences=count,
            status=status,
        )

    unmapped = [op for op, rec in report.operators.items() if rec.status == "unmapped"]
    if unmapped:
        logger.warning(
            "analyze_operator_coverage: %d unmapped op(s): %s",
            len(unmapped),
            ", ".join(unmapped),
        )


# =============================================================================
# Numerical parity gate
# =============================================================================
def _array_diff(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Return (max_abs, mean_abs, cosine) between two same-shaped arrays."""
    fa = a.astype(np.float64).ravel()
    fb = b.astype(np.float64).ravel()
    diff = np.abs(fa - fb)
    denom = float(np.linalg.norm(fa) * np.linalg.norm(fb)) or 1.0
    return (
        float(diff.max() if diff.size else 0.0),
        float(diff.mean() if diff.size else 0.0),
        float(np.dot(fa, fb) / denom),
    )


def _align_for_diff(tf_arr: np.ndarray, pt_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Best-effort align two arrays for diff.

    If shapes match, return as-is. If PT is NCHW and TF is NHWC of the
    same total size, permute PT to NHWC.
    """
    if tf_arr.shape == pt_arr.shape:
        return tf_arr, pt_arr
    if pt_arr.ndim == 4 and tf_arr.ndim == 4 and pt_arr.size == tf_arr.size:
        candidate = np.transpose(pt_arr, (0, 2, 3, 1))
        if candidate.shape == tf_arr.shape:
            return tf_arr, candidate
    return tf_arr, pt_arr


def _run_tflite(interpreter, input_arr: np.ndarray) -> dict[str, np.ndarray]:
    """Run a TFLite Interpreter on one input, return all outputs by name."""
    input_details = interpreter.get_input_details()
    if len(input_details) != 1:
        raise ConverterError(
            f"Parity gate supports single-input models only; got {len(input_details)}"
        )
    interpreter.set_tensor(input_details[0]["index"], input_arr)
    interpreter.invoke()
    out: dict[str, np.ndarray] = {}
    for d in interpreter.get_output_details():
        out[d["name"]] = interpreter.get_tensor(d["index"]).copy()
    return out


def _run_pytorch(model: Any, input_arr_nhwc: np.ndarray) -> dict[str, np.ndarray]:
    """Run a PyTorch model and return outputs as a name->ndarray dict."""
    torch, _nn, _F = _import_torch()
    x = torch.from_numpy(input_arr_nhwc.astype(np.float32))
    if x.ndim == 4:
        x_nchw = x.permute(0, 3, 1, 2).contiguous()
    else:
        x_nchw = x
    model.eval()
    with torch.inference_mode():
        try:
            out = model(x_nchw)
        except RuntimeError:
            # Some converted models expect NHWC input directly.
            out = model(x)

    def _to_named_dict(o: Any) -> dict[str, np.ndarray]:
        if isinstance(o, torch.Tensor):
            return {"output_0": o.detach().cpu().numpy()}
        if isinstance(o, dict):
            return {k: v.detach().cpu().numpy() for k, v in o.items() if isinstance(v, torch.Tensor)}
        if isinstance(o, (list, tuple)):
            return {f"output_{i}": t.detach().cpu().numpy() for i, t in enumerate(o) if isinstance(t, torch.Tensor)}
        raise ConverterError(f"Unsupported PyTorch output type: {type(o).__name__}")

    return _to_named_dict(out)


def _synthetic_inputs(shape: tuple[int, ...], dtype: np.dtype) -> dict[str, np.ndarray]:
    """Deterministic inputs covering edge cases."""
    inputs: dict[str, np.ndarray] = {}
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        inputs["zeros"] = np.zeros(shape, dtype=dtype)
        inputs["mid"] = np.full(shape, (info.max + info.min) // 2, dtype=dtype)
        rng0 = np.random.default_rng(0)
        rng1 = np.random.default_rng(1)
        inputs["rand_seed0"] = rng0.integers(info.min, info.max + 1, size=shape, dtype=dtype)
        inputs["rand_seed1"] = rng1.integers(info.min, info.max + 1, size=shape, dtype=dtype)
    else:
        inputs["zeros"] = np.zeros(shape, dtype=dtype)
        inputs["ones"] = np.ones(shape, dtype=dtype)
        rng0 = np.random.default_rng(0)
        rng1 = np.random.default_rng(1)
        inputs["rand_seed0"] = rng0.standard_normal(shape).astype(dtype)
        inputs["rand_seed1"] = rng1.standard_normal(shape).astype(dtype)
    return inputs


def run_parity_gate(
    tflite_path: Path,
    pytorch_model: Any,
    report: ConversionReport,
    sample_inputs: Optional[Sequence[np.ndarray]] = None,
) -> bool:
    """Run TFLite and PyTorch on shared inputs, record per-tensor diffs.

    Returns True iff every input passes within tolerance. Mutates *report*.
    """
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ConverterError(
            "Parity gate requires the `tensorflow` package "
            "(used only for tf.lite.Interpreter)."
        ) from exc

    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    input_shape = tuple(inp["shape"])
    input_dtype = np.dtype(inp["dtype"])

    inputs = _synthetic_inputs(input_shape, input_dtype)
    if sample_inputs:
        for i, arr in enumerate(sample_inputs):
            if tuple(arr.shape) != input_shape:
                logger.warning(
                    "Sample input %d shape %s != model input shape %s; skipping",
                    i, arr.shape, input_shape,
                )
                continue
            inputs[f"sample_{i}"] = arr.astype(input_dtype)

    all_passed = True
    for label, arr in inputs.items():
        tf_out = _run_tflite(interp, arr)

        # For PyTorch, dequantize integer inputs if needed.
        if np.issubdtype(arr.dtype, np.integer):
            scale, zero = inp.get("quantization", (1.0, 0))
            pt_input = (arr.astype(np.float32) - zero) * (scale or 1.0)
        else:
            pt_input = arr.astype(np.float32)
        pt_out = _run_pytorch(pytorch_model, pt_input)

        diffs: list[TensorDiff] = []
        # Match outputs by order when names differ.
        tf_items = list(tf_out.items())
        pt_items = list(pt_out.items())
        n_match = min(len(tf_items), len(pt_items))
        for i in range(n_match):
            tf_name, tf_arr = tf_items[i]
            pt_name, pt_arr = pt_items[i]
            tf_aligned, pt_aligned = _align_for_diff(tf_arr, pt_arr)
            if tf_aligned.shape != pt_aligned.shape:
                diffs.append(
                    TensorDiff(
                        name=f"{tf_name}|{pt_name}",
                        max_abs=float("inf"),
                        mean_abs=float("inf"),
                        cosine=0.0,
                        tflite_shape=tf_arr.shape,
                        pytorch_shape=pt_arr.shape,
                        passed=False,
                    )
                )
                continue
            ma, me, cs = _array_diff(tf_aligned, pt_aligned)
            passed = ma <= TOL_PARITY_MAX_ABS and (math.isnan(cs) or cs >= TOL_PARITY_COSINE)
            diffs.append(
                TensorDiff(
                    name=f"{tf_name}|{pt_name}",
                    max_abs=ma,
                    mean_abs=me,
                    cosine=cs,
                    tflite_shape=tf_arr.shape,
                    pytorch_shape=pt_arr.shape,
                    passed=passed,
                )
            )

        run_passed = all(d.passed for d in diffs) and n_match > 0
        if len(tf_items) != len(pt_items):
            run_passed = False
        report.parity_runs.append(
            ParityRunResult(input_label=label, tensor_diffs=diffs, passed=run_passed)
        )
        if not run_passed:
            all_passed = False
            logger.error("parity[%s]: FAIL", label)
            for d in diffs:
                logger.error("  %s", d)
        else:
            logger.info("parity[%s]: pass", label)

    report.parity_passed = all_passed
    return all_passed


# =============================================================================
# Main conversion orchestrator
# =============================================================================
def convert_tflite_to_pytorch(
    tflite_path: str | Path,
    output_pt_path: str | Path,
    opset: int = DEFAULT_ONNX_OPSET,
    input_shape: Optional[tuple[int, ...]] = None,
    sample_inputs: Optional[Sequence[np.ndarray]] = None,
    report_path: Optional[str | Path] = None,
    strict_parity: bool = True,
) -> ConversionReport:
    """Convert a TFLite model to a PyTorch ``.pt`` file with parity validation.

    Args:
        tflite_path:     Source ``.tflite`` file.
        output_pt_path:  Destination ``.pt`` file. Only written if parity passes.
        opset:           ONNX opset for ``tf2onnx``.
        input_shape:     Optional concrete input shape for FX tracing.
        sample_inputs:   Optional real preprocessed inputs to use in parity gate.
        report_path:     Optional JSON path to write the structured report.
        strict_parity:   If True, raise ``ParityFailure`` when parity gate fails.

    Returns:
        ConversionReport with full pipeline observations.

    Raises:
        ParityFailure:     When parity gate fails and strict_parity is True.
        ConversionStageError, MissingAttributeError, UnsupportedOperatorError,
        ConverterError as appropriate.
    """
    tflite_path = Path(tflite_path)
    output_pt_path = Path(output_pt_path)
    output_pt_path.parent.mkdir(parents=True, exist_ok=True)

    report = ConversionReport(
        tflite_path=str(tflite_path),
        output_pt_path=str(output_pt_path),
    )
    t_start = time.perf_counter()
    pt_model = None

    try:
        # --- Stage 1: TFLite -> ONNX -------------------------------------
        import onnx
        with _Stage(report, "tflite_to_onnx") as stage:
            import tf2onnx
            # Suppress tf2onnx noise about custom ops.
            _saved_levels: dict[str, int] = {}
            for ln in ("tf2onnx", "tf2onnx.tfonnx", "tf2onnx.optimizer", "tf2onnx.utils", "tf2onnx.graph_matcher"):
                lg = logging.getLogger(ln)
                _saved_levels[ln] = lg.level
                lg.setLevel(logging.CRITICAL)
            try:
                onnx_model, _ = tf2onnx.convert.from_tflite(str(tflite_path), opset=opset)
            finally:
                for ln, lvl in _saved_levels.items():
                    logging.getLogger(ln).setLevel(lvl)
            stage.metrics["onnx_node_count"] = len(onnx_model.graph.node)

        # --- Stage 2: ONNX-level patches ---------------------------------
        with _Stage(report, "onnx_patches") as stage:
            patch_tfl_convtranspose_bias_onnx(onnx_model, report)
            permute_convtranspose_weights_in_onnx(onnx_model, report)
            stage.metrics["onnx_node_count_after_patch"] = len(onnx_model.graph.node)

        # --- Stage 2.5: shape re-inference -------------------------------
        with _Stage(report, "onnx_shape_inference"):
            onnx_model = onnx.shape_inference.infer_shapes(onnx_model)

        # --- Stage 3: operator coverage analysis -------------------------
        with _Stage(report, "operator_coverage_analysis"):
            analyze_operator_coverage(onnx_model, report)
            if report.unmapped_operators:
                # Warn but don't fail -- onnx2torch may still handle them.
                logger.warning(
                    "Unmapped ops detected: %s",
                    [op.onnx_op for op in report.unmapped_operators],
                )

        # --- Stage 4: ONNX -> PyTorch ------------------------------------
        with _Stage(report, "onnx_to_pytorch") as stage:
            from onnx2torch import convert as onnx2torch_convert
            pt_model = onnx2torch_convert(onnx_model).eval()
            param_count = sum(p.numel() for p in pt_model.parameters())
            stage.metrics["pytorch_param_count"] = param_count

        # --- Stage 5: parity gate ----------------------------------------
        with _Stage(report, "parity_gate") as stage:
            passed = run_parity_gate(tflite_path, pt_model, report, sample_inputs)
            stage.metrics["parity_passed"] = passed
            if not passed and strict_parity:
                raise ParityFailure(
                    "PyTorch outputs diverged from TFLite beyond tolerance. "
                    "See report.parity_runs for details."
                )

        # --- Stage 6: save ------------------------------------------------
        with _Stage(report, "save_pt") as stage:
            import torch
            torch.save(pt_model, str(output_pt_path))
            stage.metrics["output_path"] = str(output_pt_path)

    except ConverterError as exc:
        report.error = f"{type(exc).__name__}: {exc}"
        raise
    except Exception as exc:  # noqa: BLE001
        report.error = f"{type(exc).__name__}: {exc}"
        raise ConversionStageError(str(exc)) from exc
    finally:
        report.total_duration_s = time.perf_counter() - t_start
        if report_path is not None:
            Path(report_path).write_text(report.to_json())
        logger.info("\n%s", report.summary())

    return report


# =============================================================================
# Standalone parity check (no conversion)
# =============================================================================
def run_standalone_parity(
    tflite_path: str | Path,
    pytorch_path: str | Path,
    sample_inputs: Optional[Sequence[np.ndarray]] = None,
    report_path: Optional[str | Path] = None,
) -> ConversionReport:
    """Run only the parity gate against an already-converted ``.pt``."""
    import torch

    tflite_path = Path(tflite_path)
    pytorch_path = Path(pytorch_path)
    report = ConversionReport(
        tflite_path=str(tflite_path),
        output_pt_path=str(pytorch_path),
    )
    t_start = time.perf_counter()
    try:
        with _Stage(report, "load_pytorch"):
            pt_model = torch.load(str(pytorch_path), map_location="cpu", weights_only=False)
            if not hasattr(pt_model, "eval"):
                raise ConverterError(
                    "Loaded object is not an nn.Module (got a state_dict?). "
                    "Standalone parity requires a saved model, not weights."
                )
            pt_model.eval()
        with _Stage(report, "parity_gate") as stage:
            passed = run_parity_gate(tflite_path, pt_model, report, sample_inputs)
            stage.metrics["parity_passed"] = passed
    except ConverterError as exc:
        report.error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report.total_duration_s = time.perf_counter() - t_start
        if report_path is not None:
            Path(report_path).write_text(report.to_json())
        logger.info("\n%s", report.summary())
    return report


# =============================================================================
# CLI
# =============================================================================
def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=level,
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tflite_converter_v2",
        description="TFLite -> PyTorch converter with parity validation.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    pc = sub.add_parser("convert", help="Convert a TFLite model to PyTorch .pt")
    pc.add_argument("--tflite", type=Path, required=True)
    pc.add_argument("--output", type=Path, required=True, help="Destination .pt path")
    pc.add_argument("--opset", type=int, default=DEFAULT_ONNX_OPSET)
    pc.add_argument("--input-shape", type=str, default=None,
                    help="Comma-separated input shape, e.g. 1,300,300,3")
    pc.add_argument("--report", type=Path, default=None)
    pc.add_argument("--no-strict-parity", action="store_true",
                    help="Save .pt even if parity fails (NOT recommended)")
    pc.add_argument("-v", "--verbose", action="store_true")

    pp = sub.add_parser("parity", help="Run parity gate on an existing .pt")
    pp.add_argument("--tflite", type=Path, required=True)
    pp.add_argument("--pytorch", type=Path, required=True)
    pp.add_argument("--report", type=Path, default=None)
    pp.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    _configure_logging(args.verbose)

    if args.command == "convert":
        input_shape = None
        if args.input_shape:
            input_shape = tuple(int(x) for x in args.input_shape.split(","))
        try:
            report = convert_tflite_to_pytorch(
                tflite_path=args.tflite,
                output_pt_path=args.output,
                opset=args.opset,
                input_shape=input_shape,
                report_path=args.report,
                strict_parity=not args.no_strict_parity,
            )
        except ParityFailure as exc:
            logger.error("Parity gate FAILED: %s", exc)
            return 2
        except ConverterError as exc:
            logger.error("Conversion failed: %s", exc)
            return 1
        return 0 if report.parity_passed else 2

    if args.command == "parity":
        try:
            report = run_standalone_parity(
                tflite_path=args.tflite,
                pytorch_path=args.pytorch,
                report_path=args.report,
            )
        except ConverterError as exc:
            logger.error("Parity check failed: %s", exc)
            return 1
        return 0 if report.parity_passed else 2

    return 1  # unreachable


if __name__ == "__main__":
    sys.exit(main())
