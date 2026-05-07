#!/usr/bin/env python3
"""convert_onnx_pytorch.py

Convert a (Q/DQ-free) ONNX model to a PyTorch .pt file via onnx2torch.

This is the third stage of the MobileNetV2 conversion pipeline:

    1. convert_tflite_to_onnx.py   TFLite (INT8) -> ONNX (with Q/DQ ops)
    2. dequantize_onnx.py          ONNX with Q/DQ -> ONNX (FP32, Q/DQ-free)
    3. convert_onnx_pytorch.py     ONNX (FP32)    -> PyTorch .pt   <-- this file

The script expects an ONNX graph that is already free of QuantizeLinear
and DequantizeLinear ops -- otherwise onnx2torch raises NotImplementedError.
A pre-flight check at the start of the run surfaces this clearly rather
than letting it explode three layers deep in onnx2torch.

After conversion the script does a smoke test: builds a zero-tensor of
the configured input shape, runs a forward pass in eval / no_grad mode,
and confirms the model produces output. This catches silent shape
mismatches between the ONNX input signature and what we expect.

Usage
-----
    python model_sources/MobileNetV2/scripts/convert_onnx_pytorch.py \\
        --input  model_sources/MobileNetV2/weights/model.fp32.onnx \\
        --output model_sources/MobileNetV2/weights/model.pt

Exit codes
----------
    0  success
    1  filesystem / load error
    2  input ONNX still contains Q/DQ ops (run dequantize_onnx.py first)
    3  onnx2torch conversion raised
    4  smoke-test forward pass failed
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import onnx
import torch
from onnx import ModelProto
from onnx2torch import convert

LOG = logging.getLogger("convert_onnx_pytorch")

_QUANT_OP = "QuantizeLinear"
_DEQUANT_OP = "DequantizeLinear"


# --------------------------------------------------------------------------- #
# Pre-flight check                                                            #
# --------------------------------------------------------------------------- #


def _count_qdq_ops(model: ModelProto) -> int:
    return sum(1 for n in model.graph.node if n.op_type in (_QUANT_OP, _DEQUANT_OP))


# --------------------------------------------------------------------------- #
# Smoke test                                                                  #
# --------------------------------------------------------------------------- #


def _smoke_test(
    pytorch_model: torch.nn.Module,
    input_shape: tuple[int, int, int, int],
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Run a single forward pass with zero input to verify the model is callable.

    The result is returned for shape logging but otherwise discarded.
    Any exception raised here propagates -- the caller treats that as
    smoke-test failure.
    """
    pytorch_model.eval()
    with torch.no_grad():
        dummy = torch.zeros(*input_shape, dtype=torch.float32)
        return pytorch_model(dummy)


def _describe_output(out: object) -> str:
    """Render a forward-pass output (tensor, tuple, dict, list) as a shape summary."""
    if isinstance(out, torch.Tensor):
        return f"Tensor{tuple(out.shape)} dtype={out.dtype}"
    if isinstance(out, (list, tuple)):
        return "(" + ", ".join(_describe_output(o) for o in out) + ")"
    if isinstance(out, dict):
        return "{" + ", ".join(f"{k}: {_describe_output(v)}" for k, v in out.items()) + "}"
    return repr(type(out))


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert an FP32 ONNX model to a PyTorch .pt file via onnx2torch.",
    )
    p.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the input .onnx file (must be Q/DQ-free).",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write the output .pt file.",
    )
    p.add_argument(
        "--input-shape",
        nargs=4,
        type=int,
        default=[1, 3, 300, 300],
        metavar=("N", "C", "H", "W"),
        help="Input tensor shape used for the post-conversion smoke test (NCHW).",
    )
    p.add_argument(
        "--skip-smoke-test",
        action="store_true",
        help="Skip the forward-pass smoke test (not recommended).",
    )
    p.add_argument("--verbose", action="store_true", help="Enable DEBUG logging.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-7s %(message)s",
    )

    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if not input_path.is_file():
        LOG.error("input file not found: %s", input_path)
        return 1

    LOG.info("loading: %s", input_path)
    try:
        onnx_model = onnx.load(str(input_path))
    except Exception as e:
        LOG.error("failed to load ONNX: %s", e)
        return 1

    # Pre-flight: refuse to attempt conversion on a graph that still
    # contains Q/DQ ops. onnx2torch will raise NotImplementedError on
    # the first one it sees; we'd rather report it here with a clear
    # remediation message.
    n_qdq = _count_qdq_ops(onnx_model)
    if n_qdq > 0:
        LOG.error(
            "input ONNX contains %d QuantizeLinear/DequantizeLinear op(s). "
            "onnx2torch cannot convert these. Run dequantize_onnx.py first "
            "to fold Q/DQ pairs into FP32 constants.",
            n_qdq,
        )
        return 2

    LOG.info("converting ONNX -> PyTorch (this may take a moment)...")
    try:
        pytorch_model = convert(onnx_model)
    except Exception as e:
        LOG.error("onnx2torch conversion failed: %s", e)
        return 3

    n_params = sum(p.numel() for p in pytorch_model.parameters())
    LOG.info("conversion complete: %s parameters across %d tensors",
             f"{n_params:,}",
             sum(1 for _ in pytorch_model.parameters()))

    if not args.skip_smoke_test:
        shape = tuple(args.input_shape)
        LOG.info("smoke test: forward pass with zeros(%s)...", list(shape))
        try:
            out = _smoke_test(pytorch_model, shape)  # type: ignore[arg-type]
        except Exception as e:
            LOG.error("smoke-test forward pass failed: %s", e)
            LOG.error(
                "this usually means the configured input shape %s does not "
                "match what the converted model expects. Re-check the TFLite "
                "signature with inspect_tflite_signature.py.",
                list(shape),
            )
            return 4
        LOG.info("smoke test OK -- output: %s", _describe_output(out))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pytorch_model, str(output_path))
    LOG.info("wrote: %s", output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
