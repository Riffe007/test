#!/usr/bin/env python3
"""convert_tflite_to_onnx.py

Convert a TFLite model to ONNX format using tf2onnx.

This is the first stage of the MobileNetV2 conversion pipeline:

    1. convert_tflite_to_onnx.py   TFLite (INT8) -> ONNX (with Q/DQ ops)  <-- this file
    2. dequantize_onnx.py          ONNX with Q/DQ -> ONNX (FP32, Q/DQ-free)
    3. convert_onnx_pytorch.py     ONNX (FP32)    -> PyTorch .pt

This script wraps tf2onnx as a subprocess (the same approach the team
uses for MobileNetV1) but adds:

    * required CLI arguments instead of hardcoded relative paths
    * exit-code propagation -- a tf2onnx failure exits non-zero rather
      than printing 'Conversion process completed' regardless
    * pre-flight checks (input file exists, output dir is writable,
      tf2onnx is installed in the active interpreter)
    * post-flight verification (output ONNX exists, parses, has nodes)

Usage
-----
    python model_sources/MobileNetV2/scripts/convert_tflite_to_onnx.py \\
        --input  model_sources/MobileNetV2/weights/model.tflite \\
        --output model_sources/MobileNetV2/weights/model.onnx

Exit codes
----------
    0  success
    1  filesystem / load error (input missing, output dir unwritable, etc.)
    2  tf2onnx is not installed in the active Python interpreter
    3  tf2onnx subprocess failed (returned non-zero or timed out)
    4  output ONNX missing or unparseable after the subprocess returned 0
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

LOG = logging.getLogger("convert_tflite_to_onnx")

# Default ONNX opset. tf2onnx supports up to 18 at the time of writing;
# 16 is a safe middle ground that has broad downstream tool support
# (onnx2torch, onnxruntime, ExecuTorch). MobileNetV1 in this repo uses 16.
_DEFAULT_OPSET = 16

# Generous ceiling on the subprocess. tf2onnx is slow on first run
# (TensorFlow imports + graph rewriting) but should finish well under
# 5 minutes for any model in this codebase.
_SUBPROCESS_TIMEOUT_SECONDS = 600


# --------------------------------------------------------------------------- #
# Pre-flight                                                                  #
# --------------------------------------------------------------------------- #


def _tf2onnx_is_importable() -> bool:
    """Confirm tf2onnx is importable in the active interpreter.

    We probe by spawning a subprocess that runs 'import tf2onnx',
    rather than importing it here directly. Importing tf2onnx in this
    process would transitively import TensorFlow, which is slow (~5s)
    and prints spurious warnings -- not something to do just to check
    availability.
    """
    proc = subprocess.run(
        [sys.executable, "-c", "import tf2onnx"],
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0


def _is_writable(directory: Path) -> bool:
    """Probe whether the calling process can create files in the given directory."""
    return os.access(str(directory), os.W_OK)


# --------------------------------------------------------------------------- #
# Subprocess                                                                  #
# --------------------------------------------------------------------------- #


def _run_tf2onnx(
    tflite_path: Path,
    onnx_path: Path,
    opset: int,
) -> int:
    """Invoke 'python -m tf2onnx.convert ...'. Returns the subprocess exit code.

    Returns 124 on timeout (matching the GNU 'timeout' command convention).
    """
    cmd = [
        sys.executable,
        "-m", "tf2onnx.convert",
        "--opset", str(opset),
        "--tflite", str(tflite_path),
        "--output", str(onnx_path),
    ]
    LOG.info("running: %s", " ".join(cmd))
    try:
        proc = subprocess.run(
            cmd,
            timeout=_SUBPROCESS_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        LOG.error("tf2onnx timed out after %d seconds", _SUBPROCESS_TIMEOUT_SECONDS)
        return 124
    return proc.returncode


# --------------------------------------------------------------------------- #
# Post-flight                                                                 #
# --------------------------------------------------------------------------- #


def _verify_output_onnx(path: Path) -> tuple[bool, str]:
    """Confirm the produced ONNX file exists, parses, and is non-trivial.

    Returns (ok, message). On success message is a one-line summary
    (size, node count, initializer count); on failure it is the reason.

    onnx is imported lazily here so that this script remains usable in
    environments where only tf2onnx (and its TensorFlow dep) is
    installed -- a degraded but still-functional mode where we skip the
    deep parse and only check that the file is non-empty.
    """
    if not path.is_file():
        return False, f"expected output ONNX not found: {path}"

    size = path.stat().st_size
    if size == 0:
        return False, f"output ONNX is empty: {path}"

    try:
        import onnx
    except ImportError:
        return True, f"output exists ({size:,} bytes) -- onnx not installed, skipped deep parse"

    try:
        model = onnx.load(str(path))
    except Exception as e:
        return False, f"output ONNX failed to parse: {e}"

    n_nodes = len(model.graph.node)
    n_inits = len(model.graph.initializer)
    if n_nodes == 0:
        return False, "output ONNX has zero nodes"

    return True, f"{size:,} bytes, {n_nodes} nodes, {n_inits} initializers"


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert a TFLite model to ONNX format using tf2onnx.",
    )
    p.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the input .tflite file.",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write the output .onnx file.",
    )
    p.add_argument(
        "--opset",
        type=int,
        default=_DEFAULT_OPSET,
        help=f"ONNX opset version (default: {_DEFAULT_OPSET}).",
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

    # Filesystem pre-flight.
    if not input_path.is_file():
        LOG.error("input file not found: %s", input_path)
        return 1
    if input_path.suffix.lower() != ".tflite":
        LOG.warning("input does not have .tflite extension: %s", input_path)

    output_dir = output_path.parent
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        LOG.error("could not create output directory %s: %s", output_dir, e)
        return 1
    if not _is_writable(output_dir):
        LOG.error("output directory is not writable: %s", output_dir)
        return 1

    # tf2onnx pre-flight.
    if not _tf2onnx_is_importable():
        LOG.error(
            "tf2onnx is not installed in %s. Install it with:\n"
            "    pip install tf2onnx",
            sys.executable,
        )
        return 2

    LOG.info("input:  %s (%s bytes)", input_path, f"{input_path.stat().st_size:,}")
    LOG.info("output: %s", output_path)
    LOG.info("opset:  %d", args.opset)

    rc = _run_tf2onnx(input_path, output_path, args.opset)
    if rc != 0:
        LOG.error("tf2onnx exited with code %d", rc)
        return 3

    ok, msg = _verify_output_onnx(output_path)
    if not ok:
        LOG.error("post-flight check failed: %s", msg)
        return 4

    LOG.info("conversion successful: %s", msg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
