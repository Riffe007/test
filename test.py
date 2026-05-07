"""Programmatic API: TFLite -> PyTorch end-to-end.

For CLI use with progress output, run `run_pipeline.py` (or `convert.sh`).
This module is for calling the conversion from other Python code.
"""

from pathlib import Path
import subprocess
import sys

SCRIPTS = Path(__file__).resolve().parent
WEIGHTS = SCRIPTS.parent / "weights"


def convert_tflite_to_pytorch(quiet: bool = False) -> Path:
    """Run the full pipeline. Returns path to final .pt file.

    Stages:
      1. TFLite -> Clean FP32 ONNX
      2. Clean FP32 ONNX -> PyTorch

    Raises subprocess.CalledProcessError on any stage failure.
    """
    stdout = subprocess.DEVNULL if quiet else None

    subprocess.run(
        [sys.executable, str(SCRIPTS / "tflite_to_clean_fp32_onnx.py")],
        check=True,
        stdout=stdout,
    )
    subprocess.run(
        [sys.executable, str(SCRIPTS / "convert_onnx_to_pytorch.py")],
        check=True,
        stdout=stdout,
    )

    return WEIGHTS / "model.pt"


if __name__ == "__main__":
    out = convert_tflite_to_pytorch()
    print(f"Saved: {out}")
