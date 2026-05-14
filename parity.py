"""Preprocess a single image into the exact tensor format the TFLite model expects.

Used by the converter's parity gate to verify TFLite vs PyTorch agreement on
real input data (not just synthetic tensors).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def preprocess_for_tflite(
    image_path: str | Path,
    input_shape: tuple[int, ...],
    input_dtype: np.dtype,
    quant_scale: float = 0.0,
    quant_zero_point: int = 0,
) -> np.ndarray:
    """Return an array matching the TFLite model's input contract exactly.

    Args:
        image_path:        Path to source image.
        input_shape:       Model input shape, e.g. (1, 300, 300, 3) NHWC.
        input_dtype:       Model input dtype (uint8 for quantized, float32 otherwise).
        quant_scale:       Quantization scale (0.0 if float input).
        quant_zero_point:  Quantization zero point (0 if float input).
    """
    _, h, w, c = input_shape
    if c != 3:
        raise ValueError(f"Expected 3-channel input, got {c}")

    img = Image.open(image_path).convert("RGB").resize((w, h), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)  # HWC, [0, 255]

    if np.issubdtype(input_dtype, np.integer):
        # Quantized model: model expects uint8 directly.
        return arr.astype(input_dtype)[None, ...]

    # Float model: MobileNetV2-SSD convention is [-1, 1] via (x - 127.5) / 127.5
    normalized = (arr - 127.5) / 127.5
    return normalized.astype(input_dtype)[None, ...]
