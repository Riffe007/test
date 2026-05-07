#!/usr/bin/env bash
# convert.sh
#
# Orchestrates the three-stage MobileNetV2 conversion pipeline:
#
#     model.tflite   -- convert_tflite_to_onnx.py -->   model.onnx
#     model.onnx     -- dequantize_onnx.py        -->   model.fp32.onnx
#     model.fp32.onnx -- convert_onnx_to_pytorch.py  -->   model.pt
#
# Run from the scripts/ directory:
#
#     cd model_sources/MobileNetV2/scripts
#     ./convert.sh
#
# All three Python scripts use distinct non-zero exit codes for distinct
# failure modes; this wrapper propagates them. 'set -e' aborts on the
# first failure rather than letting later stages run on stale or missing
# inputs.

set -euo pipefail

# Resolve paths relative to this script's location, so the pipeline works
# whether you invoke it as ./convert.sh, bash convert.sh, or with an
# absolute path from anywhere.
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WEIGHTS_DIR="$( cd -- "${SCRIPT_DIR}/../weights" &> /dev/null && pwd )"

TFLITE="${WEIGHTS_DIR}/model.tflite"
ONNX_RAW="${WEIGHTS_DIR}/model.onnx"
ONNX_FP32="${WEIGHTS_DIR}/model.fp32.onnx"
PT_OUT="${WEIGHTS_DIR}/model.pt"

# Refuse to start without the input file present.
if [[ ! -f "${TFLITE}" ]]; then
    echo "convert.sh: input not found: ${TFLITE}" >&2
    exit 1
fi

echo "==> [1/3] TFLite -> ONNX"
python3 "${SCRIPT_DIR}/convert_tflite_to_onnx.py" \
    --input  "${TFLITE}" \
    --output "${ONNX_RAW}"

echo
echo "==> [2/3] ONNX -> ONNX (Q/DQ-free)"
python3 "${SCRIPT_DIR}/dequantize_onnx.py" \
    --input  "${ONNX_RAW}" \
    --output "${ONNX_FP32}"

echo
echo "==> [3/3] ONNX -> PyTorch"
python3 "${SCRIPT_DIR}/convert_onnx_to_pytorch.py" \
    --input  "${ONNX_FP32}" \
    --output "${PT_OUT}"

echo
echo "==> done: ${PT_OUT}"
