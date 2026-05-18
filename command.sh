cd ~/Documents/projects/MetaExecuTorch/executorch-toolkit

EVAL_FILE="$(grep -RIl 'MobileNetV2-SSD\|MobileNetV2 SSD\|TFLite.*ExecuTorch\|--config' . \
  --include='evaluate.py' \
  | grep -v 'mediapipe_hair_segmentation' \
  | grep -v '__pycache__' \
  | head -n 1)"

if [ -z "$EVAL_FILE" ]; then
  echo "ERROR: Correct MobileNetV2 SSD evaluate.py was not found."
  echo
  echo "Found these evaluate.py files:"
  find . -type f -name "evaluate.py"
  exit 1
fi

echo "Using evaluator: $EVAL_FILE"

python "$EVAL_FILE" \
  --config export/configs/vision/config_mobile_net_v2_ssd.json
