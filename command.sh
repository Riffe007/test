cd ~/Documents/projects/MetaExecuTorch/executorch-toolkit

echo "=== How the toolkit loads the .pth ==="
grep -rn "mobile_net_v2_ssd\.pth\|model_path\|load_state_dict\|torch\.load" export/vision/ 2>/dev/null | grep -v __pycache__ | head -30

echo ""
echo "=== Does the toolkit use qfgaohao's Predictor? ==="
grep -rn "create_mobilenetv2_ssd_lite_predictor\|PredictionTransform\|Predictor(" export/vision/ 2>/dev/null | grep -v __pycache__

echo ""
echo "=== Or does it have its own decode/NMS path? ==="
grep -rn "def predict\|def decode\|def nms\|def postprocess\|_decode_\|_nms_" export/vision/ 2>/dev/null | grep -v __pycache__ | head -20

echo ""
echo "=== Eval scoring entry point ==="
grep -rn "def evaluate\|def eval\|mAP\|compute_metric\|COCOeval\|cocoeval" export/vision/ 2>/dev/null | grep -v __pycache__ | head -20
