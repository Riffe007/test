cd ~/Documents/projects/MetaExecuTorch/model_sources/MobileNetV2/scripts/

echo "=== exact filename match test ==="
[ -f convert_onnx_pytorch.py ] && echo "FOUND" || echo "NOT FOUND at exact name 'convert_onnx_pytorch.py'"

echo
echo "=== all .py files in this directory ==="
ls -1 *.py

echo
echo "=== same listing, with hidden characters revealed ==="
ls -1 *.py | cat -A
