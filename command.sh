cd ~/Documents/projects/MetaExecuTorch

# 1. Remove obsolete scripts from the abandoned EdgeTPU pipeline
rm -v model_sources/MobileNetV2/scripts/tflite_to_clean_fp32_onnx.py \
      model_sources/MobileNetV2/scripts/compare_models.py \
      model_sources/MobileNetV2/scripts/acquire_models.py \
      model_sources/MobileNetV2/scripts/convert_tflite_to_onnx.py \
      model_sources/MobileNetV2/scripts/dequantize_onnx.py

# 2. Patch qfgaohao's torch.load — add weights_only=False (PyTorch 2.6+ default fix).
#    -i.bak keeps a backup at ssd.py.bak you can delete after verifying.
sed -i.bak \
  's|torch\.load(model, map_location=lambda storage, loc: storage)|torch.load(model, map_location=lambda storage, loc: storage, weights_only=False)|' \
  model_sources/MobileNetV2/src/pytorch/pytorch-ssd/vision/ssd/ssd.py

# 3. Verify the patch landed (should show the line with weights_only=False)
grep -n "torch.load" model_sources/MobileNetV2/src/pytorch/pytorch-ssd/vision/ssd/ssd.py

# 4. Re-run compare.py — qfgaohao block should now load and complete
python3 model_sources/MobileNetV2/scripts/compare.py
