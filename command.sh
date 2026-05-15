mkdir -p ~/Documents/projects/MetaExecuTorch/model_sources/MobileNetV2_TFLite
cp "/mnt/c/Users/timothy_riffe/Downloads/tf2_ssd_mobilenet_v2_coco17_ptq.tflite" \
   ~/Documents/projects/MetaExecuTorch/model_sources/MobileNetV2_TFLite/
ls ~/Documents/projects/MetaExecuTorch/model_sources/MobileNetV2_TFLite/



find ~/Documents/projects/MetaExecuTorch -type d -name "vision" 2>/dev/null


python evaluation/mobilenetv2/compare_tflite_pytorch_coco.py \
    --qfgaohao-repo /correct/path/here


python evaluation/mobilenetv2/compare_tflite_pytorch_coco.py \
    --qfgaohao-repo /correct/path/here


ls ~/Documents/projects/MetaExecuTorch/model_sources/MobileNetV2/src/pytorch/pytorch-ssd/vision/ssd/mobilenetv2_ssd_lite.py
ls ~/Documents/projects/MetaExecuTorch/model_sources/MobileNetV2/src/pytorch/pytorch-ssd/vision/__init__.py
ls ~/Documents/projects/MetaExecuTorch/model_sources/MobileNetV2/src/pytorch/pytorch-ssd/vision/ssd/__init__.py


cd ~/Documents/projects/MetaExecuTorch/model_sources/MobileNetV2/src/pytorch/pytorch-ssd
python -c "from vision.ssd.mobilenetv2_ssd_lite import create_mobilenetv2_ssd_lite; print('OK')"
