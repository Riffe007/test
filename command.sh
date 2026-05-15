python -m export.vision.pytorch_to_executorch_vision \
  export/configs/vision/config_mobile_net_v2_ssd.json --generate-report
mkdir -p ~/Documents/projects/MetaExecuTorch/model_sources/MobileNetV2_TFLite
cp "/mnt/c/Users/timothy_riffe/Downloads/tf2_ssd_mobilenet_v2_coco17_ptq.tflite" \
   ~/Documents/projects/MetaExecuTorch/model_sources/MobileNetV2_TFLite/
ls ~/Documents/projects/MetaExecuTorch/model_sources/MobileNetV2_TFLite/
