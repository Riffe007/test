python -m export.vision.pytorch_to_executorch_vision \
  export/configs/vision/config_mobile_net_v2_ssd.json --generate-report


# 1. Confirm git state (do FIRST, before anything else)
cd ~/Documents/projects/MetaExecuTorch/executorch-toolkit
git status
git log --oneline -5

# 2. Stage COCO dataset
mkdir -p ~/Documents/projects/MetaExecuTorch/dataset/coco_val2017
cd ~/Documents/projects/MetaExecuTorch/dataset/coco_val2017
unzip "/mnt/c/Users/timothy_riffe/Downloads/val2017 1.zip"
unzip "/mnt/c/Users/timothy_riffe/Downloads/annotations_trainval2017 1.zip"
ls  # should show val2017/ and annotations/

# 3. Stage TFLite model
mkdir -p ~/Documents/projects/MetaExecuTorch/model_sources/MobileNetV2_TFLite
cp "/mnt/c/Users/timothy_riffe/Downloads/ssd_mobilenet_v2_coco_quant_postprocess.tflite" \
   ~/Documents/projects/MetaExecuTorch/model_sources/MobileNetV2_TFLite/

# 4. Install TFLite runtime in toolkit venv
cd ~/Documents/projects/MetaExecuTorch/executorch-toolkit
source .venv/bin/activate
pip install tflite-runtime
