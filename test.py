cd ~/Documents/projects/MetaExecuTorch/model_sources/MobileNetV2

# 1. Loading code (matches Blazeface convention: src/pytorch/<RepoName>/)
[ -d src/pytorch/pytorch-ssd ] || git clone https://github.com/qfgaohao/pytorch-ssd.git src/pytorch/pytorch-ssd

# 2. Labels
mkdir -p labels
[ -f labels/voc_labels.txt ] || cp src/pytorch/pytorch-ssd/models/voc-model-labels.txt labels/voc_labels.txt
[ -f labels/coco_labels.txt ] || wget -q -O labels/coco_labels.txt https://raw.githubusercontent.com/google-coral/test_data/master/coco_labels.txt

# 3. cv2 (qfgaohao predictor uses it)
python3 -c "import cv2" 2>/dev/null || pip install opencv-python

# 4. Verify everything resolves
ls weights/model.tflite weights/mb2-ssd-lite.pth labels/voc_labels.txt labels/coco_labels.txt src/pytorch/pytorch-ssd/vision/ssd/mobilenet_v2_ssd_lite.py
