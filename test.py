cd ~/Documents/projects/MetaExecuTorch/model_sources/MobileNetV2

# 1. Clone qfgaohao (we need its model definition code)
mkdir -p tools
[ -d tools/pytorch-ssd ] || git clone https://github.com/qfgaohao/pytorch-ssd.git tools/pytorch-ssd

# 2. Download MobileNetV2 SSD-Lite VOC weights
[ -f weights/mb2-ssd-lite.pth ] || wget -q --show-progress \
  -O weights/mb2-ssd-lite.pth \
  https://storage.googleapis.com/models-hao/mb2-ssd-lite-mp-0_686.pth

# 3. Test image (canonical COCO val sample — two cats on a couch)
mkdir -p test_data
[ -f test_data/test.jpg ] || wget -q -O test_data/test.jpg \
  https://images.cocodataset.org/val2017/000000039769.jpg

# 4. Labels
mkdir -p labels
[ -f labels/coco_labels.txt ] || wget -q -O labels/coco_labels.txt \
  https://raw.githubusercontent.com/google-coral/test_data/master/coco_labels.txt
[ -f labels/voc_labels.txt ] || cp tools/pytorch-ssd/models/voc-model-labels.txt labels/voc_labels.txt

# 5. Deps (qfgaohao needs cv2; you should already have torch + tf)
pip install opencv-python pillow

# 6. Verify
ls -lah weights/ test_data/ labels/
