cd ~/Documents/projects/MetaExecuTorch && \
source executorch-toolkit/.venv/bin/activate && \
python scripts/compare_tflite_vs_pytorch.py \
  --pt-weights model_sources/MobileNetV2/weights/mb2-ssd-lite-mp-0_686.pth \
  --tflite    model_sources/MobileNetV2/weights/model.tflite \
  --source-path model_sources/MobileNetV2/src/pytorch/pytorch-ssd \
  --gt-json   dataset/voc2012_as_coco/instances_voc2012_val.json \
  --images-dir ~/.cache/kagglehub/datasets/watanabe2362/voctrainval-11may2012/versions/1/VOCdevkit/VOC2012/JPEGImages \
  --output-dir output/comparison \
  --limit 50


  python scripts/compare_tflite_vs_pytorch.py \
  --pt-weights model_sources/MobileNetV2/weights/mb2-ssd-lite-mp-0_686.pth \
  --tflite    model_sources/MobileNetV2/weights/model.tflite \
  --source-path model_sources/MobileNetV2/src/pytorch/pytorch-ssd \
  --gt-json   dataset/voc2012_as_coco/instances_voc2012_val.json \
  --images-dir ~/.cache/kagglehub/datasets/watanabe2362/voctrainval-11may2012/versions/1/VOCdevkit/VOC2012/JPEGImages \
  --output-dir output/comparison
