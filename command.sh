mkdir -p output/baseline_voc && source executorch-toolkit/.venv/bin/activate && python executorch-toolkit/evaluation/mobilenetv2/evaluate.py --data-path ~/.cache/kagglehub/dataset/watanabe2362/voctrainval-11may2012/version/1/VOCdevkit/VOC2012/JPEGImages --coco-annotations dataset/voc2012_as_coco/instances_voc2012_val.json --tflite-model model_sources/MobileNetV2/weights/model.tflite --pytorch-weights model_sources/MobileNetV2/weights/mb2-ssd-lite-mp-0_686.pth --pytorch-source model_sources/MobileNetV2/src/pytorch/pytorch-ssd --results-dir output/baseline_voc --max-samples 5000 2>&1 | tee output/baseline_voc/run.log


python dataset/scripts/voc_to_coco.py --voc-root ~/.cache/kagglehub/dataset/watanabe2362/voctrainval-11may2012/version/1/VOCdevkit/VOC2012 --split val --output dataset/voc2012_as_coco/instances_voc2012_val.json


ls -lh dataset/voc2012_as_coco/instances_voc2012_val.json


python dataset/scripts/seg_to_coco.py --voc-root ~/.cache/kagglehub/dataset/watanabe2362/voctrainval-11may2012/version/1/VOCdevkit/VOC2012 --output dataset/voc2012_as_coco/instances_voc2012_seg.json


mkdir -p output/baseline_voc_seg && source executorch-toolkit/.venv/bin/activate && python executorch-toolkit/evaluation/mobilenetv2/evaluate.py --data-path ~/.cache/kagglehub/dataset/watanabe2362/voctrainval-11may2012/version/1/VOCdevkit/VOC2012/JPEGImages --coco-annotations dataset/voc2012_as_coco/instances_voc2012_seg.json --tflite-model model_sources/MobileNetV2/weights/model.tflite --pytorch-weights model_sources/MobileNetV2/weights/mb2-ssd-lite-mp-0_686.pth --pytorch-source model_sources/MobileNetV2/src/pytorch/pytorch-ssd --results-dir output/baseline_voc_seg --max-samples 2913 2>&1 | tee output/baseline_voc_seg/run.log


ls ~/.cache/kagglehub/dataset/watanabe2362/voctrainval-11may2012/version/1/VOCdevkit/VOC2012/

python dataset/scripts/voc_to_coco.py --voc-root ~/.cache/kagglehub/datasets/watanabe2362/voctrainval-11may2012/versions/1/VOCdevkit/VOC2012 --split val --output dataset/voc2012_as_coco/instances_voc2012_val.json


python executorch-toolkit/evaluation/mobilenetv2/evaluate.py --data-path ~/.cache/kagglehub/datasets/watanabe2362/voctrainval-11may2012/versions/1/VOCdevkit/VOC2012/JPEGImages --coco-annotations dataset/voc2012_as_coco/instances_voc2012_val.json --tflite-model model_sources/MobileNetV2/weights/model.tflite --pytorch-weights model_sources/MobileNetV2/weights/mb2-ssd-lite-mp-0_686.pth --pytorch-source model_sources/MobileNetV2/src/pytorch/pytorch-ssd --results-dir output/baseline_voc --max-samples 5823 2>&1 | tee output/baseline_voc/run.log
