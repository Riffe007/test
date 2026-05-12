cd ~/Documents/projects/MetaExecuTorch/executorch-toolkit && source .venv/bin/activate && mkdir -p ../output/eval_mobilenet_v2_ssd && PYTHONPATH=../model_sources/MobileNetV2/src/pytorch/pytorch-ssd python evaluation/mobilenetv2/evaluate.py --voc-root ~/.cache/kagglehub/datasets/watanabe2362/voctrainval-11may2012/versions/1/VOCdevkit/VOC2012 --gt-coco-json dataset/voc2012_val_coco.json --tflite-model ../model_sources/MobileNetV2/weights/model.tflite --pytorch-model ../model_sources/MobileNetV2/weights/mobile_net_v2_ssd.pth --model-dir tests/integration/outputs/mobile_net_v2_ssd/basemodel_workflow/models --results-dir ../output/eval_mobilenet_v2_ssd --generate-report 2>&1 | tee ../output/eval_run.log



find ~/Documents/projects/MetaExecuTorch -type f -name "*.json" \( -iname "*voc*" -o -iname "*coco*" \) 2>/dev/null

cd ~/Documents/projects/MetaExecuTorch/executorch-toolkit && source .venv/bin/activate && mkdir -p ../output/eval_mobilenet_v2_ssd && PYTHONPATH=../model_sources/MobileNetV2/src/pytorch/pytorch-ssd python evaluation/mobilenetv2/evaluate.py --voc-root ~/.cache/kagglehub/datasets/watanabe2362/voctrainval-11may2012/versions/1/VOCdevkit/VOC2012 --gt-coco-json ../dataset/voc2012_as_coco/instances_voc2012_val.json --tflite-model ../model_sources/MobileNetV2/weights/model.tflite --pytorch-model ../model_sources/MobileNetV2/weights/mobile_net_v2_ssd.pth --model-dir tests/integration/outputs/mobile_net_v2_ssd/basemodel_workflow/models --results-dir ../output/eval_mobilenet_v2_ssd --generate-report --max-samples 25 2>&1 | tee ../output/eval_smoke.log


grep -nE "^def |^class " ~/Documents/projects/MetaExecuTorch/executorch-toolkit/evaluation/mobilenetv2/generate_report.py && echo "---LOG---" && grep -iE "html|report|generat|error|traceback" ~/Documents/projects/MetaExecuTorch/output/eval_smoke.log | tail -30


python -c "
import json
d = json.load(open('/home/timothy_riffe/Documents/projects/MetaExecuTorch/dataset/voc2012_as_coco/instances_voc2012_val.json'))
print('Categories (id -> name):')
for c in sorted(d['categories'], key=lambda x: x['id']):
    print(f'  {c[\"id\"]}: {c[\"name\"]}')
print('Sample anns:')
for a in d['annotations'][:3]:
    print(f'  image_id={a[\"image_id\"]} cat={a[\"category_id\"]} bbox={a[\"bbox\"]}')
print('Sample image:', {k: d['images'][0][k] for k in ('id','file_name','width','height')})
"
cd ~/Documents/projects/MetaExecuTorch/executorch-toolkit && source .venv/bin/activate && PYTHONPATH=../model_sources/MobileNetV2/src/pytorch/pytorch-ssd python evaluation/mobilenetv2/evaluate.py --voc-root ~/.cache/kagglehub/datasets/watanabe2362/voctrainval-11may2012/versions/1/VOCdevkit/VOC2012 --gt-coco-json ../dataset/voc2012_as_coco/instances_voc2012_val.json --tflite-model ../model_sources/MobileNetV2/weights/model.tflite --pytorch-model ../model_sources/MobileNetV2/weights/mobile_net_v2_ssd.pth --model-dir tests/integration/outputs/mobile_net_v2_ssd/basemodel_workflow/models --results-dir ../output/eval_mobilenet_v2_ssd --generate-report --max-samples 25 2>&1 | tee ../output/eval_smoke2.log
