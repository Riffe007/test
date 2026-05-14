cd ~/Documents/projects/MetaExecuTorch/executorch-toolkit && \
python -c "
import json
p = 'export/configs/vision/config_mobile_net_v2_ssd.json'
c = json.load(open(p))
c['export']['use_recipes'] = True
json.dump(c, open(p, 'w'), indent=2)
print('use_recipes=true')
" && \
source .venv/bin/activate && \
export PYTHONPATH=\"$(realpath ../model_sources/MobileNetV2/src/pytorch/pytorch-ssd):$PYTHONPATH\" && \
python export/vision/pytorch_to_executorch_vision.py \
  export/configs/vision/config_mobile_net_v2_ssd.json --generate-report
