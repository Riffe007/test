cd ~/Documents/projects/MetaExecuTorch/executorch-toolkit && \
python -c "
import json
p = 'export/configs/vision/config_mobile_net_v2_ssd.json'
c = json.load(open(p))
c['export']['enable_profiling'] = True
json.dump(c, open(p, 'w'), indent=2)
print('enable_profiling = true')
" && \
source .venv/bin/activate && \
python export/vision/pytorch_to_executorch_vision.py \
  export/configs/vision/config_mobile_net_v2_ssd.json --generate-report


  cd ~/Documents/projects/MetaExecuTorch/executorch-toolkit && \
source .venv/bin/activate && \
python evaluation/mobilenetv2/evaluate.py \
  --config export/configs/vision/config_mobile_net_v2_ssd.json
