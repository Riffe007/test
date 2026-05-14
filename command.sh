cd ~/Documents/projects/MetaExecuTorch/executorch-toolkit && \
python -c "
import json
p = 'export/configs/vision/config_mobile_net_v2_ssd.json'
c = json.load(open(p))
c['export']['benchmark'] = True
c['export']['benchmark_iterations'] = 50
json.dump(c, open(p, 'w'), indent=2)
print('benchmark=true, iters=50')
" && \
source .venv/bin/activate && \
python export/vision/pytorch_to_executorch_vision.py \
  export/configs/vision/config_mobile_net_v2_ssd.json --generate-report
