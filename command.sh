cd ~/Documents/projects/MetaExecuTorch/executorch-toolkit
source .venv/bin/activate
python export/vision/pytorch_to_executorch_vision.py \
    tests/integration/configs/mobile_net_v2_ssd/config_mobile_net_v2_ssd_basemodel.json \
    --generate-report
