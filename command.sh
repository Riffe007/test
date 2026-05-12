cat ./executorch-toolkit/tests/integration/mobile_net_v2_ssd/config_mobile_net_v2_ssd_basemodel.json


cp executorch-toolkit/tests/integration/mobile_net_v2_ssd/config_mobile_net_v2_ssd_basemodel.json executorch-toolkit/tests/integration/mobile_net_v2_ssd/config_mobile_net_v2_ssd_basemodel.json.bak


sed -i \
  -e 's|kagglehub/watanabe2362|kagglehub/datasets/watanabe2362|g' \
  -e 's|/version/1/|/versions/1/|g' \
  -e 's|\.\./dataset/samples/coco|/home/timothy_riffe/.cache/kagglehub/datasets/watanabe2362/voctrainval-11may2012/versions/1/VOCdevkit/VOC2012/JPEGImages|g' \
  -e 's|"calibration_samples": 10|"calibration_samples": 256|g' \
  -e 's|"benchmark": false|"benchmark": true|' \
  executorch-toolkit/tests/integration/mobile_net_v2_ssd/config_mobile_net_v2_ssd_basemodel.json



  grep -E "benchmark|calibration_dir|calibration_samples" executorch-toolkit/tests/integration/mobile_net_v2_ssd/config_mobile_net_v2_ssd_basemodel.json


  cd executorch-toolkit && source .venv/bin/activate && PYTHONPATH=../model_sources/MobileNetV2/src/pytorch/pytorch-ssd python export/vision/pytorch_to_executorch_vision.py tests/integration/mobile_net_v2_ssd/config_mobile_net_v2_ssd_basemodel.json --generate-report 2>&1 | tee ../output/export_run.log
