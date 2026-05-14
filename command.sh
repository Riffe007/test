find ~/Documents/projects/MetaExecuTorch -name "*.tflite" 2>/dev/null
find ~ -name "*.tflite" 2>/dev/null | head -20


find ~/Documents/projects/MetaExecuTorch -name "*.tflite" 2>/dev/null

grep -n "tflite\|\.tflite\|Interpreter" executorch-toolkit/export/vision/compare_tflite_vs_pytorch.py | head -30




cd ~/Documents/projects/MetaExecuTorch
mkdir -p external/models
cd external/models

wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
tar -xzf ssd_mobilenet_v2_coco_2018_03_29.tar.gz

cd ~/Documents/projects/MetaExecuTorch/executorch-toolkit
source .venv/bin/activate

python - <<'PY'
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model(
    "../external/models/ssd_mobilenet_v2_coco_2018_03_29/saved_model"
)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS,
]
m = converter.convert()
open("../external/models/mobile_net_v2_ssd.tflite", "wb").write(m)
print(f"Wrote {len(m)/1e6:.2f} MB")
PY

ls -la ../external/models/mobile_net_v2_ssd.tflite



cd ~/Documents/projects/MetaExecuTorch
mkdir -p external/models
cd external/models

# Option 1: Quantized SSD MobileNet V2 (300x300, COCO, INT8) - the standard one
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03.tar.gz
tar -xzf ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03.tar.gz
ls ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03/
# Contains: model.tflite, tflite_graph.pb, pipeline.config, checkpoint files

# Rename the .tflite to match your config's expected path
cp ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03/model.tflite mobile_net_v2_ssd.tflite

# Option 2 (alternative): FP32 SSD MobileNet V2
# wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
# tar -xzf ssd_mobilenet_v2_coco_2018_03_29.tar.gz
# Note: this one ships frozen_inference_graph.pb, not .tflite — you'd need to convert

# Verify
ls -la mobile_net_v2_ssd.tflite
file mobile_net_v2_ssd.tflite


cd ~/Documents/projects/MetaExecuTorch/executorch-toolkit
python -m export.vision.tflite_converter_v2 convert \
  --tflite ../external/models/mobile_net_v2_ssd.tflite \
  --output output/models/mobile_net_v2_ssd/mobile_net_v2_ssd.pt \
  --input-shape 1,300,300,3 \
  --report output/models/mobile_net_v2_ssd/conversion_report.json \
  --verbose


  cd ~/Documents/projects/MetaExecuTorch
mkdir -p external/models
cd external/models

# Download FP32 SSD MobileNet V2 (COCO, 300x300)
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
tar -xzf ssd_mobilenet_v2_coco_2018_03_29.tar.gz
cd ssd_mobilenet_v2_coco_2018_03_29
ls
# frozen_inference_graph.pb, model.ckpt.*, pipeline.config, saved_model/



cd ~/Documents/projects/MetaExecuTorch/executorch-toolkit
source .venv/bin/activate

python - <<'PY'
import tensorflow as tf

saved_model_dir = "../external/models/ssd_mobilenet_v2_coco_2018_03_29/saved_model"
out_path = "../external/models/mobile_net_v2_ssd.tflite"

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS,  # allows ops without native TFLite kernels
]
tflite_model = converter.convert()
with open(out_path, "wb") as f:
    f.write(tflite_model)
print(f"Wrote {out_path}: {len(tflite_model)/1e6:.2f} MB")
PY

ls -la ../external/models/mobile_net_v2_ssd.tflite










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



cd ~/Documents/projects/MetaExecuTorch/executorch-toolkit
source .venv/bin/activate  # or however you activate

# === STEP 0: Place new files ===
# Copy tflite_converter_v2.py from your download into export/vision/
# Create parity_preprocessor.py from above in export/vision/
# Update the JSON config and the orchestrator import as shown
# Place one labeled COCO image at evaluation/mobile_net_v2_ssd/parity_sample.jpg

# === STEP 1: Sanity-check converter syntax ===
python -c "from export.vision.tflite_converter_v2 import convert_tflite_to_pytorch; print('import ok')"
python -c "from export.vision.parity_preprocessor import preprocess_for_tflite; print('import ok')"

# === STEP 2: Standalone parity on the EXISTING broken .pt (proves the gate detects the bug) ===
python -m export.vision.tflite_converter_v2 parity \
  --tflite external/models/mobile_net_v2_ssd.tflite \
  --pytorch output/mobile_net_v2_ssd/mobile_net_v2_ssd.pt \
  --report output/mobile_net_v2_ssd/parity_BEFORE.json \
  --verbose

# Expected: exit code 2, console shows FAIL with diverging tensors

# === STEP 3: Re-convert TFLite → PyTorch with v2 (parity gate runs automatically) ===
python -m export.vision.tflite_converter_v2 convert \
  --tflite external/models/mobile_net_v2_ssd.tflite \
  --output output/mobile_net_v2_ssd/mobile_net_v2_ssd.pt \
  --input-shape 1,300,300,3 \
  --opset 15 \
  --report output/mobile_net_v2_ssd/conversion_report.json \
  --verbose

# Expected: exit code 0, console shows "Status: PASS"
# If exit code 2: parity failed, .pt NOT saved, report tells you which tensor diverged

# === STEP 4: Verify parity passes on the NEW .pt ===
python -m export.vision.tflite_converter_v2 parity \
  --tflite external/models/mobile_net_v2_ssd.tflite \
  --pytorch output/mobile_net_v2_ssd/mobile_net_v2_ssd.pt \
  --report output/mobile_net_v2_ssd/parity_AFTER.json

# Expected: exit code 0, all inputs pass

# === STEP 5: Inspect the conversion report ===
python -c "
import json
r = json.load(open('output/mobile_net_v2_ssd/conversion_report.json'))
print(f'Parity: {r[\"parity_passed\"]}')
print(f'Duration: {r[\"total_duration_s\"]:.1f}s')
print(f'Stages: {len(r[\"stages\"])} ({sum(s[\"success\"] for s in r[\"stages\"])} passed)')
print(f'Unique ops: {len(r[\"operators\"])}')
unmapped = [op for op, rec in r['operators'].items() if rec['status'] == 'unmapped']
patched = [op for op, rec in r['operators'].items() if rec['status'] == 'patched']
print(f'Patched ops: {patched}')
print(f'Unmapped ops: {unmapped}')
"

# === STEP 6: Run the full toolkit (PyTorch → ExecuTorch + quant variants + eval) ===
python export/vision/pytorch_to_executorch_vision.py \
  --config export/configs/vision/mobile_net_v2_ssd.json \
  --generate-report

# This produces:
#  - output/mobile_net_v2_ssd/*.pte (FP32 + each quant variant)
#  - output/mobile_net_v2_ssd/*.etdp (ETDump traces)
#  - output/results/mobile_net_v2_ssd/evaluation_results.json
#  - output/results/mobile_net_v2_ssd/MobileNetV2-SSD-Lite_evaluation.html

# === STEP 7: Sanity-check the eval numbers ===
python -c "
import json
r = json.load(open('output/results/mobile_net_v2_ssd/evaluation_results.json'))
baseline = r.get('pytorch_baseline', r.get('pytorch', {}))
print(f'PyTorch baseline mAP@0.5:       {baseline.get(\"mAP@0.5\", 0):.4f}  (target >= 0.35)')
print(f'PyTorch baseline mAP@[.5:.95]: {baseline.get(\"mAP@[0.5:0.95]\", 0):.4f}  (target >= 0.20)')
print(f'PyTorch baseline F1:            {baseline.get(\"mean_f1\", 0):.4f}  (target >= 0.25)')
"

# === STEP 8: Open the HTML report ===
xdg-open output/results/mobile_net_v2_ssd/MobileNetV2-SSD-Lite_evaluation.html 2>/dev/null || \
  echo "Open in browser: file://$(pwd)/output/results/mobile_net_v2_ssd/MobileNetV2-SSD-Lite_evaluation.html"
