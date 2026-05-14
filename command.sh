

# 2️⃣  Verify the new files are in place
test -f executorch-toolkit/evaluation/mobilenetv2/evaluate.py && \
  test -f executorch-toolkit/export/configs/vision/config_mobile_net_v2_ssd.json && \
  echo "✓ files in place" || echo "❌ copy evaluate.py and/or config first"

# 3️⃣  Read paths from config (anchored at toolkit root, matching export script convention)
CFG=executorch-toolkit/export/configs/vision/config_mobile_net_v2_ssd.json
TOOLKIT_ROOT=$(realpath executorch-toolkit)
EXPORT_DIR=$(python -c "import json,os; c=json.load(open('$CFG')); print(os.path.realpath(os.path.join('$TOOLKIT_ROOT', c['export']['output_dir'])))")
RESULTS_DIR=$(python -c "import json,os; c=json.load(open('$CFG')); print(os.path.realpath(os.path.join('$TOOLKIT_ROOT', c['evaluation']['output']['results_dir'])))")
echo "Export dir : $EXPORT_DIR"
echo "Results dir: $RESULTS_DIR"

# 4️⃣  Nuke previous artifacts (HTML report will be regenerated in step 8)
rm -rf "$EXPORT_DIR" "$RESULTS_DIR"
find . -name "*_etrecord.bin" -delete 2>/dev/null
find . -name "etdump_*.etdp" -delete 2>/dev/null
echo "✓ Cleaned"

# 5️⃣  Discover the toolkit's export entry point
ls executorch-toolkit/*.py 2>/dev/null
find executorch-toolkit -maxdepth 3 -name "*.py" \( -name "*export*" -o -name "*pipeline*" -o -name "*run*" -o -name "main.py" \) 2>/dev/null
# 👉 The script that previously produced mobile_net_v2_ssd_export_analysis.html

# 6️⃣  EXPORT: PyTorch → ExecuTorch (replace <EXPORT_SCRIPT> with step 5's result)
python <EXPORT_SCRIPT> --config "$CFG" --generate-report

# Verify PTEs landed (expect 4: fp32, 8a8w_pt, 8a8w_pc, 8da4w)
ls -lh "$EXPORT_DIR"/*.pte

# 7️⃣  SMOKE EVAL — 25 samples, ~30 sec
python executorch-toolkit/evaluation/mobilenetv2/evaluate.py --max-samples 25

# Check non-zero mAP signal
python -c "
import json
r = json.load(open('$RESULTS_DIR/mobile_net_v2_ssd_evaluation.json'))
print(f\"PyTorch  {r['pytorch_baseline'].get('metrics',{}).get('mAP_0.5_0.95','MISS')}\")
for m in r.get('executorch_models',[]):
    print(f\"{m['name']:25s} {m['metrics'].get('mAP_0.5_0.95','MISS')}\")
"
# 🛑 STOP if all mAPs are 0.0 (decoder/priors broken)
# ✅ CONTINUE if any non-zero appears (refactor works; absolute mAP is separate diagnostic)

# 8️⃣  FULL EVAL — 5823 images × 5 models. THIS produces the HTML report deliverable.
rm -rf "$RESULTS_DIR" && mkdir -p "$RESULTS_DIR"
python executorch-toolkit/evaluation/mobilenetv2/evaluate.py

# 9️⃣  Confirm the HTML report deliverable exists
ls -lh "$RESULTS_DIR/MobileNetV2-SSD-Lite_evaluation_report.html"
ls -lh "$RESULTS_DIR/mobile_net_v2_ssd_evaluation.json"
echo "✅ DONE"



find ~/Documents/projects/MetaExecuTorch/executorch-toolkit \
  \( -name "generate_*.py" -o -name "report_*.py" -o -name "*orchestrator*.py" \) 2>/dev/null



cd ~/Documents/projects/MetaExecuTorch/executorch-toolkit && \
python -c "from export.vision.converter_config import ModelConfig; import dataclasses; print([f.name for f in dataclasses.fields(ModelConfig)])" && \
python -c "import json; print(json.dumps(json.load(open('export/configs/vision/config_mobile_net_v2_ssd.json'))['model'], indent=2))"
