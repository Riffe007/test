# ============================================================
# 0. Anchor at project root, activate venv
# ============================================================
cd ~/Documents/projects/MetaExecuTorch && source .venv/bin/activate && pwd

# ============================================================
# 1. Read paths from config (so we delete what config points to,
#    not what I guessed)
# ============================================================
CFG=executorch-toolkit/evaluation/mobilenetv2/config_mobile_net_v2_ssd.json
test -f "$CFG" || { echo "MISSING: $CFG — copy evaluate.py + config first"; exit 1; }

EXPORT_DIR=$(python -c "import json,os; c=json.load(open('$CFG')); p=c['export']['output_dir']; print(os.path.abspath(os.path.join(os.path.dirname('$CFG'),p)))")
RESULTS_DIR=$(python -c "import json,os; c=json.load(open('$CFG')); p=c['evaluation']['output']['results_dir']; print(os.path.abspath(os.path.join(os.path.dirname('$CFG'),p)))")
echo "Export dir:  $EXPORT_DIR"
echo "Results dir: $RESULTS_DIR"

# ============================================================
# 2. Nuke previous artifacts (PTE/PTD, ETRecord, JSON, HTML)
# ============================================================
rm -rf "$EXPORT_DIR" "$RESULTS_DIR"
find . -name "*_etrecord.bin" -delete 2>/dev/null
find . -name "etdump_*.etdp" -delete 2>/dev/null
echo "✓ Previous artifacts cleaned"

# ============================================================
# 3. PyTorch → ExecuTorch export (FP32 + 3 quant variants)
#    Generates: PTE files in $EXPORT_DIR
#               mobile_net_v2_ssd_export_analysis.html
# ============================================================
# >>> EDIT: replace with the toolkit's export entry point you've been using.
#          Based on the open-tab `mobile_net_v2_ssd_export_analysis.html`,
#          this is the command that produced that file.
# Examples of common patterns:
#   python -m executorch_toolkit.pipeline --config "$CFG" --generate-report
#   python executorch-toolkit/run.py --config "$CFG" --generate-report
#   python executorch-toolkit/export.py --config "$CFG" --generate-report
python <TOOLKIT_EXPORT_ENTRY> --config "$CFG" --generate-report

# Verify PTEs landed
ls -lh "$EXPORT_DIR"/*.pte 2>/dev/null || { echo "EXPORT FAILED — no PTEs"; exit 1; }

# ============================================================
# 4. SMOKE EVAL at 25 samples (do this BEFORE the full run)
#    Catches decoder bugs, path issues, import issues in ~30 sec
# ============================================================
cd executorch-toolkit && \
  python evaluation/mobilenetv2/evaluate.py --max-samples 25
cd ..

# Inspect the smoke result
ls -lh "$RESULTS_DIR"/
python -c "
import json
r = json.load(open('$RESULTS_DIR/mobile_net_v2_ssd_evaluation.json'))
print(f\"PyTorch mAP_0.5_0.95: {r['pytorch_baseline'].get('metrics',{}).get('mAP_0.5_0.95','MISSING')}\")
for m in r.get('executorch_models',[]):
    print(f\"{m['name']} mAP_0.5_0.95: {m['metrics'].get('mAP_0.5_0.95','MISSING')}\")
"
# STOP HERE IF: all mAPs are exactly 0.0 (decoder/priors broken) OR JSON missing keys.
# Continue if: any non-zero mAP appears, even if small. Smoke is just sanity.

# ============================================================
# 5. FULL EVAL (5823 images, all 4 ET variants + PyTorch baseline)
#    Wipe smoke results first so samples_evaluated counts cleanly
# ============================================================
rm -rf "$RESULTS_DIR" && mkdir -p "$RESULTS_DIR"
cd executorch-toolkit && \
  python evaluation/mobilenetv2/evaluate.py
cd ..

# ============================================================
# 6. Verify deliverables
# ============================================================
ls -lh "$RESULTS_DIR"/MobileNetV2-SSD-Lite_evaluation_report.html
ls -lh "$RESULTS_DIR"/mobile_net_v2_ssd_evaluation.json
ls -lh "$EXPORT_DIR"/mobile_net_v2_ssd_export_analysis.html 2>/dev/null
echo "✓ DONE"
