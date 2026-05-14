
# ─────────────────────────────────────────────────────────────
# 2️⃣  Read config paths (so cleanup targets what config says,
#     not what I'm guessing)
# ─────────────────────────────────────────────────────────────
CFG=executorch-toolkit/evaluation/mobilenetv2/config_mobile_net_v2_ssd.json
test -f "$CFG" || { echo "❌ MISSING $CFG — copy evaluate.py + config first"; }

EXPORT_DIR=$(python -c "import json,os; c=json.load(open('$CFG')); print(os.path.abspath(os.path.join(os.path.dirname('$CFG'), c['export']['output_dir'])))")
RESULTS_DIR=$(python -c "import json,os; c=json.load(open('$CFG')); print(os.path.abspath(os.path.join(os.path.dirname('$CFG'), c['evaluation']['output']['results_dir'])))")
echo "Export dir : $EXPORT_DIR"
echo "Results dir: $RESULTS_DIR"

# ─────────────────────────────────────────────────────────────
# 3️⃣  Nuke previous artifacts (PTE/PTD, ETRecord, JSON, HTML)
# ─────────────────────────────────────────────────────────────
rm -rf "$EXPORT_DIR" "$RESULTS_DIR"
find . -name "*_etrecord.bin" -delete 2>/dev/null
find . -name "etdump_*.etdp" -delete 2>/dev/null
echo "✓ Cleaned"

# ─────────────────────────────────────────────────────────────
# 4️⃣  Discover the toolkit's export entry point
#     (likely export.py / pipeline.py / run.py at toolkit root)
# ─────────────────────────────────────────────────────────────
ls executorch-toolkit/*.py 2>/dev/null
find executorch-toolkit -maxdepth 2 -name "*.py" \( -name "*export*" -o -name "*pipeline*" -o -name "*run*" -o -name "main.py" \) 2>/dev/null
# 👉 Pick the one that takes --config and generates mobile_net_v2_ssd_export_analysis.html
#    (the toolkit command you've used before to produce that HTML tab in VS Code)

# ─────────────────────────────────────────────────────────────
# 5️⃣  EXPORT: PyTorch → ExecuTorch (FP32 + 3 quant variants)
#     ⚠️ Replace EXPORT_SCRIPT with what step 4 surfaces
# ─────────────────────────────────────────────────────────────
EXPORT_SCRIPT=executorch-toolkit/export.py   # ← EDIT IF STEP 4 SHOWED A DIFFERENT NAME
python "$EXPORT_SCRIPT" --config "$CFG" --generate-report

# Verify PTEs landed (4 expected: fp32, 8a8w_pt, 8a8w_pc, 8da4w)
ls -lh "$EXPORT_DIR"/*.pte || { echo "❌ EXPORT FAILED — no PTEs"; }
ls -lh "$EXPORT_DIR"/*_export_analysis.html 2>/dev/null

# ─────────────────────────────────────────────────────────────
# 6️⃣  SMOKE EVAL (25 samples ≈ 30 sec; STOP if every mAP=0.0)
# ─────────────────────────────────────────────────────────────
python executorch-toolkit/evaluation/mobilenetv2/evaluate.py --max-samples 25

python -c "
import json
r = json.load(open('$RESULTS_DIR/mobile_net_v2_ssd_evaluation.json'))
pt = r.get('pytorch_baseline',{}).get('metrics',{})
print(f\"PyTorch mAP_0.5_0.95: {pt.get('mAP_0.5_0.95','MISSING')}\")
for m in r.get('executorch_models',[]):
    print(f\"{m['name']:25s} mAP_0.5_0.95: {m['metrics'].get('mAP_0.5_0.95','MISSING')}\")
"
# 🛑 STOP if all numbers are exactly 0.0 (decoder broken — debug at smoke scale)
# ✅ CONTINUE if any non-zero (even 0.003 means pipeline is mechanically correct)

# ─────────────────────────────────────────────────────────────
# 7️⃣  FULL EVAL (all 5823 VOC val images, all 4 ET variants + PT baseline)
#     Wipe smoke first so samples_evaluated counts cleanly
# ─────────────────────────────────────────────────────────────
rm -rf "$RESULTS_DIR" && mkdir -p "$RESULTS_DIR"
python executorch-toolkit/evaluation/mobilenetv2/evaluate.py

# ─────────────────────────────────────────────────────────────
# 8️⃣  Confirm deliverables (these are what you submit)
# ─────────────────────────────────────────────────────────────
ls -lh "$RESULTS_DIR/MobileNetV2-SSD-Lite_evaluation_report.html"
ls -lh "$RESULTS_DIR/mobile_net_v2_ssd_evaluation.json"
ls -lh "$EXPORT_DIR"/*_export_analysis.html 2>/dev/null
echo ""
echo "✅ DONE — open the HTML report in a browser"
