cd ~/Documents/projects/MetaExecuTorch/executorch-toolkit && source .venv/bin/activate && python3 << 'PATCH_EOF'
import shutil, py_compile
from pathlib import Path
EVAL = Path.home() / "Documents/projects/MetaExecuTorch/executorch-toolkit/evaluation/mobilenetv2/evaluate.py"
shutil.copy(EVAL, EVAL.with_suffix(".py.r3.bak"))
src = EVAL.read_text()
notes = []
old = "QFGAOHAO_TO_COCO_CATEGORY[int(self.category_id)]"
new = "QFGAOHAO_TO_COCO_CATEGORY.get(int(self.category_id), int(self.category_id))"
if old in src: src = src.replace(old, new); notes.append("✓ to_coco fallback")
else: notes.append("○ to_coco already patched")
for label, var in [("PyTorch", "pt_metrics"), ("TFLite", "tfl_metrics")]:
    pat = (f'ca = {var}.get("class_agnostic", {{}})\n                logger.info(\n                    "{label} mAP@0.5=%.4f  mAP@0.5:0.95=%.4f",\n                    ca.get("mAP_50", float("nan")),\n                    ca.get("mAP_50_95", float("nan")),\n                )')
    rep = (f'ca = {var}.get("class_agnostic", {{}})\n                vr = {var}.get("voc_restricted", {{}})\n                logger.info(\n                    "{label} mAP@0.5=%.4f (all)  %.4f (voc)  mAP@0.5:0.95=%.4f",\n                    ca.get("mAP_50", float("nan")),\n                    vr.get("mAP_50", float("nan")),\n                    ca.get("mAP_50_95", float("nan")),\n                )')
    if pat in src: src = src.replace(pat, rep); notes.append(f"✓ {label} log")
    else: notes.append(f"○ {label} log")
old_et = '''ca = et_metrics.get("class_agnostic", {})
                    logger.info(
                        "%s  mAP@0.5=%.4f  mAP@0.5:0.95=%.4f  mean_latency=%.2f ms",
                        pte_path.name,
                        ca.get("mAP_50", float("nan")),
                        ca.get("mAP_50_95", float("nan")),
                        entry["latency"]["mean_ms"],
                    )'''
new_et = '''ca = et_metrics.get("class_agnostic", {})
                    vr = et_metrics.get("voc_restricted", {})
                    logger.info(
                        "%s  mAP@0.5=%.4f (all)  %.4f (voc)  mAP@0.5:0.95=%.4f  mean_latency=%.2f ms",
                        pte_path.name,
                        ca.get("mAP_50", float("nan")),
                        vr.get("mAP_50", float("nan")),
                        ca.get("mAP_50_95", float("nan")),
                        entry["latency"]["mean_ms"],
                    )'''
if old_et in src: src = src.replace(old_et, new_et); notes.append("✓ ExecuTorch log")
else: notes.append("○ ExecuTorch log")
# Align HTML output filename with Phase 1 convention (InceptionV3_evaluation_results.html)
old_html = '_evaluation_report.html"'
new_html = '_evaluation_results.html"'
if old_html in src: src = src.replace(old_html, new_html); notes.append("✓ HTML named _evaluation_results.html")
else: notes.append("○ HTML name already aligned")
EVAL.write_text(src)
py_compile.compile(str(EVAL), doraise=True)
print("✓ Syntax OK")
for n in notes: print(f"  {n}")
PATCH_EOF
echo "---"
mkdir -p ../output/results/mobilenetv2 && \
PYTHONPATH=../model_sources/MobileNetV2/src/pytorch/pytorch-ssd python evaluation/mobilenetv2/evaluate.py \
  --voc-root ~/.cache/kagglehub/datasets/watanabe2362/voctrainval-11may2012/versions/1/VOCdevkit/VOC2012 \
  --gt-coco-json ../dataset/voc2012_as_coco/instances_voc2012_val.json \
  --tflite-model ../model_sources/MobileNetV2/weights/model.tflite \
  --pytorch-model ../model_sources/MobileNetV2/weights/mobile_net_v2_ssd.pth \
  --model-dir tests/integration/outputs/mobile_net_v2_ssd/basemodel_workflow/models \
  --results-dir ../output/results/mobilenetv2 \
  --generate-report \
  --max-samples 25 2>&1 | tee ../output/results/mobilenetv2/eval_smoke.log
