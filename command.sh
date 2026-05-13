cd ~/Documents/projects/MetaExecuTorch/executorch-toolkit && source .venv/bin/activate && mkdir -p ../output/eval_mobilenet_v2_ssd && PYTHONPATH=../model_sources/MobileNetV2/src/pytorch/pytorch-ssd python evaluation/mobilenetv2/evaluate.py --voc-root ~/.cache/kagglehub/datasets/watanabe2362/voctrainval-11may2012/versions/1/VOCdevkit/VOC2012 --gt-coco-json dataset/voc2012_val_coco.json --tflite-model ../model_sources/MobileNetV2/weights/model.tflite --pytorch-model ../model_sources/MobileNetV2/weights/mobile_net_v2_ssd.pth --model-dir tests/integration/outputs/mobile_net_v2_ssd/basemodel_workflow/models --results-dir ../output/eval_mobilenet_v2_ssd --generate-report 2>&1 | tee ../output/eval_run.log



find ~/Documents/projects/MetaExecuTorch -type f -name "*.json" \( -iname "*voc*" -o -iname "*coco*" \) 2>/dev/null

cd ~/Documents/projects/MetaExecuTorch/executorch-toolkit && source .venv/bin/activate && mkdir -p ../output/eval_mobilenet_v2_ssd && PYTHONPATH=../model_sources/MobileNetV2/src/pytorch/pytorch-ssd python evaluation/mobilenetv2/evaluate.py --voc-root ~/.cache/kagglehub/datasets/watanabe2362/voctrainval-11may2012/versions/1/VOCdevkit/VOC2012 --gt-coco-json ../dataset/voc2012_as_coco/instances_voc2012_val.json --tflite-model ../model_sources/MobileNetV2/weights/model.tflite --pytorch-model ../model_sources/MobileNetV2/weights/mobile_net_v2_ssd.pth --model-dir tests/integration/outputs/mobile_net_v2_ssd/basemodel_workflow/models --results-dir ../output/eval_mobilenet_v2_ssd --generate-report --max-samples 25 2>&1 | tee ../output/eval_smoke.log


grep -nE "^def |^class " ~/Documents/projects/MetaExecuTorch/executorch-toolkit/evaluation/mobilenetv2/generate_report.py && echo "---LOG---" && grep -iE "html|report|generat|error|traceback" ~/Documents/projects/MetaExecuTorch/output/eval_smoke.log | tail -30


python -c "
import json
d = json.load(open('/home/timothy_riffe/Documents/projects/MetaExecuTorch/dataset/voc2012_as_coco/instances_voc2012_val.json'))
print('Categories (id -> name):')
for c in sorted(d['categories'], key=lambda x: x['id']):
    print(f'  {c[\"id\"]}: {c[\"name\"]}')
print('Sample anns:')
for a in d['annotations'][:3]:
    print(f'  image_id={a[\"image_id\"]} cat={a[\"category_id\"]} bbox={a[\"bbox\"]}')
print('Sample image:', {k: d['images'][0][k] for k in ('id','file_name','width','height')})
"
cd ~/Documents/projects/MetaExecuTorch/executorch-toolkit && source .venv/bin/activate && PYTHONPATH=../model_sources/MobileNetV2/src/pytorch/pytorch-ssd python evaluation/mobilenetv2/evaluate.py --voc-root ~/.cache/kagglehub/datasets/watanabe2362/voctrainval-11may2012/versions/1/VOCdevkit/VOC2012 --gt-coco-json ../dataset/voc2012_as_coco/instances_voc2012_val.json --tflite-model ../model_sources/MobileNetV2/weights/model.tflite --pytorch-model ../model_sources/MobileNetV2/weights/mobile_net_v2_ssd.pth --model-dir tests/integration/outputs/mobile_net_v2_ssd/basemodel_workflow/models --results-dir ../output/eval_mobilenet_v2_ssd --generate-report --max-samples 25 2>&1 | tee ../output/eval_smoke2.log





python3 << 'PATCH_EOF'
import re, shutil, sys, py_compile
from pathlib import Path

EVAL = Path.home() / "Documents/projects/MetaExecuTorch/executorch-toolkit/evaluation/mobilenetv2/evaluate.py"
BACKUP = EVAL.with_suffix(".py.prepatch.bak")
if not EVAL.exists():
    sys.exit(f"ERROR: {EVAL} not found")
shutil.copy(EVAL, BACKUP)
src = EVAL.read_text()
notes = []

# 1. Category-mapping constants
if "QFGAOHAO_TO_COCO_CATEGORY" not in src:
    block = """

# Translate qfgaohao class index (1..20) to COCO 80-class category_id.
# voc_to_coco.py emits GT in COCO space (person=1, ..., tv=72) not 1..20.
QFGAOHAO_TO_COCO_CATEGORY: Dict[int, int] = {
    1: 5, 2: 2, 3: 16, 4: 9, 5: 44,
    6: 6, 7: 3, 8: 17, 9: 62, 10: 21,
    11: 67, 12: 18, 13: 19, 14: 4, 15: 1,
    16: 64, 17: 20, 18: 63, 19: 7, 20: 72,
}
VOC_COCO_CATEGORY_IDS: Tuple[int, ...] = tuple(
    sorted(QFGAOHAO_TO_COCO_CATEGORY.values())
)
"""
    src, n = re.subn(r"(NUM_VOC_CLASSES\s*=\s*len\(VOC_CLASS_NAMES\)[^\n]*\n)",
                     r"\1" + block, src, count=1)
    notes.append("✓ category mapping added" if n else "✗ NUM_VOC_CLASSES anchor missing")
else:
    notes.append("○ category mapping already present")

# 2. Detection.to_coco
old = '"category_id": int(self.category_id),'
new = '"category_id": QFGAOHAO_TO_COCO_CATEGORY[int(self.category_id)],'
if old in src:
    src = src.replace(old, new, 1); notes.append("✓ Detection.to_coco translates")
elif new in src:
    notes.append("○ Detection.to_coco already patched")
else:
    notes.append("✗ Detection.to_coco pattern not found")

# 3. voc_restricted subset
old = "class_subset=list(range(1, NUM_VOC_CLASSES + 1)),"
new = "class_subset=list(VOC_COCO_CATEGORY_IDS),"
if old in src:
    src = src.replace(old, new, 1); notes.append("✓ voc_restricted uses COCO IDs")
elif new in src:
    notes.append("○ voc_restricted already patched")
else:
    notes.append("✗ voc_restricted pattern not found")

# 4. per-class AP int cast
old = "coco_gt.loadCats([cat_id])"
new = "coco_gt.loadCats([int(cat_id)])"
if old in src:
    src = src.replace(old, new); notes.append("✓ cat_id cast to int")
else:
    notes.append("○ cat_id cast already applied")

# 5. parse_tflite_outputs disambiguation rewrite
old = '''    raw = [interp.get_tensor(d["index"]) for d in output_details]

    boxes_arr = classes_arr = scores_arr = None
    count_val: Optional[int] = None

    for arr in raw:
        if arr.ndim == 3 and arr.shape[-1] == 4:
            boxes_arr = arr
        elif arr.ndim == 1 and arr.size == 1:
            count_val = int(arr[0])
        elif arr.ndim == 2:
            # Either classes or scores; distinguish by dtype + value range.
            flat = arr.reshape(-1).astype(np.float32)
            if flat.size == 0:
                continue
            if (flat == flat.astype(np.int32)).all() and float(flat.max()) <= NUM_VOC_CLASSES:
                classes_arr = arr
            else:
                scores_arr = arr

    if boxes_arr is None or classes_arr is None or scores_arr is None:
        raise RuntimeError(
            "Could not identify TFLite output tensors "
            f"(shapes={[a.shape for a in raw]})"
        )'''
new = '''    raw = [interp.get_tensor(d["index"]) for d in output_details]

    boxes_arr: Optional[np.ndarray] = None
    count_val: Optional[int] = None
    two_d_arrays: List[Tuple[int, np.ndarray]] = []

    for idx, arr in enumerate(raw):
        if arr.ndim == 3 and arr.shape[-1] == 4:
            boxes_arr = arr
        elif arr.ndim == 1 and arr.size <= 4:
            count_val = int(arr.flat[0])
        elif arr.ndim == 2:
            two_d_arrays.append((idx, arr))

    if boxes_arr is None or len(two_d_arrays) != 2:
        raise RuntimeError(
            "Could not identify TFLite output tensors "
            f"(shapes={[a.shape for a in raw]})"
        )

    (i_a, a), (i_b, b) = two_d_arrays
    a_max = float(np.max(a)) if a.size else 0.0
    b_max = float(np.max(b)) if b.size else 0.0
    if a_max > 1.0 + 1e-6:
        classes_arr, scores_arr = a, b
    elif b_max > 1.0 + 1e-6:
        classes_arr, scores_arr = b, a
    elif i_a < i_b:
        classes_arr, scores_arr = a, b
    else:
        classes_arr, scores_arr = b, a'''
if old in src:
    src = src.replace(old, new, 1); notes.append("✓ TFLite disambiguation fixed")
elif "two_d_arrays" in src:
    notes.append("○ TFLite parser already patched")
else:
    notes.append("✗ TFLite parser block not found exactly")

# 6. HTML report -> render_html
old = '''    if args.generate_report:
        _log_section("GENERATING HTML REPORT")
        try:
            from evaluation.mobilenetv2.generate_report import (  # noqa: WPS433
                generate_html_for_json,
            )

            report_path = generate_html_for_json(
                output_file, output_dir=results_dir
            )
            if report_path:
                logger.info("HTML report: %s", report_path)
            else:
                logger.warning("HTML report generation returned no path")
        except Exception as exc:
            logger.error("HTML report generation failed: %s", exc, exc_info=True)'''
new = '''    if args.generate_report:
        _log_section("GENERATING HTML REPORT")
        try:
            from evaluation.mobilenetv2.generate_report import (  # noqa: WPS433
                render_html,
            )

            with open(output_file, "r", encoding="utf-8") as f:
                saved_results = json.load(f)

            html_str = render_html(output_file, saved_results)
            model_name = saved_results.get("model_name", "model")
            report_path = results_dir / f"{model_name}_evaluation_report.html"
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(html_str)
            logger.info("HTML report: %s", report_path)
        except Exception as exc:
            logger.error("HTML report generation failed: %s", exc, exc_info=True)'''
if old in src:
    src = src.replace(old, new, 1); notes.append("✓ HTML uses render_html")
elif "render_html" in src:
    notes.append("○ HTML already patched")
else:
    notes.append("✗ HTML block not found exactly")

EVAL.write_text(src)
try:
    py_compile.compile(str(EVAL), doraise=True)
    print("✓ Syntax OK")
except py_compile.PyCompileError as e:
    print(f"✗ SYNTAX ERROR — restoring backup: {e}")
    shutil.copy(BACKUP, EVAL); sys.exit(1)

print(f"Backup: {BACKUP}")
for note in notes:
    print(f"  {note}")
PATCH_EOF




cd ~/Documents/projects/MetaExecuTorch && {
  echo "========== SOURCE MODELS =========="
  ls -lh model_sources/MobileNetV2/weights/ 2>/dev/null || echo "  (missing)"

  echo; echo "========== GROUND TRUTH =========="
  ls -lh dataset/voc2012_as_coco/ 2>/dev/null || echo "  (missing)"

  echo; echo "========== EXECUTORCH .PTE EXPORTS =========="
  ls -lh executorch-toolkit/tests/integration/outputs/mobile_net_v2_ssd/basemodel_workflow/models/ 2>/dev/null || echo "  (no .pte triplet yet — toolkit export not run)"

  echo; echo "========== TOOLKIT RUN REPORTS / LOGS =========="
  find executorch-toolkit/tests/integration/outputs/mobile_net_v2_ssd/ -maxdepth 4 -type f \( -name "*.html" -o -name "*.json" -o -name "*.log" -o -name "*.md" \) 2>/dev/null

  echo; echo "========== ETDUMP TRACES =========="
  find executorch-toolkit -name "*.etdump" -o -name "etdump*.bin" 2>/dev/null | head -10
  find output -name "*.etdump" 2>/dev/null | head -10

  echo; echo "========== EVAL OUTPUTS (existing) =========="
  ls -lh output/eval_mobilenet_v2_ssd/ 2>/dev/null || echo "  (no eval outputs yet)"

  echo; echo "========== PHASE 1 (INCEPTION V3) FOR REFERENCE =========="
  find output -path "*inception_v3*" -type f 2>/dev/null | head -15

  echo; echo "========== VOC IMAGES =========="
  VOC=~/.cache/kagglehub/datasets/watanabe2362/voctrainval-11may2012/versions/1/VOCdevkit/VOC2012/JPEGImages
  if [ -d "$VOC" ]; then echo "  $VOC: $(ls "$VOC" | wc -l) files"; else echo "  (missing)"; fi

  echo; echo "========== EVALUATOR CODE =========="
  ls -lh executorch-toolkit/evaluation/mobilenetv2/ 2>/dev/null

  echo; echo "========== TOOLKIT CONFIG =========="
  ls -lh executorch-toolkit/tests/integration/mobile_net_v2_ssd/ 2>/dev/null
}




