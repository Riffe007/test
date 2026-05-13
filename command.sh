cd ~/Documents/projects/MetaExecuTorch/executorch-toolkit && source .venv/bin/activate && python3 << 'DIAG_EOF'
import json
from pathlib import Path

ROOT = Path.home() / "Documents/projects/MetaExecuTorch"
GT = ROOT / "dataset/voc2012_as_coco/instances_voc2012_val.json"
RES = ROOT / "output/results/mobilenetv2/evaluation_results.json"

gt = json.load(open(GT))
print("=== GT ===")
print(f"  images: {len(gt['images'])}  annotations: {len(gt['annotations'])}  categories: {len(gt['categories'])}")
print(f"  sample image:  {gt['images'][0]}")
print(f"  sample annot:  {gt['annotations'][0]}")
print(f"  categories: {[(c['id'], c['name']) for c in gt['categories']]}")

res = json.load(open(RES))
# Walk to find pytorch detections (structure varies)
def find_dets(obj, path=""):
    if isinstance(obj, dict):
        if "detections_coco" in obj and isinstance(obj["detections_coco"], list):
            yield path, obj["detections_coco"]
        for k, v in obj.items():
            yield from find_dets(v, f"{path}.{k}" if path else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from find_dets(v, f"{path}[{i}]")

for path, dets in find_dets(res):
    if not dets: continue
    print(f"\n=== DETECTIONS @ {path}  (n={len(dets)}) ===")
    print(f"  sample: {dets[0]}")
    pred_imgs = {d['image_id'] for d in dets}
    pred_cats = {d['category_id'] for d in dets}
    gt_imgs   = {im['id'] for im in gt['images']}
    gt_cats   = {c['id'] for c in gt['categories']}
    gt_imgs_with_annot = {a['image_id'] for a in gt['annotations']}
    print(f"  pred image_ids: {len(pred_imgs)} unique, range {min(pred_imgs)}..{max(pred_imgs)}")
    print(f"  GT   image_ids: {len(gt_imgs)} unique, range {min(gt_imgs)}..{max(gt_imgs)}")
    print(f"  intersection of pred image_ids ∩ GT image_ids: {len(pred_imgs & gt_imgs)}")
    print(f"  intersection of pred image_ids ∩ GT-with-annotations: {len(pred_imgs & gt_imgs_with_annot)}")
    print(f"  pred category_ids: {sorted(pred_cats)}")
    print(f"  GT   category_ids: {sorted(gt_cats)}")
    print(f"  bbox xywh first 3: {[d['bbox'] for d in dets[:3]]}")
    print(f"  bbox xywh ranges:  x∈[{min(d['bbox'][0] for d in dets):.1f}, {max(d['bbox'][0] for d in dets):.1f}]  y∈[{min(d['bbox'][1] for d in dets):.1f}, {max(d['bbox'][1] for d in dets):.1f}]  w∈[{min(d['bbox'][2] for d in dets):.1f}, {max(d['bbox'][2] for d in dets):.1f}]  h∈[{min(d['bbox'][3] for d in dets):.1f}, {max(d['bbox'][3] for d in dets):.1f}]")
    print(f"  score range: [{min(d['score'] for d in dets):.4f}, {max(d['score'] for d in dets):.4f}]")
    break  # just look at the first detection set
DIAG_EOF
