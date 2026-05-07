mkdir -p scripts reports

cat > scripts/compare.py <<'PYSCRIPT'
"""Compare FP32 TFLite (SSD MobileNetV2, COCO 90) vs qfgaohao MobileNetV2 SSD-Lite (VOC 21).

Class-AGNOSTIC IoU on box geometry. Picks an image from the existing
dataset/samples/coco/val2017/ (canonical 000000039769.jpg if present).
"""
from pathlib import Path
import sys
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
TFLITE_PATH = ROOT / "weights" / "model.tflite"
PT_WEIGHTS  = ROOT / "weights" / "mb2-ssd-lite.pth"
COCO_LABELS = ROOT / "labels" / "coco_labels.txt"
VOC_LABELS  = ROOT / "labels" / "voc_labels.txt"
QFGAOHAO    = ROOT / "src" / "pytorch" / "pytorch-ssd"
REPORT      = ROOT / "reports" / "comparison.md"
VIZ         = ROOT / "reports" / "comparison.png"

COCO_VAL = ROOT.parent.parent / "dataset" / "samples" / "coco" / "val2017"
CANONICAL = COCO_VAL / "000000039769.jpg"
_jpgs = sorted(COCO_VAL.glob("*.jpg"))
TEST_IMAGE = CANONICAL if CANONICAL.exists() else (_jpgs[0] if _jpgs else None)

INPUT_SIZE = 300
SCORE_THRESHOLD = 0.30
IOU_MATCH = 0.50
TOP_K = 10


def load_labels(path):
    out = {}
    for i, line in enumerate(path.read_text().splitlines()):
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        try:
            out[int(parts[0])] = parts[1] if len(parts) > 1 else f"class_{int(parts[0])}"
        except ValueError:
            out[i] = line
    return out


def name(labels, cid):
    return labels.get(int(cid), f"id={int(cid)}")


def load_image(path, size):
    img = Image.open(path).convert("RGB").resize((size, size))
    return np.array(img, dtype=np.uint8)


def run_tflite(path, image_uint8):
    import tensorflow as tf
    interp = tf.lite.Interpreter(model_path=str(path))
    interp.allocate_tensors()
    in_d = interp.get_input_details()[0]
    out_ds = interp.get_output_details()
    x = ((image_uint8.astype(np.float32) - 127.5) / 127.5)[None]
    interp.set_tensor(in_d["index"], x)
    interp.invoke()
    raws = [interp.get_tensor(o["index"]) for o in out_ds]
    boxes = scores = classes = num = None
    for arr in raws:
        if arr.ndim == 3 and arr.shape[-1] == 4:
            boxes = arr[0]
        elif arr.ndim == 1 and arr.size == 1:
            num = int(arr[0])
        elif arr.ndim == 2:
            v = arr[0]
            if scores is None and v.dtype.kind == "f" and v.size > 0 and 0 <= v.min() and v.max() <= 1.0:
                scores = v
            else:
                classes = v
    n = num if num is not None else (len(scores) if scores is not None else 0)
    return {"boxes": boxes[:n].astype(np.float32),
            "scores": scores[:n].astype(np.float32),
            "classes": classes[:n].astype(int)}


def run_pytorch(weights_path, image_uint8):
    sys.path.insert(0, str(QFGAOHAO))
    from vision.ssd.mobilenet_v2_ssd_lite import (
        create_mobilenetv2_ssd_lite,
        create_mobilenetv2_ssd_lite_predictor,
    )
    net = create_mobilenetv2_ssd_lite(21, is_test=True)
    net.load(str(weights_path))
    predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200)
    boxes_px, labels, probs = predictor.predict(image_uint8, TOP_K, SCORE_THRESHOLD)
    boxes_px = boxes_px.cpu().numpy()
    labels = labels.cpu().numpy()
    probs = probs.cpu().numpy()
    H, W = image_uint8.shape[:2]
    boxes_norm = np.column_stack([
        boxes_px[:, 1] / H, boxes_px[:, 0] / W,
        boxes_px[:, 3] / H, boxes_px[:, 2] / W,
    ]).astype(np.float32)
    return {"boxes": boxes_norm, "scores": probs, "classes": labels.astype(int)}


def box_iou(a, b):
    ya1, xa1, ya2, xa2 = a
    yb1, xb1, yb2, xb2 = b
    iw = max(0.0, min(xa2, xb2) - max(xa1, xb1))
    ih = max(0.0, min(ya2, yb2) - max(ya1, yb1))
    inter = iw * ih
    ua = max(0.0, ya2 - ya1) * max(0.0, xa2 - xa1)
    ub = max(0.0, yb2 - yb1) * max(0.0, xb2 - xb1)
    union = ua + ub - inter
    return inter / union if union > 0 else 0.0


def match_class_agnostic(a, b):
    matches, used = [], set()
    for i in range(len(a["boxes"])):
        if a["scores"][i] < SCORE_THRESHOLD:
            continue
        best_j, best_iou = -1, 0.0
        for j in range(len(b["boxes"])):
            if j in used or b["scores"][j] < SCORE_THRESHOLD:
                continue
            iou = box_iou(a["boxes"][i], b["boxes"][j])
            if iou > best_iou:
                best_j, best_iou = j, iou
        if best_iou >= IOU_MATCH:
            matches.append((i, best_j, best_iou))
            used.add(best_j)
    return matches


def visualize(image, tfl, pt, coco, voc, out):
    H, W = image.shape[:2]
    a, b = Image.fromarray(image).copy(), Image.fromarray(image).copy()
    da, db = ImageDraw.Draw(a), ImageDraw.Draw(b)
    for i, box in enumerate(tfl["boxes"]):
        if tfl["scores"][i] < SCORE_THRESHOLD: continue
        ymin, xmin, ymax, xmax = box
        da.rectangle([xmin*W, ymin*H, xmax*W, ymax*H], outline="red", width=2)
        da.text((xmin*W+2, ymin*H+2),
                f"{name(coco, tfl['classes'][i])} {tfl['scores'][i]:.2f}", fill="red")
    for j, box in enumerate(pt["boxes"]):
        if pt["scores"][j] < SCORE_THRESHOLD: continue
        ymin, xmin, ymax, xmax = box
        db.rectangle([xmin*W, ymin*H, xmax*W, ymax*H], outline="lime", width=2)
        db.text((xmin*W+2, ymin*H+2),
                f"{name(voc, pt['classes'][j])} {pt['scores'][j]:.2f}", fill="darkgreen")
    canvas = Image.new("RGB", (W*2+10, H+20), "white")
    canvas.paste(a, (0, 20)); canvas.paste(b, (W+10, 20))
    d = ImageDraw.Draw(canvas)
    d.text((5, 2), "TFLite (red) - COCO 90", fill="red")
    d.text((W+15, 2), "qfgaohao PT (green) - VOC 21", fill="darkgreen")
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)


def write_report(tfl, pt, matches, coco, voc, image_path, out):
    n_t = sum(1 for s in tfl["scores"] if s >= SCORE_THRESHOLD)
    n_p = sum(1 for s in pt["scores"]  if s >= SCORE_THRESHOLD)
    agree = (len(matches) / max(n_t, n_p)) if max(n_t, n_p) else 0.0
    L = [
        "# TFLite vs qfgaohao PyTorch - detection comparison\n",
        f"- Test image: `{image_path}`",
        f"- TFLite: SSD MobileNetV2 FP32, COCO 90 classes, 300x300",
        f"- PyTorch: qfgaohao mb2-ssd-lite, VOC 21 classes, 300x300",
        f"- Score threshold: {SCORE_THRESHOLD}",
        f"- IoU match threshold: {IOU_MATCH} (class-AGNOSTIC; class spaces differ)",
        "",
        "## Summary",
        f"- TFLite detections (>={SCORE_THRESHOLD}): **{n_t}**",
        f"- PyTorch detections (>={SCORE_THRESHOLD}): **{n_p}**",
        f"- Geometric box matches (IoU >={IOU_MATCH}): **{len(matches)}**",
        f"- Geometric agreement: **{agree:.1%}**",
        "",
    ]
    def table(title, dets, lbls):
        out = [f"## {title}", "",
               "| # | class | score | box (ymin, xmin, ymax, xmax) |",
               "|---|-------|-------|------------------------------|"]
        for k, s in enumerate(dets["scores"]):
            if s < SCORE_THRESHOLD: continue
            b = dets["boxes"][k]
            out.append(f"| {k} | {name(lbls, dets['classes'][k])} | {s:.3f} | "
                       f"({b[0]:.3f}, {b[1]:.3f}, {b[2]:.3f}, {b[3]:.3f}) |")
        out.append("")
        return out
    L += table("TFLite (COCO)", tfl, coco)
    L += table("PyTorch (VOC)", pt,  voc)
    L.append("## Geometric matches (greedy IoU, class-agnostic)")
    if matches:
        L.append("| TFLite # | TFLite class | PyTorch # | PyTorch class | IoU |")
        L.append("|----------|--------------|-----------|---------------|-----|")
        for ia, ib, iou in matches:
            L.append(f"| {ia} | {name(coco, tfl['classes'][ia])} | "
                     f"{ib} | {name(voc, pt['classes'][ib])} | {iou:.3f} |")
    else:
        L.append("_No geometric matches above IoU threshold._")
    L.append("")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L))


def main():
    if TEST_IMAGE is None:
        sys.exit(f"ERROR: no .jpg in {COCO_VAL}")
    print(f"[1/5] Loading test image: {TEST_IMAGE}")
    img = load_image(TEST_IMAGE, INPUT_SIZE)
    coco = load_labels(COCO_LABELS)
    voc = load_labels(VOC_LABELS)
    print(f"      COCO labels: {len(coco)}   VOC labels: {len(voc)}")
    print("[2/5] Running TFLite (FP32, COCO 90) ...")
    tfl = run_tflite(TFLITE_PATH, img)
    print(f"      detections >= {SCORE_THRESHOLD}: "
          f"{sum(1 for s in tfl['scores'] if s >= SCORE_THRESHOLD)}")
    print("[3/5] Running qfgaohao mb2-ssd-lite (VOC 21) ...")
    pt = run_pytorch(PT_WEIGHTS, img)
    print(f"      detections >= {SCORE_THRESHOLD}: "
          f"{sum(1 for s in pt['scores'] if s >= SCORE_THRESHOLD)}")
    print("[4/5] Matching boxes class-agnostically ...")
    matches = match_class_agnostic(tfl, pt)
    print(f"      geometric matches (IoU >= {IOU_MATCH}): {len(matches)}")
    print("[5/5] Writing report + visualization ...")
    visualize(img, tfl, pt, coco, voc, VIZ)
    write_report(tfl, pt, matches, coco, voc, TEST_IMAGE, REPORT)
    print(f"\n  Report: {REPORT}")
    print(f"  Viz:    {VIZ}")


if __name__ == "__main__":
    main()
PYSCRIPT

python3 scripts/compare.py
