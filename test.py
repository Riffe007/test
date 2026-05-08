"""Minimal TFLite (COCO 90) vs qfgaohao mb2-ssd-lite (VOC 21) comparison."""
from pathlib import Path
import sys
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "pytorch" / "pytorch-ssd"))

import tensorflow as tf
from vision.ssd.mobilenet_v2_ssd_lite import (
    create_mobilenetv2_ssd_lite,
    create_mobilenetv2_ssd_lite_predictor,
)

COCO_VAL = ROOT.parent.parent / "dataset" / "samples" / "coco" / "val2017"
img_path = COCO_VAL / "000000039769.jpg"
if not img_path.exists():
    img_path = next(COCO_VAL.glob("*.jpg"))
print(f"Image: {img_path}\n")
img = np.array(Image.open(img_path).convert("RGB").resize((300, 300)), dtype=np.uint8)

print("=== TFLite (FP32, COCO 90) ===")
ip = tf.lite.Interpreter(model_path=str(ROOT / "weights" / "model.tflite"))
ip.allocate_tensors()
in_d = ip.get_input_details()[0]
ip.set_tensor(in_d["index"], ((img.astype(np.float32) - 127.5) / 127.5)[None])
ip.invoke()
raws = [ip.get_tensor(o["index"]) for o in ip.get_output_details()]
boxes = scores = classes = num = None
for arr in raws:
    if arr.ndim == 3 and arr.shape[-1] == 4:
        boxes = arr[0]
    elif arr.ndim == 1 and arr.size == 1:
        num = int(arr[0])
    elif arr.ndim == 2:
        v = arr[0]
        if scores is None and v.dtype.kind == "f" and v.max() <= 1.0:
            scores = v
        else:
            classes = v
n = num if num is not None else len(scores)
print(f"detections returned: {n}")
for i in range(min(n, 10)):
    if scores[i] >= 0.3:
        print(f"  cls={int(classes[i])} score={scores[i]:.3f} box={boxes[i].tolist()}")

print("\n=== qfgaohao mb2-ssd-lite (VOC 21) ===")
net = create_mobilenetv2_ssd_lite(21, is_test=True)
state_dict = torch.load(
    str(ROOT / "weights" / "mb2-ssd-lite.pth"),
    map_location="cpu",
    weights_only=False,
)
net.load_state_dict(state_dict)
pred = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200)
boxes, labels, probs = pred.predict(img, 10, 0.3)
print(f"detections: {len(boxes)}")
for i in range(len(boxes)):
    print(f"  cls={labels[i].item()} score={probs[i].item():.3f} box={boxes[i].tolist()}")
