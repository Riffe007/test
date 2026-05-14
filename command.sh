cd ~/Documents/projects/MetaExecuTorch/executorch-toolkit && source .venv/bin/activate

python - <<'PY'
import sys, numpy as np, cv2, torch
from pathlib import Path

ROOT = Path.home() / "Documents/projects/MetaExecuTorch"
sys.path.insert(0, str(ROOT / "model_sources/MobileNetV2/src/pytorch/pytorch-ssd"))

from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor

WEIGHTS  = ROOT / "model_sources/MobileNetV2/weights/mobile_net_v2_ssd.pth"
LABELS   = ROOT / "model_sources/MobileNetV2/weights/voc-model-labels.txt"
IMG_DIR  = ROOT / "dataset/samples/voc2012/JPEGImages"

class_names = [n.strip() for n in open(LABELS)] if LABELS.exists() else [
    "BACKGROUND","aeroplane","bicycle","bird","boat","bottle","bus","car","cat",
    "chair","cow","diningtable","dog","horse","motorbike","person","pottedplant",
    "sheep","sofa","train","tvmonitor"]
print(f"classes ({len(class_names)}): {class_names[:5]}...")

obj = torch.load(str(WEIGHTS), map_location="cpu", weights_only=False)
print(f"loaded object type: {type(obj).__name__}")

net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True)
sd = obj.state_dict() if isinstance(obj, torch.nn.Module) else obj
print(f"state_dict tensors: {len(sd)}")

missing, unexpected = net.load_state_dict(sd, strict=False)
print(f"  missing keys   : {len(missing)}  (first 3: {missing[:3]})")
print(f"  unexpected keys: {len(unexpected)}  (first 3: {unexpected[:3]})")
net.eval()

samples = sorted(IMG_DIR.glob("*.jpg"))[:1]
print(f"\nsample: {samples[0]}")

img_bgr = cv2.imread(str(samples[0]))
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
print(f"img dtype={img_rgb.dtype} shape={img_rgb.shape}")

predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200)

boxes, labels, probs = predictor.predict(img_rgb, top_k=10, prob_threshold=0.20)
print(f"\nRGB input: {len(boxes)} detections @ p>=0.20")
for b, l, p in zip(boxes[:10], labels[:10], probs[:10]):
    print(f"  {class_names[int(l)]:15s}  p={float(p):.3f}  box={b.tolist()}")

boxes, labels, probs = predictor.predict(img_bgr, top_k=10, prob_threshold=0.20)
print(f"\nBGR input: {len(boxes)} detections @ p>=0.20")
for b, l, p in zip(boxes[:10], labels[:10], probs[:10]):
    print(f"  {class_names[int(l)]:15s}  p={float(p):.3f}  box={b.tolist()}")
PY
