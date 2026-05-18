cd ~/Documents/projects/MetaExecuTorch/executorch-toolkit

python -m pip install dominate

python - <<'PY'
import json
from pathlib import Path

cfg_path = Path("export/configs/vision/config_mobile_net_v2_ssd.json")
cfg = json.loads(cfg_path.read_text())

root = Path.home() / "Documents/projects/MetaExecuTorch"

# Find actual TFLite file
tflites = list((root / "model_sources/MobileNetV2").rglob("*.tflite"))
if not tflites:
    raise FileNotFoundError("No .tflite file found under model_sources/MobileNetV2")
tflite_path = str(tflites[0])

# Find actual PyTorch SSD repo root: parent directory that contains vision/
vision_dirs = list((root / "model_sources/MobileNetV2").rglob("vision"))
vision_dirs = [p for p in vision_dirs if p.is_dir() and (p / "ssd").exists()]
if not vision_dirs:
    raise FileNotFoundError("Could not find vision/ssd package under model_sources/MobileNetV2")
repo_root = str(vision_dirs[0].parent)

cfg["model"]["source_path"] = repo_root
cfg["model"]["model_sources_repo_path"] = repo_root

cfg["tflite_parity"]["tflite_source_path"] = tflite_path
cfg["evaluation"]["backends"]["tflite_baseline"]["model_path"] = tflite_path

cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")

print("Fixed config:")
print("  TFLite:", tflite_path)
print("  Vision repo root:", repo_root)
PY

python ./evaluation/mobilenetv2/evaluate.py \
  --config export/configs/vision/config_mobile_net_v2_ssd.jsoncd ~/Documents/projects/MetaExecuTorch/executorch-toolkit

python -m pip install dominate

python - <<'PY'
import json
from pathlib import Path

cfg_path = Path("export/configs/vision/config_mobile_net_v2_ssd.json")
cfg = json.loads(cfg_path.read_text())

root = Path.home() / "Documents/projects/MetaExecuTorch"

# Find actual TFLite file
tflites = list((root / "model_sources/MobileNetV2").rglob("*.tflite"))
if not tflites:
    raise FileNotFoundError("No .tflite file found under model_sources/MobileNetV2")
tflite_path = str(tflites[0])

# Find actual PyTorch SSD repo root: parent directory that contains vision/
vision_dirs = list((root / "model_sources/MobileNetV2").rglob("vision"))
vision_dirs = [p for p in vision_dirs if p.is_dir() and (p / "ssd").exists()]
if not vision_dirs:
    raise FileNotFoundError("Could not find vision/ssd package under model_sources/MobileNetV2")
repo_root = str(vision_dirs[0].parent)

cfg["model"]["source_path"] = repo_root
cfg["model"]["model_sources_repo_path"] = repo_root

cfg["tflite_parity"]["tflite_source_path"] = tflite_path
cfg["evaluation"]["backends"]["tflite_baseline"]["model_path"] = tflite_path

cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")

print("Fixed config:")
print("  TFLite:", tflite_path)
print("  Vision repo root:", repo_root)
PY

python ./evaluation/mobilenetv2/evaluate.py \
  --config export/configs/vision/config_mobile_net_v2_ssd.json
