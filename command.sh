cd ~/Documents/projects/MetaExecuTorch/executorch-toolkit

python - <<'PY'
import json
from pathlib import Path

cfg_path = Path("export/configs/vision/config_mobile_net_v2_ssd.json")
cfg = json.loads(cfg_path.read_text())

cfg["evaluation"]["backends"]["executorch_models"] = [
    {
        "name": "executorch_fp32",
        "pte_path": str(
            Path.cwd() / "output/models/mobile_net_v2_ssd/mobile_net_v2_ssd_executorch.pte"
        ),
        "is_baseline": True
    },
    {
        "name": "executorch_int8_8a8w_pt",
        "pte_path": str(
            Path.cwd() / "output/models/mobile_net_v2_ssd/mobile_net_v2_ssd_executorch_8a8w_pt.pte"
        )
    },
    {
        "name": "executorch_int8_8a8w_pc",
        "pte_path": str(
            Path.cwd() / "output/models/mobile_net_v2_ssd/mobile_net_v2_ssd_executorch_8a8w_pc.pte"
        )
    },
    {
        "name": "executorch_int4_8da4w",
        "pte_path": str(
            Path.cwd() / "output/models/mobile_net_v2_ssd/mobile_net_v2_ssd_executorch_8da4w.pte"
        )
    }
]

cfg_path.write_text(json.dumps(cfg, indent=2))
print("Fixed .pte absolute paths.")
PY

export PYTHONPATH=$PYTHONPATH:$(pwd)

python ./evaluation/mobilenetv2/evaluate.py \
  --config export/configs/vision/config_mobile_net_v2_ssd.json
