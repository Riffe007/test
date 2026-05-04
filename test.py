import json
import importlib
from pathlib import Path

RESULTS_DIR = Path("evaluation/gemma_3_1b_pt/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

modules_to_check = [
    "torchao",
    "torchao.quantization",
    "executorch",
    "executorch.exir",
    "executorch.backends.xnnpack.partition.xnnpack_partitioner",
]

results = {}

for module_name in modules_to_check:
    try:
        module = importlib.import_module(module_name)
        results[module_name] = {
            "available": True,
            "path": getattr(module, "__file__", "built-in"),
        }
        print(f"FOUND: {module_name}")
    except Exception as e:
        results[module_name] = {
            "available": False,
            "error": str(e),
        }
        print(f"MISSING: {module_name} -> {e}")

out_file = RESULTS_DIR / "quantization_capability_check.json"
out_file.write_text(json.dumps(results, indent=2))

print(f"Wrote: {out_file}")
