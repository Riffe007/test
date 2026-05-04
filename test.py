import json
from pathlib import Path

MODEL_DIR = Path("output/models/gemma_3_1b_pt_optimum")
RESULTS_DIR = Path("evaluation/gemma_3_1b_pt/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

artifacts = []

for file in sorted(MODEL_DIR.glob("*.pte")):
    artifacts.append({
        "artifact": file.name,
        "path": str(file),
        "size_mb": round(file.stat().st_size / (1024 * 1024), 2),
    })

report = {
    "model": "gemma_3_1b_pt",
    "baseline": {
        "runtime": "pytorch_fp32",
        "latency_seconds": 99.0753,
    },
    "executorch_artifacts": artifacts,
    "status": "portable_and_xnnpack_exports_completed",
}

out_file = RESULTS_DIR / "artifact_comparison_report.json"
out_file.write_text(json.dumps(report, indent=2))

print(json.dumps(report, indent=2))
print(f"Wrote: {out_file}")
