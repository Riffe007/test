import json
from pathlib import Path

MODEL_DIR = Path("output/models/gemma_3_1b_pt_optimum")
RESULTS_DIR = Path("evaluation/gemma_3_1b_pt/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

pte_files = list(MODEL_DIR.glob("*.pte"))

if not pte_files:
    raise FileNotFoundError(f"No .pte files found in {MODEL_DIR}")

artifacts = []

for file in pte_files:
    artifacts.append({
        "name": file.name,
        "path": str(file),
        "size_mb": round(file.stat().st_size / (1024 * 1024), 2),
    })

report = {
    "model": "gemma_3_1b_pt",
    "artifact_dir": str(MODEL_DIR),
    "artifacts": artifacts,
    "status": "export_successful",
}

out_file = RESULTS_DIR / "artifact_report.json"
out_file.write_text(json.dumps(report, indent=2))

print(json.dumps(report, indent=2))
print(f"Wrote: {out_file}")
