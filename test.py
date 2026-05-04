import json
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path.cwd()

MODEL_NAME = "Gemma 3-1B-PT"
MODEL_KEY = "gemma_3_1b_pt"

MODEL_DIR = PROJECT_ROOT / "output/models/gemma_3_1b_pt_optimum"
RESULTS_DIR = PROJECT_ROOT / "evaluation/gemma_3_1b_pt/results"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

PYTORCH_RESULTS = RESULTS_DIR / "pytorch_baseline_results.json"
OUTPUT_REPORT = REPORTS_DIR / "gemma_3_1b_pt_final_benchmark_summary.md"


def size_mb(path: Path) -> float:
    return round(path.stat().st_size / (1024 * 1024), 2)


def pct_reduction(old: float, new: float) -> float:
    if old == 0:
        return 0.0
    return round(((old - new) / old) * 100, 2)


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def collect_artifacts() -> list[dict]:
    artifacts = []

    for file in sorted(MODEL_DIR.glob("*.pte")):
        name = file.name

        if "int8" in name:
            backend = "XNNPACK"
            precision = "INT8 weight-only"
        elif "xnnpack" in name:
            backend = "XNNPACK"
            precision = "FP32"
        elif "portable" in name:
            backend = "Portable CPU"
            precision = "FP32"
        else:
            backend = "Unknown"
            precision = "Unknown"

        artifacts.append({
            "name": name,
            "path": str(file),
            "backend": backend,
            "precision": precision,
            "size_mb": size_mb(file),
        })

    return artifacts


def find_artifact(artifacts: list[dict], contains: str) -> dict | None:
    for artifact in artifacts:
        if contains in artifact["name"]:
            return artifact
    return None


def generate_markdown() -> str:
    artifacts = collect_artifacts()
    pytorch_results = load_json(PYTORCH_RESULTS)

    latency = pytorch_results.get("latency_seconds")
    latency_display = f"{latency:.2f} seconds" if isinstance(latency, (int, float)) else "Not recorded"

    portable = find_artifact(artifacts, "portable")
    xnnpack_fp32 = next(
        (a for a in artifacts if "xnnpack" in a["name"] and "int8" not in a["name"]),
        None,
    )
    int8 = find_artifact(artifacts, "int8")

    lines = []

    lines.append(f"# {MODEL_NAME} ExecuTorch Final Benchmark Summary")
    lines.append("")
    lines.append(f"Generated: `{datetime.now().isoformat(timespec='seconds')}`")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Status")
    lines.append("")
    lines.append(f"{MODEL_NAME} has been successfully exported to ExecuTorch deployment artifacts.")
    lines.append("")

    lines.append("## Artifact Summary")
    lines.append("")
    lines.append("| Artifact | Backend | Precision | Size |")
    lines.append("|---|---|---:|---:|")

    for artifact in artifacts:
        lines.append(
            f"| `{artifact['name']}` | {artifact['backend']} | {artifact['precision']} | {artifact['size_mb']} MB |"
        )

    lines.append("")
    lines.append("## PyTorch Baseline")
    lines.append("")
    lines.append("| Runtime | Precision | Latency |")
    lines.append("|---|---:|---:|")
    lines.append(f"| PyTorch | FP32 | {latency_display} |")
    lines.append("")

    prompt = pytorch_results.get("prompt", "Not recorded")
    lines.append("Baseline prompt:")
    lines.append("")
    lines.append("```text")
    lines.append(prompt)
    lines.append("```")
    lines.append("")

    if int8 and xnnpack_fp32:
        reduction = pct_reduction(xnnpack_fp32["size_mb"], int8["size_mb"])
        lines.append("## XNNPACK FP32 to INT8 Size Reduction")
        lines.append("")
        lines.append("```text")
        lines.append(f"{xnnpack_fp32['size_mb']} MB -> {int8['size_mb']} MB")
        lines.append("```")
        lines.append("")
        lines.append(f"Size reduction: **{reduction}%**")
        lines.append("")

    if int8 and portable:
        reduction = pct_reduction(portable["size_mb"], int8["size_mb"])
        lines.append("## Portable FP32 to INT8 Size Reduction")
        lines.append("")
        lines.append("```text")
        lines.append(f"{portable['size_mb']} MB -> {int8['size_mb']} MB")
        lines.append("```")
        lines.append("")
        lines.append(f"Size reduction: **{reduction}%**")
        lines.append("")

    lines.append("## Key Results")
    lines.append("")
    lines.append("### 1. Portable FP32 Export")
    lines.append("")
    lines.append("The model was exported through `torch.export`, lowered into an ExecuTorch Edge program, and emitted as a portable CPU `.pte` artifact.")
    lines.append("")
    lines.append("### 2. XNNPACK FP32 Export")
    lines.append("")
    lines.append("XNNPACK export completed successfully using:")
    lines.append("")
    lines.append("```python")
    lines.append("partitioner=[XnnpackPartitioner()]")
    lines.append("```")
    lines.append("")
    lines.append("### 3. INT8 Weight-Only Export")
    lines.append("")
    lines.append("TorchAO INT8 weight-only quantization successfully produced an XNNPACK-backed `.pte` artifact.")
    lines.append("")

    recommended = int8 or xnnpack_fp32 or portable

    lines.append("## Recommended Deployment Candidate")
    lines.append("")
    if recommended:
        lines.append("```text")
        lines.append(recommended["path"])
        lines.append("```")
    else:
        lines.append("No deployment artifact found.")
    lines.append("")

    lines.append("## Completed Deliverables")
    lines.append("")
    completed = [
        "Local Gemma model validation",
        "PyTorch FP32 baseline",
        "Portable ExecuTorch FP32 export",
        "XNNPACK FP32 export",
        "TorchAO INT8 weight-only quantization",
        "XNNPACK INT8 `.pte` export",
        "Artifact size comparison",
        "Automated report generation",
    ]
    for item in completed:
        lines.append(f"- {item}")
    lines.append("")

    lines.append("## Known Issues Resolved")
    lines.append("")
    issues = [
        "Malformed JSON config fixed with `python -m json.tool` validation.",
        "Local model path corrected by adding leading `/`.",
        "`DynamicCache` export issue fixed with `model.config.use_cache = False`.",
        "XNNPACK API mismatch fixed by using `partitioner=[XnnpackPartitioner()]`.",
        "Prompt repetition improved with sampling, repetition penalty, and generated-token-only decoding.",
        "TorchAO INT8 API mismatch fixed by using `Int8WeightOnlyConfig()`.",
    ]
    for issue in issues:
        lines.append(f"- {issue}")
    lines.append("")

    lines.append("## Remaining Work")
    lines.append("")
    remaining = [
        "Runtime execution benchmarking for Portable FP32 `.pte`",
        "Runtime execution benchmarking for XNNPACK FP32 `.pte`",
        "Runtime execution benchmarking for XNNPACK INT8 `.pte`",
        "ETDump trace generation",
        "Inspector performance analysis",
        "INT4 weight-only quantization attempt",
        "Final latency comparison table",
    ]
    for item in remaining:
        lines.append(f"- {item}")
    lines.append("")

    lines.append("## Conclusion")
    lines.append("")
    lines.append("The project successfully moved from local Hugging Face model validation to multiple ExecuTorch deployment artifacts, including an optimized INT8 XNNPACK artifact with substantial size reduction.")
    lines.append("")
    lines.append("The next phase should focus on runtime benchmarking, profiling, and INT4 optimization.")
    lines.append("")

    return "\n".join(lines)


def main():
    report = generate_markdown()
    OUTPUT_REPORT.write_text(report)
    print(f"Wrote report: {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
