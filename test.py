import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path("/home/timothy_riffe/Documents/projects/llm")
MODEL_DIR = PROJECT_ROOT / "models/gemma-3-1b-pt"
CONFIG = PROJECT_ROOT / "export/configs/llm/config_gemma_3_1b_pt_optimum.json"
OUTPUT_DIR = PROJECT_ROOT / "output/models/gemma_3_1b_pt_optimum"
RESULTS_DIR = PROJECT_ROOT / "evaluation/gemma_3_1b_pt/results"
REPORTS_DIR = PROJECT_ROOT / "reports"


COMMANDS = [
    {
        "name": "Validate JSON config",
        "cmd": [
            sys.executable,
            "-m",
            "json.tool",
            str(CONFIG),
        ],
        "required": True,
    },
    {
        "name": "Test local Gemma load",
        "cmd": [
            sys.executable,
            "export/llm/test_gemma_load.py",
        ],
        "required": True,
    },
    {
        "name": "Run PyTorch baseline evaluation",
        "cmd": [
            sys.executable,
            "evaluation/gemma_3_1b_pt/evaluate_pytorch.py",
        ],
        "required": True,
    },
    {
        "name": "Export FP32 XNNPACK ExecuTorch artifact",
        "cmd": [
            sys.executable,
            "export/llm/pytorch_to_executorch_llm.py",
        ],
        "required": True,
    },
    {
        "name": "Export INT8 XNNPACK ExecuTorch artifact",
        "cmd": [
            sys.executable,
            "export/llm/pytorch_to_executorch_llm_int8.py",
        ],
        "required": True,
    },
    {
        "name": "Check quantization capability",
        "cmd": [
            sys.executable,
            "evaluation/gemma_3_1b_pt/quantization_capability_check.py",
        ],
        "required": True,
    },
    {
        "name": "Check artifacts",
        "cmd": [
            sys.executable,
            "evaluation/gemma_3_1b_pt/check_artifacts.py",
        ],
        "required": True,
    },
    {
        "name": "Compare artifacts",
        "cmd": [
            sys.executable,
            "evaluation/gemma_3_1b_pt/compare_artifacts.py",
        ],
        "required": True,
    },
    {
        "name": "Generate final markdown report",
        "cmd": [
            sys.executable,
            "reports/generate_gemma_report.py",
        ],
        "required": True,
    },
]


EXPECTED_ARTIFACTS = [
    OUTPUT_DIR / "gemma_3_1b_pt_portable_fp32.pte",
    OUTPUT_DIR / "gemma_3_1b_pt_xnnpack_fp32.pte",
    OUTPUT_DIR / "gemma_3_1b_pt_xnnpack_int8_weight_only.pte",
]


def run_command(name: str, cmd: list[str], required: bool = True) -> bool:
    print("\n" + "=" * 80)
    print(f"Running: {name}")
    print("=" * 80)
    print(" ".join(cmd))

    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        text=True,
    )

    if result.returncode != 0:
        print(f"FAILED: {name}")

        if required:
            raise RuntimeError(f"Required step failed: {name}")

        return False

    print(f"PASSED: {name}")
    return True


def validate_paths() -> None:
    print("\nValidating required paths...")

    required_paths = [
        PROJECT_ROOT,
        MODEL_DIR,
        CONFIG,
        PROJECT_ROOT / "export/llm/test_gemma_load.py",
        PROJECT_ROOT / "export/llm/pytorch_to_executorch_llm.py",
        PROJECT_ROOT / "export/llm/pytorch_to_executorch_llm_int8.py",
        PROJECT_ROOT / "evaluation/gemma_3_1b_pt/evaluate_pytorch.py",
        PROJECT_ROOT / "evaluation/gemma_3_1b_pt/check_artifacts.py",
        PROJECT_ROOT / "evaluation/gemma_3_1b_pt/compare_artifacts.py",
        PROJECT_ROOT / "reports/generate_gemma_report.py",
    ]

    missing = [path for path in required_paths if not path.exists()]

    if missing:
        print("Missing required paths:")
        for path in missing:
            print(f"- {path}")
        raise FileNotFoundError("One or more required files are missing.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Required paths validated.")


def summarize_artifacts() -> None:
    print("\n" + "=" * 80)
    print("Artifact Summary")
    print("=" * 80)

    for artifact in EXPECTED_ARTIFACTS:
        if artifact.exists():
            size_mb = artifact.stat().st_size / (1024 * 1024)
            print(f"FOUND: {artifact.name} — {size_mb:.2f} MB")
        else:
            print(f"MISSING: {artifact.name}")

    report = REPORTS_DIR / "gemma_3_1b_pt_final_benchmark_summary.md"

    if report.exists():
        print(f"\nFinal report generated: {report}")
    else:
        print("\nFinal report missing.")


def main() -> None:
    print("Gemma 3-1B-PT ExecuTorch Automation Pipeline")
    print("=" * 80)

    validate_paths()

    for step in COMMANDS:
        run_command(
            name=step["name"],
            cmd=step["cmd"],
            required=step["required"],
        )

    summarize_artifacts()

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
