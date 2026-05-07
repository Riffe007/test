"""Pipeline orchestrator: TFLite -> PyTorch with formatted stage output.

Stages:
  1. TFLite -> Clean FP32 ONNX
  2. Clean FP32 ONNX -> PyTorch
"""

from pathlib import Path
import argparse
import subprocess
import sys
import time

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent  # MobileNetV2/
WEIGHTS = ROOT / "weights"


# --- ANSI colors --------------------------------------------------------
class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    OK = "\033[32m"
    FAIL = "\033[31m"
    INFO = "\033[36m"
    DIM = "\033[2m"


# --- Stage definitions --------------------------------------------------
STAGES = [
    {
        "num": 1,
        "title": "TFLite -> Clean FP32 ONNX",
        "description": (
            "Convert TFLite, force FP32 inputs, fold weight DQs into "
            "FP32 initializers, strip activation Q/DQ, simplify, "
            "parity-check vs source TFLite."
        ),
        "script": "tflite_to_clean_fp32_onnx.py",
        "input": WEIGHTS / "model.tflite",
        "output": WEIGHTS / "model.fp32.onnx",
        "runtime_est": "~10-15s",
    },
    {
        "num": 2,
        "title": "ONNX -> PyTorch",
        "description": "Convert clean FP32 ONNX to PyTorch via onnx2torch.",
        "script": "convert_onnx_to_pytorch.py",
        "input": WEIGHTS / "model.fp32.onnx",
        "output": WEIGHTS / "model.pt",
        "runtime_est": "~5-10s",
    },
]


# --- Output helpers -----------------------------------------------------
def banner(text="", char="="):
    print(char * 80)
    if text:
        print(text)
        print(char * 80)


def stage_header(stage, total):
    print()
    banner()
    print(f"[{stage['num']}/{total}] {C.BOLD}{stage['title']}{C.RESET}")
    print(f"{C.DIM}{stage['description']}{C.RESET}")
    banner()


def info_line(label, value):
    print(f"  {C.INFO}[info]{C.RESET} {label:<10} {value}")


def ok_line(elapsed, summary=""):
    msg = f"  {C.OK}[ ok ]{C.RESET} stage done in {elapsed:.1f}s"
    if summary:
        msg += f" -- {summary}"
    print(msg)


def fail_line(elapsed, exit_code, debug_cmd=""):
    print(f"  {C.FAIL}[fail]{C.RESET} stage failed with exit code "
          f"{exit_code} after {elapsed:.1f}s")
    if debug_cmd:
        print(f"         to debug, re-run with verbose logging:")
        print(f"           {debug_cmd}")


# --- Stage runner -------------------------------------------------------
def run_stage(stage, total):
    stage_header(stage, total)
    info_line("input:", stage["input"])
    info_line("output:", stage["output"])
    info_line("runtime:", stage["runtime_est"])
    info_line("command:", f"python3 {stage['script']}")
    print("-" * 80)

    start = time.time()
    try:
        subprocess.run(
            [sys.executable, str(SCRIPTS / stage["script"])],
            check=True,
            cwd=str(SCRIPTS),
        )
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start
        print("-" * 80)
        fail_line(
            elapsed, e.returncode,
            debug_cmd=f"python3 {SCRIPTS / stage['script']}",
        )
        return elapsed, "FAIL", ""

    elapsed = time.time() - start
    print("-" * 80)

    # Summarize output file
    summary = ""
    if stage["output"].exists():
        size_kb = stage["output"].stat().st_size / 1024
        if size_kb > 1024:
            summary = f"{stage['output'].name} ({size_kb / 1024:.1f} MB)"
        else:
            summary = f"{stage['output'].name} ({size_kb:.1f} KB)"

    ok_line(elapsed, summary)
    return elapsed, "ok", summary


# --- Summary table ------------------------------------------------------
def print_summary(results, total):
    print()
    banner("Summary")
    header = f"{'stage':<40} {'status':<7} {'time':<8} {'output'}"
    print(header)
    print("-" * len(header))

    total_time = 0.0
    for stage, (elapsed, status, output) in zip(STAGES, results):
        title = f"[{stage['num']}/{total}] {stage['title']}"
        color = C.OK if status == "ok" else (C.FAIL if status == "FAIL" else C.DIM)
        time_str = f"{elapsed:.1f}s" if elapsed > 0 else "-"
        print(f"{title:<40} {color}{status:<7}{C.RESET} {time_str:<8} {output}")
        total_time += elapsed

    print()
    print(f"total wall-clock: {total_time:.1f}s")


# --- Main ---------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--continue-on-fail",
        action="store_true",
        help="Run remaining stages even if one fails (default: stop)",
    )
    args = parser.parse_args()

    total = len(STAGES)
    results = []
    failed_stage = None

    for stage in STAGES:
        elapsed, status, output = run_stage(stage, total)
        results.append((elapsed, status, output))

        if status == "FAIL" and not args.continue_on_fail:
            failed_stage = stage
            # Pad results so summary table aligns
            for _ in STAGES[len(results):]:
                results.append((0.0, "-", ""))
            break

    print_summary(results, total)

    if failed_stage:
        print()
        print(f"{C.FAIL}[fail]{C.RESET} pipeline aborted at stage "
              f"{failed_stage['num']} ({failed_stage['title']})")
        sys.exit(1)

    print()
    print(f"{C.OK}[ ok ]{C.RESET} pipeline complete -- "
          f"final output: {STAGES[-1]['output']}")
    sys.exit(0)


if __name__ == "__main__":
    main()
