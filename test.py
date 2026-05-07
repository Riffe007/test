"""Pipeline orchestrator: TFLite -> PyTorch with formatted stage output.

Stages:
  1. TFLite -> Clean FP32 ONNX
  2. Clean FP32 ONNX -> PyTorch
"""

from pathlib import Path
import argparse
import subprocess
import shlex
import sys
import time

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
WEIGHTS = ROOT / "weights"

TFLITE_PATH = WEIGHTS / "model.tflite"
ONNX_PATH = WEIGHTS / "model.fp32.onnx"
PT_PATH = WEIGHTS / "model.pt"


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
        "input": TFLITE_PATH,
        "output": ONNX_PATH,
        "args": [
            "--input", str(TFLITE_PATH),
            "--output", str(ONNX_PATH),
        ],
        "runtime_est": "~10-15s",
    },
    {
        "num": 2,
        "title": "ONNX -> PyTorch",
        "description": "Convert clean FP32 ONNX to PyTorch via onnx2torch.",
        "script": "convert_onnx_to_pytorch.py",
        "input": ONNX_PATH,
        "output": PT_PATH,
        "args": [
            "--input", str(ONNX_PATH),
            "--output", str(PT_PATH),
        ],
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
def run_stage(stage, total, extra_args=None):
    extra_args = extra_args or []
    full_args = stage["args"] + extra_args

    stage_header(stage, total)
    info_line("input:", stage["input"])
    info_line("output:", stage["output"])
    info_line("runtime:", stage["runtime_est"])

    pretty_cmd = "python3 " + shlex.join([stage["script"]] + full_args)
    info_line("command:", pretty_cmd)
    print("-" * 80)

    cmd = [sys.executable, str(SCRIPTS / stage["script"])] + full_args
    debug_cmd = "python3 " + shlex.join(
        [str(SCRIPTS / stage["script"])] + full_args + ["--verbose"]
    )

    start = time.time()
    try:
        subprocess.run(cmd, check=True, cwd=str(SCRIPTS))
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start
        print("-" * 80)
        fail_line(elapsed, e.returncode, debug_cmd=debug_cmd)
        return elapsed, "FAIL", ""

    elapsed = time.time() - start
    print("-" * 80)

    summary = ""
    if stage["output"].exists():
        size_kb = stage["output"].stat().st_size / 1024
        summary = (f"{stage['output'].name} ({size_kb / 1024:.1f} MB)"
                   if size_kb > 1024
                   else f"{stage['output'].name} ({size_kb:.1f} KB)")

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
        color = (C.OK if status == "ok"
                 else C.FAIL if status == "FAIL"
                 else C.DIM)
        time_str = f"{elapsed:.1f}s" if elapsed > 0 else "-"
        print(f"{title:<40} {color}{status:<7}{C.RESET} "
              f"{time_str:<8} {output}")
        total_time += elapsed

    print()
    print(f"total wall-clock: {total_time:.1f}s")


# --- Main ---------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--continue-on-fail", action="store_true",
        help="Run remaining stages even if one fails (default: stop)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Pass --verbose through to each stage script",
    )
    parser.add_argument(
        "--skip-smoke-test", action="store_true",
        help="Pass --skip-smoke-test through to stage 1",
    )
    args = parser.parse_args()

    total = len(STAGES)
    results = []
    failed_stage = None

    for stage in STAGES:
        extra = []
        if args.verbose:
            extra.append("--verbose")
        if args.skip_smoke_test and stage["num"] == 1:
            extra.append("--skip-smoke-test")

        elapsed, status, output = run_stage(stage, total, extra_args=extra)
        results.append((elapsed, status, output))

        if status == "FAIL" and not args.continue_on_fail:
            failed_stage = stage
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
