#!/usr/bin/env python3
"""
MetaExecuTorch model conversion pipeline.

Stages:
  [1/2] TFLite -> Clean FP32 ONNX
  [2/2] ONNX   -> PyTorch

Drop in next to tflite_to_clean_fp32_onnx.py and onnx_to_pytorch.py.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


# ---- pretty printing ---------------------------------------------------------

class C:
    INFO = "\033[36m"
    OK = "\033[32m"
    FAIL = "\033[31m"
    DIM = "\033[2m"
    RESET = "\033[0m"


def banner(title: str, width: int = 96) -> None:
    print("=" * width)
    print(title)
    print("=" * width)


def info(label: str, value: str) -> None:
    print(f"  {C.INFO}[info]{C.RESET} {label:<10} {value}")


def fail(msg: str) -> None:
    print(f"  {C.FAIL}[fail]{C.RESET} {msg}")


# ---- stage definition --------------------------------------------------------

@dataclass
class Stage:
    description: str           # e.g. "TFLite -> Clean FP32 ONNX"
    script: Path               # absolute path to the stage script
    input_path: Path
    output_path: Path
    extra_args: list[str] = field(default_factory=list)
    runtime_hint: str = ""

    status: str = "PENDING"
    elapsed: float = 0.0

    def build_cmd(self, verbose: bool) -> list[str]:
        cmd = [
            sys.executable, str(self.script),
            "--input",  str(self.input_path),
            "--output", str(self.output_path),
            *self.extra_args,
        ]
        if verbose:
            cmd.append("--verbose")
        return cmd


# ---- runner ------------------------------------------------------------------

def run_stage(stage: Stage, idx: int, total: int, verbose: bool) -> bool:
    banner(f"[{idx}/{total}] {stage.description}")

    cmd = stage.build_cmd(verbose)

    info("input:",   str(stage.input_path))
    info("output:",  str(stage.output_path))
    if stage.runtime_hint:
        info("runtime:", stage.runtime_hint)
    info("command:", " ".join(cmd))
    print("-" * 96)

    t0 = time.perf_counter()
    result = subprocess.run(cmd)
    stage.elapsed = time.perf_counter() - t0

    if result.returncode == 0:
        stage.status = "OK"
        return True

    stage.status = "FAIL"
    fail(f"stage failed with exit code {result.returncode} after {stage.elapsed:.1f}s")
    print("        to debug, re-run with verbose logging:")
    debug_cmd = stage.build_cmd(verbose=True)
    print(f"          {' '.join(debug_cmd)}")
    return False


def print_summary(stages: list[Stage], total_elapsed: float) -> None:
    print()
    banner("Summary")
    print(f"  {'stage':<40} {'status':<8} {'time':<8} output")
    print("  " + "-" * 90)
    for s in stages:
        color = C.OK if s.status == "OK" else (C.FAIL if s.status == "FAIL" else C.DIM)
        t_str = f"{s.elapsed:.1f}s" if s.elapsed else "-"
        o_str = str(s.output_path) if s.status == "OK" else "-"
        print(f"  {s.description:<40} {color}{s.status:<8}{C.RESET} {t_str:<8} {o_str}")
    print()
    print(f"  total wall-clock: {total_elapsed:.1f}s")


# ---- entry -------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description="MetaExecuTorch model conversion pipeline (TFLite -> ONNX -> PyTorch)"
    )
    p.add_argument("--input", required=True, type=Path,
                   help="Source .tflite model")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Directory for intermediate + final outputs (default: alongside input)")
    p.add_argument("--input-shape", nargs=4, type=int, metavar=("N", "C", "H", "W"),
                   default=None, help="Override input shape, forwarded to stage 1")
    p.add_argument("--skip-smoke-test", action="store_true",
                   help="Forwarded to stages that support it")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    tflite_path = args.input.resolve()
    if not tflite_path.is_file():
        print(f"[fail] input not found: {tflite_path}", file=sys.stderr)
        return 2

    out_dir = (args.output_dir or tflite_path.parent).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    fp32_onnx = out_dir / f"{tflite_path.stem}.fp32.onnx"
    pt_path   = out_dir / f"{tflite_path.stem}.pt"

    scripts_dir = Path(__file__).resolve().parent

    # Args forwarded to whichever stage accepts them
    shape_args: list[str] = (
        ["--input-shape", *map(str, args.input_shape)] if args.input_shape else []
    )
    smoke_args: list[str] = ["--skip-smoke-test"] if args.skip_smoke_test else []

    stages = [
        Stage(
            description="TFLite -> Clean FP32 ONNX",
            script=scripts_dir / "tflite_to_clean_fp32_onnx.py",
            input_path=tflite_path,
            output_path=fp32_onnx,
            extra_args=[*shape_args, *smoke_args],
            runtime_hint="~10-15s",
        ),
        Stage(
            description="ONNX -> PyTorch",
            script=scripts_dir / "onnx_to_pytorch.py",
            input_path=fp32_onnx,
            output_path=pt_path,
            extra_args=[*smoke_args],
            runtime_hint="~5-10s",
        ),
    ]

    t_start = time.perf_counter()
    aborted_at: int | None = None

    for idx, stage in enumerate(stages, start=1):
        ok = run_stage(stage, idx, len(stages), args.verbose)
        print()
        if not ok:
            aborted_at = idx
            break

    total_elapsed = time.perf_counter() - t_start
    print_summary(stages, total_elapsed)

    if aborted_at is not None:
        print()
        fail(f"pipeline aborted at stage {aborted_at} "
             f"({stages[aborted_at - 1].description})")
        return 1

    print()
    print(f"  {C.OK}[ok]{C.RESET} pipeline complete -> {stages[-1].output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
