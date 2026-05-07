#!/usr/bin/env python3
"""run_pipeline.py

Orchestrate the full MobileNetV2 conversion pipeline with pre-flight
checks, per-stage timing, exit-code-aware diagnostics, and a summary
table. Designed for interactive use during development; convert.sh
remains the lighter-weight entry point for CI / scripting.

Pipeline stages:

    [1/3] TFLite (INT8)     -> ONNX (with Q/DQ ops)
    [2/3] ONNX (with Q/DQ)  -> ONNX (FP32, Q/DQ-free)
    [3/3] ONNX (FP32)       -> PyTorch .pt

What this script does that convert.sh does not:

    * Pre-flight: Python version, required packages, input file, and
      the three stage scripts -- all checked before launching anything.
      Stops with a clear remediation message if any check fails.
    * Per-stage banner with input/output paths and expected runtime hint.
    * Live-streamed subprocess output (you see warnings as they happen).
    * Per-stage exit-code lookup -- each script's documented non-zero
      codes are mapped to one-line failure-mode descriptions, so a
      'tf2onnx not installed' is reported as such rather than just
      'exited with 2'.
    * Final summary table: stage status, wall-clock time, output sizes.

Run from anywhere:

    python3 path/to/scripts/run_pipeline.py

Or from the scripts/ directory:

    ./run_pipeline.py

Exit codes:

    0  every stage succeeded
    N  the exit code of the first stage that failed (1-9 from the
       respective stage scripts; see their own docstrings)
   10  pre-flight check failed (no stage was executed)
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #

# Directory layout: this file lives in scripts/, weights live one up.
_SCRIPT_DIR = Path(__file__).resolve().parent
_WEIGHTS_DIR = (_SCRIPT_DIR / ".." / "weights").resolve()

# Required Python interpreter floor.
_MIN_PYTHON = (3, 9)

# Packages that must be importable in the active interpreter for the
# whole pipeline to succeed. Listed with the stage that needs them so
# the failure message can be precise.
_REQUIRED_PACKAGES: list[tuple[str, str]] = [
    ("tf2onnx",     "stage 1 (TFLite -> ONNX)"),
    ("tensorflow",  "stage 1 (transitive dep of tf2onnx)"),
    ("onnx",        "stages 1, 2, 3"),
    ("numpy",       "stage 2 (Q/DQ folding math)"),
    ("onnx2torch",  "stage 3 (ONNX -> PyTorch)"),
    ("torch",       "stage 3 (PyTorch model save / smoke test)"),
]


# --------------------------------------------------------------------------- #
# Stage definitions                                                           #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Stage:
    number: int
    name: str
    script: Path
    input_path: Path
    output_path: Path
    description: str
    runtime_hint: str
    exit_code_meanings: dict[int, str] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"[{self.number}/3] {self.name}"


def _build_stages() -> list[Stage]:
    return [
        Stage(
            number=1,
            name="TFLite -> ONNX",
            script=_SCRIPT_DIR / "convert_tflite_to_onnx.py",
            input_path=_WEIGHTS_DIR / "model.tflite",
            output_path=_WEIGHTS_DIR / "model.onnx",
            description="Convert the source TFLite (INT8) to ONNX via tf2onnx. "
                        "The output ONNX still contains QuantizeLinear/DequantizeLinear ops.",
            runtime_hint="~30-60s on first run (TF imports), ~10s thereafter",
            exit_code_meanings={
                1: "filesystem error (input missing or output dir unwritable)",
                2: "tf2onnx is not installed in this Python interpreter",
                3: "tf2onnx subprocess failed (returned non-zero or timed out)",
                4: "tf2onnx returned 0 but produced an empty / unparseable .onnx",
            },
        ),
        Stage(
            number=2,
            name="ONNX -> ONNX (Q/DQ-free)",
            script=_SCRIPT_DIR / "dequantize_onnx.py",
            input_path=_WEIGHTS_DIR / "model.onnx",
            output_path=_WEIGHTS_DIR / "model.fp32.onnx",
            description="Fold Q/DQ ops into FP32 constants so onnx2torch can consume the graph.",
            runtime_hint="~2-5s",
            exit_code_meanings={
                1: "filesystem / load error",
                2: "Q/DQ ops remain after the fold passes -- unhandled pattern; "
                    "inspect with Netron and extend dequantize_onnx.py",
                3: "output ONNX failed onnx.checker validation",
            },
        ),
        Stage(
            number=3,
            name="ONNX -> PyTorch",
            script=_SCRIPT_DIR / "convert_onnx_pytorch.py",
            input_path=_WEIGHTS_DIR / "model.fp32.onnx",
            output_path=_WEIGHTS_DIR / "model.pt",
            description="Convert the FP32 ONNX to PyTorch via onnx2torch and run a forward-pass smoke test.",
            runtime_hint="~5-15s",
            exit_code_meanings={
                1: "filesystem / load error",
                2: "input still contains Q/DQ -- re-run stage 2",
                3: "onnx2torch conversion raised (unsupported op or graph structure)",
                4: "smoke-test forward pass failed (likely input shape mismatch)",
            },
        ),
    ]


# --------------------------------------------------------------------------- #
# Pretty output                                                               #
# --------------------------------------------------------------------------- #


class _Style:
    """ANSI escape codes with TTY-aware no-op fallback."""

    _enabled: bool = sys.stdout.isatty()

    RESET   = "\033[0m"   if _enabled else ""
    BOLD    = "\033[1m"   if _enabled else ""
    DIM     = "\033[2m"   if _enabled else ""
    RED     = "\033[31m"  if _enabled else ""
    GREEN   = "\033[32m"  if _enabled else ""
    YELLOW  = "\033[33m"  if _enabled else ""
    BLUE    = "\033[34m"  if _enabled else ""
    CYAN    = "\033[36m"  if _enabled else ""

    @classmethod
    def wrap(cls, color: str, text: str) -> str:
        return f"{color}{text}{cls.RESET}"


_RULE_CHAR = "="
_RULE_WIDTH = 78


def _print_rule(char: str = _RULE_CHAR, color: str = _Style.DIM) -> None:
    print(_Style.wrap(color, char * _RULE_WIDTH))


def _print_banner(title: str, subtitle: str = "") -> None:
    print()
    _print_rule()
    print(_Style.wrap(_Style.BOLD + _Style.CYAN, f"  {title}"))
    if subtitle:
        print(_Style.wrap(_Style.DIM, f"  {subtitle}"))
    _print_rule()


def _print_section(title: str) -> None:
    print()
    print(_Style.wrap(_Style.BOLD, title))
    print(_Style.wrap(_Style.DIM, "-" * len(title)))


def _info(msg: str) -> None:
    print(f"  {_Style.wrap(_Style.BLUE, '[info]')} {msg}")


def _warn(msg: str) -> None:
    print(f"  {_Style.wrap(_Style.YELLOW, '[warn]')} {msg}")


def _ok(msg: str) -> None:
    print(f"  {_Style.wrap(_Style.GREEN, '[ ok ]')} {msg}")


def _fail(msg: str) -> None:
    print(f"  {_Style.wrap(_Style.RED, '[fail]')} {msg}")


def _human_bytes(n: int) -> str:
    """Render a byte count as a short human-readable string."""
    for unit, factor in (("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)):
        if n >= factor:
            return f"{n / factor:.1f} {unit}"
    return f"{n} B"


# --------------------------------------------------------------------------- #
# Pre-flight                                                                  #
# --------------------------------------------------------------------------- #


@dataclass
class PreflightReport:
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures


def _check_python_version(report: PreflightReport) -> None:
    if sys.version_info < _MIN_PYTHON:
        report.failures.append(
            f"Python {_MIN_PYTHON[0]}.{_MIN_PYTHON[1]}+ required, "
            f"got {sys.version_info.major}.{sys.version_info.minor}"
        )
    else:
        _ok(f"python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")


def _check_packages(report: PreflightReport) -> None:
    missing: list[tuple[str, str]] = []
    for pkg, used_by in _REQUIRED_PACKAGES:
        if importlib.util.find_spec(pkg) is None:
            missing.append((pkg, used_by))
        else:
            _ok(f"package: {pkg}  ({_Style.wrap(_Style.DIM, used_by)})")
    if missing:
        names = ", ".join(p for p, _ in missing)
        report.failures.append(
            "missing required packages: " + names + "\n"
            "         install with:  pip install -r requirements.txt"
        )


def _check_stages_runnable(stages: Sequence[Stage], report: PreflightReport) -> None:
    for stage in stages:
        if not stage.script.is_file():
            report.failures.append(f"stage {stage.number} script missing: {stage.script}")
        else:
            _ok(f"stage {stage.number} script: {stage.script.name}")


def _check_input_present(stages: Sequence[Stage], report: PreflightReport) -> None:
    initial_input = stages[0].input_path
    if not initial_input.is_file():
        report.failures.append(
            f"pipeline input not found: {initial_input}\n"
            f"         place the source .tflite at this path before running."
        )
    else:
        size = _human_bytes(initial_input.stat().st_size)
        _ok(f"pipeline input: {initial_input.name} ({size})")


def _check_weights_dir_writable(report: PreflightReport) -> None:
    if not _WEIGHTS_DIR.is_dir():
        report.failures.append(f"weights directory does not exist: {_WEIGHTS_DIR}")
        return
    # Light probe: try to create-and-delete a file.
    probe = _WEIGHTS_DIR / ".pipeline_writable_probe"
    try:
        probe.touch()
        probe.unlink()
        _ok(f"weights dir writable: {_WEIGHTS_DIR}")
    except OSError as e:
        report.failures.append(f"weights dir is not writable ({_WEIGHTS_DIR}): {e}")


def _warn_about_stale_outputs(stages: Sequence[Stage], report: PreflightReport) -> None:
    """Note any pre-existing intermediate outputs that this run will overwrite."""
    for stage in stages:
        if stage.output_path.is_file():
            age = time.time() - stage.output_path.stat().st_mtime
            mins = age / 60
            note = f"will overwrite existing {stage.output_path.name} (age: {mins:.1f}m)"
            report.warnings.append(note)
            _warn(note)


def _run_preflight(stages: Sequence[Stage]) -> PreflightReport:
    _print_banner("Pre-flight", "checking environment, dependencies, and inputs")
    report = PreflightReport()
    _check_python_version(report)
    _check_packages(report)
    _check_stages_runnable(stages, report)
    _check_input_present(stages, report)
    _check_weights_dir_writable(report)
    _warn_about_stale_outputs(stages, report)

    print()
    if report.ok:
        _ok(f"pre-flight passed ({len(report.warnings)} warning(s))")
    else:
        _fail(f"pre-flight failed: {len(report.failures)} blocker(s)")
        for f in report.failures:
            print(f"         - {f}")
    return report


# --------------------------------------------------------------------------- #
# Stage execution                                                             #
# --------------------------------------------------------------------------- #


@dataclass
class StageResult:
    stage: Stage
    exit_code: int
    elapsed_seconds: float
    output_size_bytes: int = 0

    @property
    def succeeded(self) -> bool:
        return self.exit_code == 0


def _run_stage(stage: Stage) -> StageResult:
    _print_banner(
        title=f"{stage.label}",
        subtitle=stage.description,
    )
    _info(f"input:    {stage.input_path}")
    _info(f"output:   {stage.output_path}")
    _info(f"runtime:  {stage.runtime_hint}")
    _info(f"command:  python3 {stage.script.name} --input ... --output ...")
    print()
    _print_rule("-")

    start = time.perf_counter()
    proc = subprocess.run(
        [
            sys.executable,
            str(stage.script),
            "--input", str(stage.input_path),
            "--output", str(stage.output_path),
        ],
        check=False,
    )
    elapsed = time.perf_counter() - start

    output_size = (
        stage.output_path.stat().st_size
        if stage.output_path.is_file()
        else 0
    )
    result = StageResult(
        stage=stage,
        exit_code=proc.returncode,
        elapsed_seconds=elapsed,
        output_size_bytes=output_size,
    )

    _print_rule("-")
    if result.succeeded:
        _ok(
            f"stage {stage.number} done in {elapsed:.1f}s -- "
            f"{stage.output_path.name} ({_human_bytes(output_size)})"
        )
    else:
        _fail(f"stage {stage.number} failed with exit code {proc.returncode} after {elapsed:.1f}s")
        meaning = stage.exit_code_meanings.get(proc.returncode)
        if meaning:
            print(f"         meaning: {meaning}")
        else:
            print(f"         (no documented meaning for exit code {proc.returncode})")
        print(
            f"         to debug, re-run with verbose logging:\n"
            f"             python3 {stage.script} \\\n"
            f"                 --input  {stage.input_path} \\\n"
            f"                 --output {stage.output_path} \\\n"
            f"                 --verbose"
        )
    return result


# --------------------------------------------------------------------------- #
# Summary                                                                     #
# --------------------------------------------------------------------------- #


def _print_summary(results: Sequence[StageResult], total_elapsed: float) -> None:
    _print_banner("Summary")

    # Header
    print(f"  {'stage':<28} {'status':<6} {'time':>8} {'output':>14}")
    print(f"  {'-' * 28} {'-' * 6} {'-' * 8} {'-' * 14}")

    for r in results:
        # Pad the visible text BEFORE applying color escapes, otherwise
        # f-string width specifiers count the (zero-visible-width) escape
        # codes and produce misaligned columns.
        if r.succeeded:
            status = _Style.wrap(_Style.GREEN, f"{'ok':<6}")
            output = _human_bytes(r.output_size_bytes)
        else:
            status = _Style.wrap(_Style.RED, f"{'FAIL':<6}")
            output = "-"
        name = r.stage.label
        time_str = f"{r.elapsed_seconds:.1f}s"
        print(f"  {name:<28} {status} {time_str:>8} {output:>14}")

    print()
    print(f"  total wall-clock: {total_elapsed:.1f}s")

    if all(r.succeeded for r in results):
        print()
        _ok("pipeline complete")
        final = results[-1].stage.output_path
        print(f"         final artifact: {final}")
    else:
        first_failure = next(r for r in results if not r.succeeded)
        print()
        _fail(
            f"pipeline aborted at stage {first_failure.stage.number} "
            f"({first_failure.stage.name})"
        )


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #


def main() -> int:
    stages = _build_stages()

    preflight = _run_preflight(stages)
    if not preflight.ok:
        return 10

    results: list[StageResult] = []
    pipeline_start = time.perf_counter()
    for stage in stages:
        result = _run_stage(stage)
        results.append(result)
        if not result.succeeded:
            break  # do not run later stages on stale / missing inputs
    total = time.perf_counter() - pipeline_start

    _print_summary(results, total)

    if results and not results[-1].succeeded:
        return results[-1].exit_code
    return 0


if __name__ == "__main__":
    sys.exit(main())
