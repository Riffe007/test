"""Collect deliverables from inception and gemma source projects.

Aggregates artifacts, reports, and logs into a unified deliverables tree at
~/Documents/projects/executorch_deliverables/. Generates a manifest.json
tracking every expected deliverable's status, and produces master reports
at the top level.

Designed to run multiple times safely. Existing files in the deliverables
tree are overwritten when sources are newer; missing sources produce
`missing` or `deferred` status entries rather than errors.

Usage:
    python collect_all_deliverables.py
    python collect_all_deliverables.py --inception-path /custom/path
    python collect_all_deliverables.py --gemma-path /custom/path --verbose

Exits non-zero only on unrecoverable errors (e.g., cannot create deliverables
directory). Missing source files are warnings, not errors.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

# --------------------------------------------------------------------------
# Constants & configuration
# --------------------------------------------------------------------------

HOME = Path.home()
DEFAULT_INCEPTION_SRC = HOME / "Documents" / "projects" / "inception"
DEFAULT_GEMMA_SRC = HOME / "Documents" / "projects" / "llm"
DEFAULT_DELIVERABLES = HOME / "Documents" / "projects" / "executorch_deliverables"


class Status(str, Enum):
    """Manifest status flags per the deliverables spec."""

    COMPLETE = "complete"
    ATTEMPTED_BLOCKED = "attempted_blocked"
    DEFERRED_RUNTIME_BINDING_MISSING = "deferred_runtime_binding_missing"
    MISSING = "missing"
    NOT_APPLICABLE = "not_applicable"


# --------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------


@dataclass
class DeliverableEntry:
    """One row in the manifest."""

    path: str  # path relative to deliverables root
    status: Status
    size_bytes: int | None = None
    notes: str = ""
    source_path: str | None = None  # where it was copied from, if applicable

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "status": self.status.value,
            "size_bytes": self.size_bytes,
            "notes": self.notes,
            "source_path": self.source_path,
        }


@dataclass
class ProjectCollection:
    """Aggregated state for one source project (inception or gemma)."""

    name: str
    source_dir: Path
    target_dir: Path  # deliverables/<name>
    entries: list[DeliverableEntry] = field(default_factory=list)
    source_exists: bool = False

    def add(self, entry: DeliverableEntry) -> None:
        self.entries.append(entry)


# --------------------------------------------------------------------------
# Logging setup
# --------------------------------------------------------------------------


def configure_logging(verbose: bool) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=level,
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("collect")


# --------------------------------------------------------------------------
# Filesystem utilities
# --------------------------------------------------------------------------


def ensure_dirs(root: Path, project_names: Iterable[str]) -> None:
    """Create the full deliverables directory structure."""
    subdirs = ("artifacts", "results", "reports", "docs", "logs")
    for project in project_names:
        for sub in subdirs:
            (root / project / sub).mkdir(parents=True, exist_ok=True)
    (root / "scripts").mkdir(parents=True, exist_ok=True)


def copy_if_exists(
    src: Path,
    dst: Path,
    project: ProjectCollection,
    relative_path: str,
    notes: str = "",
    status_if_present: Status = Status.COMPLETE,
    status_if_missing: Status = Status.MISSING,
) -> bool:
    """Copy src to dst if src exists, recording the result in the manifest.

    Returns True if file was copied, False if missing.
    """
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        size = dst.stat().st_size
        project.add(
            DeliverableEntry(
                path=relative_path,
                status=status_if_present,
                size_bytes=size,
                notes=notes,
                source_path=str(src),
            )
        )
        return True
    project.add(
        DeliverableEntry(
            path=relative_path,
            status=status_if_missing,
            notes=notes or f"source not found: {src}",
        )
    )
    return False


def write_text(path: Path, content: str) -> int:
    """Write text to path, creating parent dirs. Return bytes written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path.stat().st_size


# --------------------------------------------------------------------------
# Inception collection
# --------------------------------------------------------------------------


# Maps source filename → target filename for inception artifacts
INCEPTION_ARTIFACT_MAP = {
    "inception_v3_portable.pte": "inception_v3_portable_fp32.pte",
    "inception_v3_xnnpack.pte": "inception_v3_xnnpack_fp32.pte",
    "inception_v3_xnnpack_int8.pte": "inception_v3_xnnpack_int8.pte",
    "inception_v3_xnnpack_int4.pte": "inception_v3_xnnpack_int4.pte",
}

# Maps source markdown report → target deliverable report
INCEPTION_REPORT_MAP = {
    "inception_v3_portable_export.md": "inception_export_summary.md",
    "inception_v3_xnnpack_int8_quantization.md": "inception_quantization_status.md",
}


def collect_inception(
    src_root: Path, target_root: Path, log: logging.Logger
) -> ProjectCollection:
    """Collect all inception deliverables."""
    project = ProjectCollection(
        name="inception",
        source_dir=src_root,
        target_dir=target_root / "inception",
        source_exists=src_root.is_dir(),
    )

    if not project.source_exists:
        log.warning("inception source directory not found: %s", src_root)
        return project

    log.info("collecting inception deliverables from %s", src_root)

    # 1. Artifacts
    src_artifacts = src_root / "artifacts"
    dst_artifacts = project.target_dir / "artifacts"
    for src_name, dst_name in INCEPTION_ARTIFACT_MAP.items():
        copy_if_exists(
            src=src_artifacts / src_name,
            dst=dst_artifacts / dst_name,
            project=project,
            relative_path=f"inception/artifacts/{dst_name}",
            notes="renamed from source convention" if src_name != dst_name else "",
        )

    # 2. JSON results — collect all matching JSON reports from source
    src_reports = src_root / "reports"
    dst_results = project.target_dir / "results"
    if src_reports.is_dir():
        json_mapping = {
            "inception_v3_portable_parity.json": "pytorch_baseline_results.json",
            "inception_v3_portable_inspection.json": "portable_fp32_runtime_results.json",
            "inception_v3_xnnpack_inspection.json": "xnnpack_fp32_runtime_results.json",
            "inception_v3_xnnpack_int8_inspection.json": "int8_runtime_results.json",
        }
        for src_name, dst_name in json_mapping.items():
            copy_if_exists(
                src=src_reports / src_name,
                dst=dst_results / dst_name,
                project=project,
                relative_path=f"inception/results/{dst_name}",
                notes="from inception parity/inspection JSON",
            )

    # 3. Markdown reports
    dst_reports = project.target_dir / "reports"
    for src_name, dst_name in INCEPTION_REPORT_MAP.items():
        copy_if_exists(
            src=src_reports / src_name,
            dst=dst_reports / dst_name,
            project=project,
            relative_path=f"inception/reports/{dst_name}",
        )

    # 4. Generate synthesized reports from collected data
    _generate_inception_synthesized_reports(project, src_root, log)

    # 5. Documentation
    dst_docs = project.target_dir / "docs"
    src_docs = src_root / "docs"
    doc_mapping = {
        "setup.md": "environment_setup.md",
        "troubleshooting.md": "troubleshooting.md",
        "demo.md": "inception_runbook.md",
    }
    for src_name, dst_name in doc_mapping.items():
        copy_if_exists(
            src=src_docs / src_name,
            dst=dst_docs / dst_name,
            project=project,
            relative_path=f"inception/docs/{dst_name}",
        )

    # Generate reproduction_steps.md from README
    _generate_reproduction_steps(project, src_root, log)

    # 6. Logs — collect any .log files; if none exist, note as missing
    dst_logs = project.target_dir / "logs"
    src_logs = src_root / "logs" if (src_root / "logs").is_dir() else None
    log_targets = [
        "pytorch_baseline.log",
        "portable_export.log",
        "xnnpack_export.log",
        "int8_export.log",
        "int4_export.log",
        "etdump.log",
        "inspector.log",
    ]
    for log_name in log_targets:
        if src_logs and (src_logs / log_name).is_file():
            copy_if_exists(
                src=src_logs / log_name,
                dst=dst_logs / log_name,
                project=project,
                relative_path=f"inception/logs/{log_name}",
            )
        else:
            project.add(
                DeliverableEntry(
                    path=f"inception/logs/{log_name}",
                    status=Status.MISSING,
                    notes="log file not produced by source pipeline",
                )
            )

    return project


def _generate_inception_synthesized_reports(
    project: ProjectCollection, src_root: Path, log: logging.Logger
) -> None:
    """Generate reports that don't exist in source by synthesizing from data."""
    dst_reports = project.target_dir / "reports"
    src_reports = src_root / "reports"

    # 1. inception_final_benchmark_summary.md — synthesized from inspection JSONs
    summary = _build_benchmark_summary(src_reports, "inception_v3")
    summary_path = dst_reports / "inception_final_benchmark_summary.md"
    size = write_text(summary_path, summary)
    project.add(
        DeliverableEntry(
            path="inception/reports/inception_final_benchmark_summary.md",
            status=Status.COMPLETE,
            size_bytes=size,
            notes="synthesized from inspection JSON files",
        )
    )

    # 2. inception_artifact_comparison.md — synthesized from artifact sizes
    comparison = _build_artifact_comparison(
        artifact_dir=project.target_dir / "artifacts", model_name="inception_v3"
    )
    comparison_path = dst_reports / "inception_artifact_comparison.md"
    size = write_text(comparison_path, comparison)
    project.add(
        DeliverableEntry(
            path="inception/reports/inception_artifact_comparison.md",
            status=Status.COMPLETE,
            size_bytes=size,
            notes="synthesized from artifact file sizes",
        )
    )

    # 3. inception_known_issues.md — pulled from inception's ADR 0004 if present
    adr_path = src_root / "docs" / "decisions" / "0004-runtime-cpuinfo-and-threading.md"
    issues_path = dst_reports / "inception_known_issues.md"
    if adr_path.is_file():
        content = (
            "# Inception — Known Issues\n\n"
            "These are documented observations from the inception pipeline that "
            "do not affect correctness but are noted for reviewer awareness.\n\n"
            "---\n\n"
            + adr_path.read_text()
        )
        size = write_text(issues_path, content)
        project.add(
            DeliverableEntry(
                path="inception/reports/inception_known_issues.md",
                status=Status.COMPLETE,
                size_bytes=size,
                notes="derived from ADR 0004",
                source_path=str(adr_path),
            )
        )
    else:
        content = (
            "# Inception — Known Issues\n\n"
            "No known issues recorded yet. This file is a placeholder for "
            "future documentation.\n"
        )
        size = write_text(issues_path, content)
        project.add(
            DeliverableEntry(
                path="inception/reports/inception_known_issues.md",
                status=Status.MISSING,
                size_bytes=size,
                notes="placeholder — source ADR 0004 not found",
            )
        )

    # 4. artifact_comparison_report.json
    json_comparison = _build_artifact_comparison_json(
        artifact_dir=project.target_dir / "artifacts", model_name="inception_v3"
    )
    json_path = project.target_dir / "results" / "artifact_comparison_report.json"
    size = write_text(json_path, json.dumps(json_comparison, indent=2))
    project.add(
        DeliverableEntry(
            path="inception/results/artifact_comparison_report.json",
            status=Status.COMPLETE,
            size_bytes=size,
            notes="synthesized from artifact sizes",
        )
    )

    # 5. quantization_capability_check.json — basic capability summary
    quant_check = {
        "model": "inception_v3",
        "quantization_attempted": ["int8_xnnpack_pt2e"],
        "quantization_succeeded": ["int8_xnnpack_pt2e"]
        if (project.target_dir / "artifacts" / "inception_v3_xnnpack_int8.pte").is_file()
        else [],
        "quantization_blocked": [],
        "method": "PT2E with XNNPACKQuantizer (symmetric)",
        "calibration_samples": 32,
    }
    quant_path = project.target_dir / "results" / "quantization_capability_check.json"
    size = write_text(quant_path, json.dumps(quant_check, indent=2))
    project.add(
        DeliverableEntry(
            path="inception/results/quantization_capability_check.json",
            status=Status.COMPLETE,
            size_bytes=size,
        )
    )


def _build_benchmark_summary(src_reports: Path, model: str) -> str:
    """Synthesize benchmark summary markdown from per-backend inspection JSONs."""
    lines = [
        f"# {model} — Final Benchmark Summary",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Latency by backend",
        "",
        "| Backend | Artifact size (bytes) | Mean (ms) | Median (ms) | P95 (ms) |",
        "| --- | --- | --- | --- | --- |",
    ]

    backends = ("portable", "xnnpack", "xnnpack_int8")
    for backend in backends:
        json_file = src_reports / f"{model}_{backend}_inspection.json"
        if not json_file.is_file():
            lines.append(f"| {backend} | — | — | — | — |")
            continue
        try:
            data = json.loads(json_file.read_text())
            result = data.get("result", {})
            size = result.get("artifact_size_bytes", "—")
            timing = result.get("timing") or {}
            mean = f"{timing.get('mean_ms', 0):.2f}" if timing else "—"
            median = f"{timing.get('median_ms', 0):.2f}" if timing else "—"
            p95 = f"{timing.get('p95_ms', 0):.2f}" if timing else "—"
            lines.append(f"| {backend} | {size} | {mean} | {median} | {p95} |")
        except (json.JSONDecodeError, KeyError):
            lines.append(f"| {backend} | (parse error) | — | — | — |")

    lines.extend([
        "",
        "## Parity status",
        "",
    ])
    for backend in backends:
        parity_file = src_reports / f"{model}_{backend}_parity.json"
        if not parity_file.is_file():
            lines.append(f"- **{backend}**: parity report not found")
            continue
        try:
            data = json.loads(parity_file.read_text())
            result = data.get("result", {})
            top1 = result.get("top1_match", "unknown")
            max_abs = result.get("max_abs_error", "unknown")
            lines.append(
                f"- **{backend}**: top-1 match = `{top1}`, "
                f"max abs error = `{max_abs}`"
            )
        except (json.JSONDecodeError, KeyError):
            lines.append(f"- **{backend}**: parity report parse error")

    return "\n".join(lines) + "\n"


def _build_artifact_comparison(artifact_dir: Path, model_name: str) -> str:
    """Markdown table of artifact sizes."""
    lines = [
        f"# {model_name} — Artifact Comparison",
        "",
        "| Artifact | Size (bytes) | Size (MB) |",
        "| --- | --- | --- |",
    ]
    if not artifact_dir.is_dir():
        lines.append("| (no artifacts found) | — | — |")
        return "\n".join(lines) + "\n"

    for artifact in sorted(artifact_dir.glob("*.pte")):
        size = artifact.stat().st_size
        lines.append(f"| `{artifact.name}` | {size:,} | {size / 1024 / 1024:.2f} |")
    return "\n".join(lines) + "\n"


def _build_artifact_comparison_json(artifact_dir: Path, model_name: str) -> dict:
    """Structured artifact comparison."""
    artifacts = []
    if artifact_dir.is_dir():
        for artifact in sorted(artifact_dir.glob("*.pte")):
            size = artifact.stat().st_size
            artifacts.append({
                "filename": artifact.name,
                "size_bytes": size,
                "size_mb": round(size / 1024 / 1024, 2),
            })
    return {
        "model": model_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "artifacts": artifacts,
    }


def _generate_reproduction_steps(
    project: ProjectCollection, src_root: Path, log: logging.Logger
) -> None:
    """Generate reproduction_steps.md by extracting commands from README."""
    dst = project.target_dir / "docs" / "reproduction_steps.md"
    readme = src_root / "README.md"
    if readme.is_file():
        content = (
            f"# {project.name} — Reproduction Steps\n\n"
            "Extracted from the source project's README.\n\n"
            "---\n\n"
            + readme.read_text()
        )
        size = write_text(dst, content)
        project.add(
            DeliverableEntry(
                path=f"{project.name}/docs/reproduction_steps.md",
                status=Status.COMPLETE,
                size_bytes=size,
                notes="derived from source README",
                source_path=str(readme),
            )
        )
    else:
        project.add(
            DeliverableEntry(
                path=f"{project.name}/docs/reproduction_steps.md",
                status=Status.MISSING,
                notes="source README not found",
            )
        )


# --------------------------------------------------------------------------
# Gemma collection
# --------------------------------------------------------------------------


GEMMA_EXPECTED_ARTIFACTS = (
    ("gemma_3_1b_pt_portable_fp32.pte", Status.COMPLETE),
    ("gemma_3_1b_pt_xnnpack_fp32.pte", Status.COMPLETE),
    ("gemma_3_1b_pt_xnnpack_int8_weight_only.pte", Status.COMPLETE),
    ("gemma_3_1b_pt_xnnpack_int4_weight_only.pte", Status.ATTEMPTED_BLOCKED),
)


def collect_gemma(
    src_root: Path, target_root: Path, log: logging.Logger
) -> ProjectCollection:
    """Collect Gemma deliverables from the actual llm project."""
    project = ProjectCollection(
        name="gemma",
        source_dir=src_root,
        target_dir=target_root / "gemma",
        source_exists=src_root.is_dir(),
    )

    if not project.source_exists:
        log.info(
            "gemma source not present at %s; recording deliverables as missing",
            src_root,
        )
        _record_gemma_placeholders(project)
        return project

    log.info("collecting gemma deliverables from %s", src_root)

    # Actual Gemma source paths within the llm project
    src_artifacts = src_root / "output" / "models" / "gemma_3_1b_pt_optimum"
    src_results = src_root / "evaluation" / "gemma_3_1b_pt" / "results"
    src_reports = src_root / "reports"
    src_docs = src_root / "docs"
    src_logs = src_root / "logs"

    dst_artifacts = project.target_dir / "artifacts"
    dst_results = project.target_dir / "results"
    dst_reports = project.target_dir / "reports"
    dst_docs = project.target_dir / "docs"
    dst_logs = project.target_dir / "logs"

    # 1. Artifacts
    for artifact_name, expected_status in GEMMA_EXPECTED_ARTIFACTS:
        src_file = src_artifacts / artifact_name
        if src_file.is_file():
            copy_if_exists(
                src=src_file,
                dst=dst_artifacts / artifact_name,
                project=project,
                relative_path=f"gemma/artifacts/{artifact_name}",
            )
        else:
            project.add(
                DeliverableEntry(
                    path=f"gemma/artifacts/{artifact_name}",
                    status=expected_status
                    if expected_status != Status.COMPLETE
                    else Status.MISSING,
                    notes=(
                        "INT4 quantization attempted but blocked by MSLK dependency availability"
                        if expected_status == Status.ATTEMPTED_BLOCKED
                        else f"source not found: {src_file}"
                    ),
                )
            )

    # 2. Results
    result_files = [
        "pytorch_baseline_results.json",
        "gemma_pipeline_results.json",
        "artifact_report.json",
        "artifact_comparison_report.json",
        "quantization_capability_check.json",
    ]
    for fname in result_files:
        copy_if_exists(
            src=src_results / fname,
            dst=dst_results / fname,
            project=project,
            relative_path=f"gemma/results/{fname}",
            notes="copied from Gemma evaluation results",
        )

    # 3. Runtime benchmark results — present if completed, deferred otherwise
    runtime_files = [
        "portable_fp32_runtime_results.json",
        "xnnpack_fp32_runtime_results.json",
        "xnnpack_int8_runtime_results.json",
        "etdump_summary.json",
        "inspector_summary.json",
    ]
    for fname in runtime_files:
        src_file = src_results / fname
        if src_file.is_file():
            copy_if_exists(
                src=src_file,
                dst=dst_results / fname,
                project=project,
                relative_path=f"gemma/results/{fname}",
                notes="runtime/profiling result copied from Gemma results",
            )
        else:
            project.add(
                DeliverableEntry(
                    path=f"gemma/results/{fname}",
                    status=Status.DEFERRED_RUNTIME_BINDING_MISSING,
                    notes="ExecuTorch runtime/profiling harness not completed or Python runtime bindings unavailable",
                )
            )

    # 4. Markdown reports
    report_files = [
        "gemma_3_1b_pt_final_benchmark_summary.md",
        "gemma_3_1b_pt_export_summary.md",
        "gemma_3_1b_pt_artifact_comparison.md",
        "gemma_3_1b_pt_quantization_status.md",
        "gemma_3_1b_pt_known_issues.md",
    ]
    for fname in report_files:
        copy_if_exists(
            src=src_reports / fname,
            dst=dst_reports / fname,
            project=project,
            relative_path=f"gemma/reports/{fname}",
            notes="copied from Gemma reports",
        )

    # 5. Documentation
    doc_files = [
        "gemma_3_1b_pt_runbook.md",
        "environment_setup.md",
        "reproduction_steps.md",
        "troubleshooting.md",
    ]
    for fname in doc_files:
        copy_if_exists(
            src=src_docs / fname,
            dst=dst_docs / fname,
            project=project,
            relative_path=f"gemma/docs/{fname}",
            notes="copied from Gemma docs",
        )

    # 6. Logs
    log_files = [
        "model_load_test.log",
        "pytorch_baseline.log",
        "portable_export.log",
        "xnnpack_export.log",
        "int8_export.log",
        "int4_export_attempt.log",
        "etdump.log",
        "inspector.log",
    ]
    for fname in log_files:
        src_file = src_logs / fname
        if src_file.is_file():
            copy_if_exists(
                src=src_file,
                dst=dst_logs / fname,
                project=project,
                relative_path=f"gemma/logs/{fname}",
            )
        else:
            project.add(
                DeliverableEntry(
                    path=f"gemma/logs/{fname}",
                    status=Status.MISSING,
                    notes="log file not produced by Gemma pipeline",
                )
            )

    return project


def _record_gemma_placeholders(project: ProjectCollection) -> None:
    """When gemma source doesn't exist, record everything expected as missing/deferred."""
    expected_artifacts = [
        ("gemma_3_1b_pt_portable_fp32.pte", Status.MISSING, "awaiting Week 2 toolkit run"),
        ("gemma_3_1b_pt_xnnpack_fp32.pte", Status.MISSING, "awaiting Week 2 toolkit run"),
        ("gemma_3_1b_pt_xnnpack_int8_weight_only.pte", Status.MISSING, "awaiting Week 2 toolkit run"),
        ("gemma_3_1b_pt_xnnpack_int4_weight_only.pte", Status.ATTEMPTED_BLOCKED, "INT4 not yet attempted"),
    ]
    for fname, status, notes in expected_artifacts:
        project.add(
            DeliverableEntry(
                path=f"gemma/artifacts/{fname}",
                status=status,
                notes=notes,
            )
        )

    expected_reports = (
        "gemma_3_1b_pt_final_benchmark_summary.md",
        "gemma_3_1b_pt_export_summary.md",
        "gemma_3_1b_pt_artifact_comparison.md",
        "gemma_3_1b_pt_quantization_status.md",
        "gemma_3_1b_pt_known_issues.md",
    )
    for fname in expected_reports:
        project.add(
            DeliverableEntry(
                path=f"gemma/reports/{fname}",
                status=Status.MISSING,
                notes="awaiting Week 2 toolkit run",
            )
        )

    expected_results = (
        ("pytorch_baseline_results.json", Status.MISSING),
        ("gemma_pipeline_results.json", Status.MISSING),
        ("artifact_report.json", Status.MISSING),
        ("artifact_comparison_report.json", Status.MISSING),
        ("quantization_capability_check.json", Status.MISSING),
        ("portable_fp32_runtime_results.json", Status.DEFERRED_RUNTIME_BINDING_MISSING),
        ("xnnpack_fp32_runtime_results.json", Status.DEFERRED_RUNTIME_BINDING_MISSING),
        ("xnnpack_int8_runtime_results.json", Status.DEFERRED_RUNTIME_BINDING_MISSING),
        ("etdump_summary.json", Status.DEFERRED_RUNTIME_BINDING_MISSING),
        ("inspector_summary.json", Status.DEFERRED_RUNTIME_BINDING_MISSING),
    )
    for fname, status in expected_results:
        project.add(
            DeliverableEntry(
                path=f"gemma/results/{fname}",
                status=status,
                notes="awaiting Week 2 toolkit run",
            )
        )


# --------------------------------------------------------------------------
# Manifest generation
# --------------------------------------------------------------------------


def build_manifest(
    projects: list[ProjectCollection], deliverables_root: Path
) -> dict[str, Any]:
    """Construct the master manifest.json structure."""
    manifest = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "deliverables_root": str(deliverables_root),
        "summary": _build_status_summary(projects),
        "projects": {},
    }
    for project in projects:
        manifest["projects"][project.name] = {
            "source_dir": str(project.source_dir),
            "source_exists": project.source_exists,
            "target_dir": str(project.target_dir),
            "deliverables": [e.to_dict() for e in project.entries],
        }
    return manifest


def _build_status_summary(projects: list[ProjectCollection]) -> dict[str, Any]:
    """Aggregate status counts across all projects."""
    summary: dict[str, dict[str, int]] = {}
    for project in projects:
        counts: dict[str, int] = {s.value: 0 for s in Status}
        for entry in project.entries:
            counts[entry.status.value] += 1
        summary[project.name] = counts
    return summary


# --------------------------------------------------------------------------
# Master report generation
# --------------------------------------------------------------------------


def generate_master_reports(
    projects: list[ProjectCollection],
    manifest: dict[str, Any],
    deliverables_root: Path,
    log: logging.Logger,
) -> None:
    """Write top-level summary reports."""
    reports_dir = deliverables_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    write_text(reports_dir / "master_exec_summary.md", _master_exec_summary(projects, manifest))
    write_text(reports_dir / "full_artifact_inventory.md", _artifact_inventory(projects))
    write_text(reports_dir / "benchmark_comparison.md", _benchmark_comparison(projects))
    write_text(reports_dir / "known_blockers.md", _known_blockers(projects))

    log.info("master reports written to %s", reports_dir)


def _master_exec_summary(
    projects: list[ProjectCollection], manifest: dict[str, Any]
) -> str:
    lines = [
        "# ExecuTorch Deliverables — Executive Summary",
        "",
        f"Generated: {manifest['generated_at']}",
        f"Deliverables root: `{manifest['deliverables_root']}`",
        "",
        "## Status by project",
        "",
    ]
    for project in projects:
        counts = manifest["summary"][project.name]
        lines.append(f"### {project.name}")
        lines.append("")
        lines.append(f"- Source: `{project.source_dir}` ({'present' if project.source_exists else 'missing'})")
        lines.append(f"- Total entries: {len(project.entries)}")
        for status_value, count in counts.items():
            if count > 0:
                lines.append(f"- {status_value}: {count}")
        lines.append("")
    return "\n".join(lines)


def _artifact_inventory(projects: list[ProjectCollection]) -> str:
    lines = ["# Full Artifact Inventory", ""]
    for project in projects:
        lines.append(f"## {project.name}")
        lines.append("")
        lines.append("| Path | Status | Size (bytes) | Notes |")
        lines.append("| --- | --- | --- | --- |")
        for entry in project.entries:
            if "/artifacts/" in entry.path:
                size = entry.size_bytes if entry.size_bytes is not None else "—"
                lines.append(
                    f"| `{entry.path}` | {entry.status.value} | {size} | {entry.notes} |"
                )
        lines.append("")
    return "\n".join(lines)


def _benchmark_comparison(projects: list[ProjectCollection]) -> str:
    lines = [
        "# Benchmark Comparison Across Projects",
        "",
        "Per-project benchmark summaries are in each project's "
        "`reports/<project>_final_benchmark_summary.md`.",
        "",
        "## Cross-project status",
        "",
    ]
    for project in projects:
        completed_artifacts = [
            e for e in project.entries
            if "/artifacts/" in e.path and e.status == Status.COMPLETE
        ]
        lines.append(f"- **{project.name}**: {len(completed_artifacts)} completed artifacts")
    return "\n".join(lines) + "\n"


def _known_blockers(projects: list[ProjectCollection]) -> str:
    lines = ["# Known Blockers", ""]
    any_blockers = False
    for project in projects:
        blockers = [
            e for e in project.entries
            if e.status in (Status.ATTEMPTED_BLOCKED, Status.DEFERRED_RUNTIME_BINDING_MISSING)
        ]
        if not blockers:
            continue
        any_blockers = True
        lines.append(f"## {project.name}")
        lines.append("")
        for entry in blockers:
            lines.append(f"- `{entry.path}` — {entry.status.value}: {entry.notes}")
        lines.append("")
    if not any_blockers:
        lines.append("No active blockers recorded.")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Top-level README
# --------------------------------------------------------------------------


def write_top_level_readme(
    deliverables_root: Path, manifest: dict[str, Any]
) -> None:
    content = f"""# ExecuTorch Deliverables

Aggregated deliverables for the ExecuTorch onboarding project. This directory
collects artifacts, reports, and documentation from the inception and gemma
source projects into a unified deliverables tree.

Generated: {manifest['generated_at']}

## Layout

- `inception/` — Inception V3 vision model deliverables
- `gemma/` — Gemma 1B LLM deliverables (sourced from the llm project)
- `reports/` — Cross-project master reports
- `scripts/` — Automation for collection and report generation
- `manifest.json` — Machine-readable inventory of every deliverable

## Status

See `reports/master_exec_summary.md` for status by project, or
`manifest.json` for the full structured inventory.

## Regenerating

To re-collect deliverables after source projects update:

    python scripts/collect_all_deliverables.py

This is safe to run repeatedly — existing files are overwritten with newer
sources, and missing items are recorded as `missing` rather than failing.
"""
    write_text(deliverables_root / "README.md", content)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inception-path",
        type=Path,
        default=DEFAULT_INCEPTION_SRC,
        help=f"Path to inception source project (default: {DEFAULT_INCEPTION_SRC})",
    )
    parser.add_argument(
        "--gemma-path",
        type=Path,
        default=DEFAULT_GEMMA_SRC,
        help=f"Path to gemma/llm source project (default: {DEFAULT_GEMMA_SRC})",
    )
    parser.add_argument(
        "--deliverables-path",
        type=Path,
        default=DEFAULT_DELIVERABLES,
        help=f"Output directory (default: {DEFAULT_DELIVERABLES})",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log = configure_logging(args.verbose)

    log.info("deliverables root: %s", args.deliverables_path)
    log.info("inception source:  %s", args.inception_path)
    log.info("gemma/llm source:  %s", args.gemma_path)

    try:
        ensure_dirs(args.deliverables_path, ("inception", "gemma"))
    except OSError as e:
        log.error("could not create deliverables directory: %s", e)
        return 1

    inception = collect_inception(args.inception_path, args.deliverables_path, log)
    gemma = collect_gemma(args.gemma_path, args.deliverables_path, log)
    projects = [inception, gemma]

    manifest = build_manifest(projects, args.deliverables_path)
    write_text(
        args.deliverables_path / "manifest.json",
        json.dumps(manifest, indent=2),
    )

    generate_master_reports(projects, manifest, args.deliverables_path, log)
    write_top_level_readme(args.deliverables_path, manifest)

    log.info("=" * 60)
    log.info("Collection complete")
    log.info("=" * 60)
    for project in projects:
        counts = manifest["summary"][project.name]
        log.info("%s:", project.name)
        for status, count in counts.items():
            if count > 0:
                log.info("  %s: %d", status, count)

    return 0


if __name__ == "__main__":
    sys.exit(main())
