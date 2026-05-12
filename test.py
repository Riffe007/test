"""generate_report.py — render evaluate.py output as a standalone HTML report.

Reads the ``evaluation_results.json`` produced by
``executorch-toolkit/evaluation/mobilenetv2/evaluate.py`` and emits a
self-contained HTML file with metric tables, side-by-side branch comparisons,
and best-value highlighting. No external dependencies — single file,
openable in any browser.

The renderer is intentionally defensive about JSON structure: branch detection,
metric lookup, and latency lookup all try several plausible key layouts and
fall back gracefully when fields are absent. Any branch added later — e.g.
an ``executorch_baseline`` block — appears in the report automatically with
no code changes.

Usage::

    python generate_report.py \\
        --results output/baseline_voc/evaluation_results.json \\
        --output  output/baseline_voc/evaluation_report.html
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import logging
import sys
from pathlib import Path
from typing import Any, Final

# ---------------------------------------------------------------------------
# Metric registry — mirrors the 8 keys defined in detection_metrics.py.
# ---------------------------------------------------------------------------
HEADLINE_METRICS: Final[list[str]] = ["mAP_0.5", "mAP_0.5_0.95"]
ALL_METRICS: Final[list[str]] = [
    "mean_precision", "mean_recall", "mean_f1", "mean_iou",
    "mAP_0.5", "mAP_0.65", "mAP_0.75", "mAP_0.5_0.95",
]
METRIC_LABELS: Final[dict[str, str]] = {
    "mean_precision": "Mean Precision",
    "mean_recall":    "Mean Recall",
    "mean_f1":        "Mean F1",
    "mean_iou":       "Mean IoU",
    "mAP_0.5":        "mAP @ 0.5",
    "mAP_0.65":       "mAP @ 0.65",
    "mAP_0.75":       "mAP @ 0.75",
    "mAP_0.5_0.95":   "mAP @ 0.5:0.95",
}

CSS = """
:root {
  --bg: #ffffff;
  --fg: #111827;
  --muted: #6b7280;
  --border: #e5e7eb;
  --accent: #2563eb;
  --accent-bg: #eff6ff;
  --row-alt: #f9fafb;
  --header-bg: #f3f4f6;
  --mono: "SF Mono", Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
}
* { box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  margin: 0 auto;
  padding: 2rem 1.5rem 3rem;
  color: var(--fg);
  background: var(--bg);
  max-width: 1100px;
  line-height: 1.5;
}
h1 { margin: 0 0 0.25rem 0; font-size: 1.5rem; font-weight: 600; }
h2 {
  margin: 2rem 0 0.75rem 0;
  font-size: 1.1rem;
  font-weight: 600;
  border-bottom: 1px solid var(--border);
  padding-bottom: 0.4rem;
}
h3 { margin: 1.5rem 0 0.5rem 0; font-size: 0.95rem; font-weight: 600; color: var(--fg); }
.subtitle { color: var(--muted); margin: 0 0 1.5rem 0; font-size: 0.85rem; }
table { border-collapse: collapse; width: 100%; font-size: 0.875rem; margin: 0.5rem 0; }
th, td { padding: 0.55rem 0.85rem; text-align: left; border-bottom: 1px solid var(--border); }
th { background: var(--header-bg); font-weight: 600; }
tbody tr:nth-child(even) { background: var(--row-alt); }
td.metric-name { font-weight: 500; }
td.num { font-family: var(--mono); text-align: right; }
td.best { color: var(--accent); font-weight: 600; background: var(--accent-bg); }
.meta-grid {
  display: grid;
  grid-template-columns: max-content 1fr;
  gap: 0.3rem 1.5rem;
  font-size: 0.875rem;
  margin: 0.5rem 0 1rem;
}
.meta-grid dt { color: var(--muted); }
.meta-grid dd { margin: 0; font-family: var(--mono); word-break: break-all; }
.note { font-size: 0.8rem; color: var(--muted); margin: 0.25rem 0 0; }
footer {
  margin-top: 3rem;
  padding-top: 1rem;
  border-top: 1px solid var(--border);
  color: var(--muted);
  font-size: 0.75rem;
  text-align: center;
}
"""

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
_LOG = logging.getLogger("generate_report")


# ---------------------------------------------------------------------------
# Defensive JSON extraction — tolerant of layout variations.
# ---------------------------------------------------------------------------
def _extract_branches(results: dict) -> list[tuple[str, dict]]:
    """Return ``[(display_name, branch_dict), ...]`` for every model branch.

    A branch is any top-level dict that carries *evidence of an eval result*:
    metric values, a metrics sub-dict, OR model identification (``model_path``
    / ``model_size_mb``). The ``name`` field alone is not enough — dataset
    metadata blocks often carry a name without any metrics, and those should
    not appear as columns. Future branches (e.g. ExecuTorch) are picked up
    automatically since they'll carry metrics or model info by definition.
    """
    branches: list[tuple[str, dict]] = []
    for key, value in results.items():
        if not isinstance(value, dict):
            continue
        has_metrics = (
            isinstance(value.get("metrics"), dict)
            or any(
                isinstance(value.get(f"metrics_{scope}"), dict)
                for scope in ("class_agnostic", "voc_restricted", "class_aware")
            )
            or any(m in value for m in ALL_METRICS)
        )
        has_model_info = "model_path" in value or "model_size_mb" in value
        if has_metrics or has_model_info:
            display = value.get("name") or key.replace("_baseline", "").replace("_", " ").title()
            branches.append((display, value))
    return branches


def _get_metric(branch: dict, metric: str, scope: str = "class_agnostic") -> Any:
    """Read a metric value, tolerating several plausible nesting layouts."""
    candidates = [
        branch.get("metrics", {}).get(scope, {}).get(metric) if isinstance(branch.get("metrics"), dict) else None,
        branch.get(f"metrics_{scope}", {}).get(metric) if isinstance(branch.get(f"metrics_{scope}"), dict) else None,
        branch.get(scope, {}).get(metric) if isinstance(branch.get(scope), dict) else None,
        branch.get("metrics", {}).get(metric) if isinstance(branch.get("metrics"), dict) else None,
        branch.get(metric),
    ]
    for c in candidates:
        if c is not None:
            return c
    return None


def _get_latency_stats(branch: dict) -> dict[str, float]:
    """Extract a uniform ``{stat: value}`` mapping from a branch's latency block."""
    raw = branch.get("latency_ms") or branch.get("times_ms") or branch.get("latency")
    if raw is None:
        return {}
    if isinstance(raw, (int, float)):
        return {"mean": float(raw)}
    if isinstance(raw, list) and raw:
        return {"mean": sum(raw) / len(raw)}
    if isinstance(raw, dict):
        return {k: float(v) for k, v in raw.items() if isinstance(v, (int, float))}
    return {}


# ---------------------------------------------------------------------------
# Formatting.
# ---------------------------------------------------------------------------
def _fmt_num(value: Any, decimals: int = 4) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return html.escape(str(value))


def _fmt_size_mb(value: Any) -> str:
    return "—" if value is None else f"{_fmt_num(value, 2)} MB"


def _fmt_latency(value: Any) -> str:
    return "—" if value is None else f"{_fmt_num(value, 2)} ms"


def _best_index(values: list[Any], higher_is_better: bool = True) -> int | None:
    """Index of the best (or None if all missing/equal)."""
    numeric = [(i, float(v)) for i, v in enumerate(values) if isinstance(v, (int, float))]
    if len(numeric) < 2:
        return None
    chooser = max if higher_is_better else min
    return chooser(numeric, key=lambda iv: iv[1])[0]


# ---------------------------------------------------------------------------
# Table rendering.
# ---------------------------------------------------------------------------
def _render_metric_row(label: str, values: list[Any], higher_is_better: bool = True) -> str:
    """Render one metric row with the best value highlighted."""
    best = _best_index(values, higher_is_better=higher_is_better)
    cells = []
    for i, v in enumerate(values):
        cls = "num best" if i == best else "num"
        cells.append(f'<td class="{cls}">{_fmt_num(v)}</td>')
    return f'<tr><td class="metric-name">{html.escape(label)}</td>{"".join(cells)}</tr>'


def _render_metrics_table(branches: list[tuple[str, dict]], scope: str) -> str:
    headers = "".join(f"<th>{html.escape(name)}</th>" for name, _ in branches)
    rows: list[str] = []
    for metric in ALL_METRICS:
        values = [_get_metric(branch, metric, scope) for _, branch in branches]
        rows.append(_render_metric_row(METRIC_LABELS.get(metric, metric), values))
    return (
        f'<table><thead><tr><th>Metric</th>{headers}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>'
    )


def _render_summary_table(branches: list[tuple[str, dict]]) -> str:
    headers = ["<th>Model</th>", "<th>Size</th>", "<th>Latency (mean)</th>"]
    for m in HEADLINE_METRICS:
        headers.append(f"<th>{html.escape(METRIC_LABELS.get(m, m))}</th>")

    rows: list[str] = []
    sizes = [b.get("model_size_mb") for _, b in branches]
    latencies = [_get_latency_stats(b).get("mean") for _, b in branches]
    metric_columns = {
        m: [_get_metric(b, m) for _, b in branches] for m in HEADLINE_METRICS
    }

    # Smaller size and lower latency are better; metrics are higher-better.
    best_size = _best_index(sizes, higher_is_better=False)
    best_latency = _best_index(latencies, higher_is_better=False)
    best_metric = {m: _best_index(metric_columns[m]) for m in HEADLINE_METRICS}

    for i, (name, _) in enumerate(branches):
        cells = [f'<td class="metric-name">{html.escape(name)}</td>']
        cells.append(f'<td class="{"num best" if i == best_size else "num"}">{_fmt_size_mb(sizes[i])}</td>')
        cells.append(f'<td class="{"num best" if i == best_latency else "num"}">{_fmt_latency(latencies[i])}</td>')
        for m in HEADLINE_METRICS:
            cls = "num best" if i == best_metric[m] else "num"
            cells.append(f'<td class="{cls}">{_fmt_num(metric_columns[m][i])}</td>')
        rows.append(f"<tr>{''.join(cells)}</tr>")

    return (
        f'<table><thead><tr>{"".join(headers)}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>'
    )


def _render_config(config: dict) -> str:
    if not config:
        return '<p class="note">No configuration block found in results.</p>'
    items = "".join(
        f"<dt>{html.escape(str(k))}</dt><dd>{html.escape(str(v))}</dd>"
        for k, v in config.items()
    )
    return f'<dl class="meta-grid">{items}</dl>'


def _render_per_branch_detail(branches: list[tuple[str, dict]]) -> str:
    sections: list[str] = []
    for name, branch in branches:
        items: list[str] = []
        for key in ("model_path", "model_size_mb"):
            if key in branch:
                items.append(
                    f"<dt>{html.escape(key)}</dt>"
                    f"<dd>{html.escape(str(branch[key]))}</dd>"
                )
        for stat, val in _get_latency_stats(branch).items():
            items.append(f"<dt>latency {html.escape(stat)} (ms)</dt><dd>{_fmt_num(val, 2)}</dd>")
        body = f'<dl class="meta-grid">{"".join(items)}</dl>' if items else '<p class="note">No detail fields found.</p>'
        sections.append(f"<h3>{html.escape(name)}</h3>{body}")
    return "".join(sections)


# ---------------------------------------------------------------------------
# Top-level render.
# ---------------------------------------------------------------------------
def render_html(results_path: Path, results: dict) -> str:
    branches = _extract_branches(results)
    config = results.get("config") if isinstance(results.get("config"), dict) else {}
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    body: list[str] = [
        '<h1>MobileNetV2 SSD Detection Evaluation</h1>',
        f'<p class="subtitle">Generated {html.escape(now)} · '
        f'Source: <code>{html.escape(str(results_path))}</code></p>',
        '<h2>Configuration</h2>',
        _render_config(config),
    ]

    if not branches:
        body.append('<h2>Results</h2>')
        body.append('<p class="note">No model branches detected in results JSON.</p>')
    else:
        body.append('<h2>Summary</h2>')
        body.append('<p class="note">Headline comparison — best value per column highlighted.</p>')
        body.append(_render_summary_table(branches))

        body.append('<h2>Full Metrics — Class-Agnostic</h2>')
        body.append('<p class="note">Best value per row highlighted. Higher is better for all 8 metrics.</p>')
        body.append(_render_metrics_table(branches, scope="class_agnostic"))

        # Render VOC-restricted table only if any branch carries those metrics.
        has_voc_restricted = any(
            _get_metric(b, "mAP_0.5", "voc_restricted") is not None
            or _get_metric(b, "mAP_0.5", "class_aware") is not None
            for _, b in branches
        )
        if has_voc_restricted:
            body.append('<h2>Full Metrics — VOC-Restricted</h2>')
            body.append('<p class="note">Restricted to the 20 VOC-overlap COCO classes (per evaluate.py).</p>')
            body.append(_render_metrics_table(branches, scope="voc_restricted"))

        body.append('<h2>Per-Model Detail</h2>')
        body.append(_render_per_branch_detail(branches))

    body.append(f'<footer>generate_report.py · {html.escape(now)}</footer>')

    return (
        '<!DOCTYPE html>\n<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        '<title>MobileNetV2 SSD Evaluation Report</title>\n'
        f'<style>{CSS}</style>\n'
        '</head>\n<body>\n'
        + "\n".join(body)
        + '\n</body>\n</html>\n'
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--results", type=Path, required=True,
                   help="Path to evaluation_results.json produced by evaluate.py")
    p.add_argument("--output", type=Path, required=True,
                   help="Output HTML report path")
    args = p.parse_args(argv)

    if not args.results.is_file():
        raise FileNotFoundError(args.results)

    with args.results.open() as f:
        results = json.load(f)

    html_out = render_html(args.results, results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html_out)

    branches = _extract_branches(results)
    _LOG.info("wrote %s", args.output)
    _LOG.info("  branches: %d (%s)", len(branches), ", ".join(n for n, _ in branches) or "none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
