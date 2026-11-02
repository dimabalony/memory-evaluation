"""Shared IO for the benchmark framework.

The genuinely-generic report plumbing lifted out of
``scripts/run_priming_benchmark.py`` so every benchmark (recall, dedup,
fixture-health, …) shares one implementation instead of hand-wiring it:

- ``current_git_sha`` — stamp reports with the code version
- ``join_s3_key`` — slash-collapsing S3 key join
- ``write_report_outputs`` — JSON + markdown to a directory (date-prefix optional)
- ``redact_list_field`` — strip PII from a per-item list before committing a report
- ``compare_overall_to_baseline`` — per-metric current/baseline/delta + regressed verdict

S3 *uploads* go through the private host benchmark store (which reuses
the cost-archive bucket/prefix); this module deliberately opens no S3 client of
its own. Recall's legacy ``recall-baselines/<date>-recall.json`` archive stays in
the recall script for parity — new benchmarks use ``benchmark_store``'s
``benchmark/results/`` layout via the generic runner.

These helpers change no behavior versus the recall originals; the recall
characterization suite (``tests/scripts/test_priming_benchmark.py``) is the
parity gate that proves it.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def current_git_sha(repo_root: Path | None = None) -> str:
    """Best-effort short SHA. Returns ``'unknown'`` off-tree.

    Resolution order:
    1. ``BENCHMARK_GIT_SHA`` env var (set by CI / containers without ``.git``).
    2. ``GITHUB_SHA`` env var (auto-set by GitHub Actions).
    3. ``git rev-parse --short HEAD`` against ``repo_root``.
    4. Fallback ``'unknown'``.
    """
    import os

    override = os.environ.get("BENCHMARK_GIT_SHA") or os.environ.get("GITHUB_SHA")
    if override:
        normalized = override.strip()
        # Only honor overrides that look like a real SHA — 7..40 hex chars.
        if re.fullmatch(r"[0-9a-fA-F]{7,40}", normalized):
            return normalized[:7]
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            timeout=5,  # a hung git (lock/stuck mount) must not block report writing
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def join_s3_key(*parts: str) -> str:
    """Join S3 key fragments without introducing duplicate slashes."""
    return "/".join(part.strip("/") for part in parts if part and part.strip("/"))


def redact_list_field(
    report: dict[str, Any],
    *,
    list_key: str,
    keep_fields: tuple[str, ...],
) -> dict[str, Any]:
    """Return a shallow copy of ``report`` with ``report[list_key]`` reduced to
    only ``keep_fields`` per row.

    Committed baselines should not embed raw per-item content (question text,
    trace UUIDs, …). Aggregate per-item fields (id, language, metrics) are kept
    so the baseline still supports per-item comparison.
    """
    redacted = dict(report)
    redacted[list_key] = [{k: row[k] for k in keep_fields if k in row} for row in report.get(list_key, [])]
    return redacted


def write_report_outputs(
    report: dict[str, Any],
    run_tag: str,
    out_dir: Path,
    *,
    render_markdown: Callable[[dict[str, Any]], str],
    date_prefix: bool = True,
    redact: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> tuple[Path, Path]:
    """Write ``report`` as JSON + markdown to ``out_dir``. Returns (json, md).

    ``date_prefix=False`` writes ``<run_tag>.json`` (fixed path for committed
    reference baselines) instead of ``<date>-<run_tag>.json`` (nightlies).
    ``redact`` transforms the report before the JSON is written (markdown always
    renders the full report). ``render_markdown`` receives the report and
    returns the markdown body (the trailing newline is added here).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if date_prefix:
        base = f"{datetime.now(UTC).date().isoformat()}-{run_tag}"
    else:
        base = run_tag
    json_path = out_dir / f"{base}.json"
    md_path = out_dir / f"{base}.md"
    payload = redact(report) if redact is not None else report
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    md_path.write_text(render_markdown(report) + "\n")
    return json_path, md_path


def compare_overall_to_baseline(
    cur_overall: dict[str, Any],
    base_overall: dict[str, Any],
    *,
    metric_keys: tuple[str, ...],
    gate_metric: str,
    threshold: float,
) -> dict[str, Any]:
    """Per-metric current/baseline/delta + a ``regressed`` verdict on ``gate_metric``.

    ``regressed`` is ``delta < -threshold`` for ``gate_metric`` (a drop past the
    threshold). Deltas are rounded to 4 decimals to match report precision.
    """
    if gate_metric not in metric_keys:
        raise ValueError(f"gate_metric {gate_metric!r} must be included in metric_keys {metric_keys!r}")
    metrics: dict[str, dict[str, float]] = {}
    for key in metric_keys:
        c = float(cur_overall.get(key, 0.0))
        b = float(base_overall.get(key, 0.0))
        metrics[key] = {"current": round(c, 4), "baseline": round(b, 4), "delta": round(c - b, 4)}
    delta = metrics[gate_metric]["delta"]
    return {
        "metrics": metrics,
        "gate_metric": gate_metric,
        "delta": round(delta, 4),
        "threshold": threshold,
        "regressed": delta < -threshold,
    }


def dataset_version(path: Path) -> int | str:
    """Best-effort dataset version from a JSON file's top-level ``version`` key.

    Returns ``"unknown"`` if the file is missing/unreadable or the version is not
    an int/str. Shared by every benchmark's ``Dataset`` construction so the logic
    lives in one place.
    """
    try:
        version = json.loads(path.read_text()).get("version", "unknown")
    except Exception:
        return "unknown"
    return version if isinstance(version, (int, str)) else "unknown"
