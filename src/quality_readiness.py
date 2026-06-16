"""No-secret quality maturity readiness for FluxMind eval targets."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT
from src.evaluation import (
    evaluate_quality_maturity_targets,
    load_eval_config,
    quality_metric_values,
)


QUALITY_READINESS_SCHEMA_VERSION = 1
LIVE_METRIC_KEYS = ("live_answer_result_count", "live_retrieval_result_count")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _format_bool(value: Any) -> str:
    return "true" if bool(value) else "false"


def _target_by_id(targets: list[dict[str, Any]], target_id: str) -> dict[str, Any]:
    for target in targets:
        if target.get("id") == target_id:
            return target
    return {"id": target_id, "ok": False, "status": "missing", "missing_metrics": ["target_missing"]}


def _summary_total(summary: dict[str, Any], key: str) -> int:
    try:
        return int((summary.get(key, {}) or {}).get("total", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _live_report_summary(path: Path) -> dict[str, Any]:
    """Read a no-secret eval JSON report and return only maturity evidence."""
    report: dict[str, Any] = {
        "name": path.name,
        "ok": False,
        "schema_version": None,
        "live_answer_result_count": 0,
        "live_retrieval_result_count": 0,
        "reason": "unreadable",
    }
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return report
    if not isinstance(data, dict):
        report["reason"] = "invalid_report_shape"
        return report

    metrics = (data.get("quality_maturity", {}) or {}).get("metrics", {}) or {}
    summary = data.get("summary", {}) or {}
    try:
        report["schema_version"] = data.get("schema_version")
        report["live_answer_result_count"] = max(
            int(metrics.get("live_answer_result_count", 0) or 0),
            _summary_total(summary, "live_answers"),
        )
        report["live_retrieval_result_count"] = max(
            int(metrics.get("live_retrieval_result_count", 0) or 0),
            _summary_total(summary, "live_retrieval"),
        )
    except (TypeError, ValueError):
        report["reason"] = "invalid_live_metric"
        return report

    report["ok"] = True
    report["reason"] = "ok"
    return report


def collect_quality_readiness(
    *,
    eval_file: Path | None = None,
    live_report_paths: list[Path] | None = None,
    generated_at: str | None = None,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Collect no-secret staged quality maturity readiness."""
    eval_path = eval_file or (project_root / "eval" / "rag_baseline.json")
    config = load_eval_config(eval_path)
    metrics = quality_metric_values(config, project_root=project_root)
    report_summaries = [
        _live_report_summary(path)
        for path in (live_report_paths or [])
    ]

    for report in report_summaries:
        if not report.get("ok"):
            continue
        for key in LIVE_METRIC_KEYS:
            metrics[key] = max(int(metrics.get(key, 0) or 0), int(report.get(key, 0) or 0))

    targets = evaluate_quality_maturity_targets(config, metrics)
    self_use = _target_by_id(targets, "self_use")
    small_group = _target_by_id(targets, "small_group")
    community = _target_by_id(targets, "community")
    live_reports_ok = all(report.get("ok") for report in report_summaries)

    maturity_blockers: list[str] = []
    for target in targets:
        target_id = str(target.get("id", "unknown"))
        for metric in target.get("missing_metrics", []):
            maturity_blockers.append(f"{target_id}_{metric}_gap")
    if not live_reports_ok:
        maturity_blockers.append("live_report_unreadable")

    local_foundation_ready = bool(self_use.get("ok")) and live_reports_ok
    return {
        "mode": "quality_readiness",
        "schema_version": QUALITY_READINESS_SCHEMA_VERSION,
        "generated_at": generated_at or _utc_now(),
        "eval_file": eval_path.name,
        "local_foundation_ready": local_foundation_ready,
        "small_group_ready": bool(small_group.get("ok")),
        "community_ready": bool(community.get("ok")),
        "live_evidence_included": bool(report_summaries),
        "content_exported": False,
        "secrets_exported": False,
        "paths_exported": False,
        "metrics": metrics,
        "targets": targets,
        "reports": report_summaries,
        "blockers": {
            "local_foundation": [] if local_foundation_ready else list(self_use.get("missing_metrics", [])),
            "maturity": maturity_blockers,
        },
        "notes": [
            "Quality readiness reuses eval/rag_baseline.json quality_maturity_targets.",
            "Live evidence is included only when explicit no-secret eval reports are supplied.",
        ],
    }


def format_quality_readiness_markdown(status: dict[str, Any]) -> str:
    """Render quality readiness as no-secret Markdown."""
    lines = [
        "# FluxMind Quality Readiness",
        "",
        f"- Mode: {status.get('mode', '')}",
        f"- Schema version: {status.get('schema_version', '')}",
        f"- Generated at: {status.get('generated_at', '')}",
        f"- Eval file: {status.get('eval_file', '')}",
        f"- Local foundation ready: {_format_bool(status.get('local_foundation_ready', False))}",
        f"- Small-group ready: {_format_bool(status.get('small_group_ready', False))}",
        f"- Community ready: {_format_bool(status.get('community_ready', False))}",
        f"- Live evidence included: {_format_bool(status.get('live_evidence_included', False))}",
        f"- Content exported: {_format_bool(status.get('content_exported', False))}",
        f"- Secrets exported: {_format_bool(status.get('secrets_exported', False))}",
        f"- Paths exported: {_format_bool(status.get('paths_exported', False))}",
        "",
        "## Metrics",
        "",
    ]
    for key, value in sorted((status.get("metrics", {}) or {}).items()):
        lines.append(f"- {key}: {value}")

    lines.extend(["", "## Targets", ""])
    for target in status.get("targets", []):
        missing = ", ".join(target.get("missing_metrics", [])) or "none"
        lines.append(
            f"- {target.get('id', '')}: {target.get('status', '')}; missing={missing}"
        )

    reports = status.get("reports", []) or []
    lines.extend(["", "## Live Reports", ""])
    if not reports:
        lines.append("- none")
    for report in reports:
        lines.append(
            f"- {report.get('name', '')}: ok={_format_bool(report.get('ok', False))}, "
            f"live_retrieval={report.get('live_retrieval_result_count', 0)}, "
            f"live_answers={report.get('live_answer_result_count', 0)}, "
            f"reason={report.get('reason', '')}"
        )

    blockers = status.get("blockers", {}) or {}
    lines.extend(
        [
            "",
            "## Blockers",
            "",
            f"- Local foundation: {', '.join(blockers.get('local_foundation', [])) or 'none'}",
            f"- Maturity: {', '.join(blockers.get('maturity', [])) or 'none'}",
        ]
    )
    return "\n".join(lines)
