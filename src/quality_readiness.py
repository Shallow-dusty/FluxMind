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
LIVE_QUALITY_RULES = (
    (
        "live_retrieval_result_count",
        "live_retrieval_pass_rate",
        "minimum_live_retrieval_pass_rate",
    ),
    (
        "live_answer_result_count",
        "live_answer_pass_rate",
        "minimum_live_answer_pass_rate",
    ),
    (
        "live_answer_result_count",
        "average_live_answer_term_coverage",
        "minimum_average_live_answer_term_coverage",
    ),
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _format_bool(value: Any) -> str:
    return "true" if bool(value) else "false"


def _format_optional_float(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "n/a"


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


def _summary_pass_rate(summary: dict[str, Any], key: str) -> float | None:
    try:
        values = summary.get(key, {}) or {}
        total = int(values.get("total", 0) or 0)
        ok = int(values.get("ok", 0) or 0)
    except (TypeError, ValueError):
        return None
    if total <= 0:
        return None
    return ok / total


def _result_items(data: dict[str, Any], key: str) -> list[dict[str, Any]]:
    raw = (data.get("results", {}) or {}).get(key, [])
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]


def _result_pass_rate(items: list[dict[str, Any]]) -> float | None:
    if not items:
        return None
    ok = sum(1 for item in items if item.get("ok"))
    return ok / len(items)


def _average_result_metric(items: list[dict[str, Any]], metric: str) -> float | None:
    values: list[float] = []
    for item in items:
        try:
            values.append(float(item[metric]))
        except (KeyError, TypeError, ValueError):
            continue
    if not values:
        return None
    return sum(values) / len(values)


def _live_report_summary(path: Path) -> dict[str, Any]:
    """Read a no-secret eval JSON report and return only maturity evidence."""
    report: dict[str, Any] = {
        "name": path.name,
        "ok": False,
        "schema_version": None,
        "live_answer_result_count": 0,
        "live_retrieval_result_count": 0,
        "live_answer_pass_rate": None,
        "live_retrieval_pass_rate": None,
        "average_live_answer_term_coverage": None,
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
    live_answer_items = _result_items(data, "live_answers")
    live_retrieval_items = _result_items(data, "live_retrieval")
    try:
        report["schema_version"] = data.get("schema_version")
        report["live_answer_result_count"] = max(
            int(metrics.get("live_answer_result_count", 0) or 0),
            _summary_total(summary, "live_answers"),
            len(live_answer_items),
        )
        report["live_retrieval_result_count"] = max(
            int(metrics.get("live_retrieval_result_count", 0) or 0),
            _summary_total(summary, "live_retrieval"),
            len(live_retrieval_items),
        )
    except (TypeError, ValueError):
        report["reason"] = "invalid_live_metric"
        return report

    answer_pass_rate = _summary_pass_rate(summary, "live_answers")
    retrieval_pass_rate = _summary_pass_rate(summary, "live_retrieval")
    report["live_answer_pass_rate"] = (
        answer_pass_rate
        if answer_pass_rate is not None
        else _result_pass_rate(live_answer_items)
    )
    report["live_retrieval_pass_rate"] = (
        retrieval_pass_rate
        if retrieval_pass_rate is not None
        else _result_pass_rate(live_retrieval_items)
    )
    report["average_live_answer_term_coverage"] = _average_result_metric(
        live_answer_items,
        "answer_term_coverage",
    )
    report["ok"] = True
    report["reason"] = "ok"
    return report


def _best_report_by_count(
    reports: list[dict[str, Any]],
    count_key: str,
) -> dict[str, Any] | None:
    candidates = [
        report
        for report in reports
        if report.get("ok") and int(report.get(count_key, 0) or 0) > 0
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda report: int(report.get(count_key, 0) or 0))


def _target_required_metrics(config: dict[str, Any], target_id: str) -> dict[str, int]:
    for target in config.get("quality_maturity_targets", []):
        if str(target.get("id", "")) != target_id:
            continue
        raw = target.get("required_metrics", {})
        if not isinstance(raw, dict):
            return {}
        required: dict[str, int] = {}
        for key, value in raw.items():
            try:
                required[str(key)] = int(value)
            except (TypeError, ValueError):
                continue
        return required
    return {}


def _number_metric(metrics: dict[str, Any], key: str) -> float | None:
    if key not in metrics:
        return None
    try:
        return float(metrics[key])
    except (TypeError, ValueError):
        return None


def _enrich_targets_with_live_quality(
    config: dict[str, Any],
    targets: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    gates = config.get("quality_gates", {})
    if not isinstance(gates, dict):
        gates = {}

    enriched: list[dict[str, Any]] = []
    for target in targets:
        target_id = str(target.get("id", ""))
        required = _target_required_metrics(config, target_id)
        quality_checks: list[dict[str, Any]] = []
        missing_metrics = list(target.get("missing_metrics", []))

        for count_metric, quality_metric, gate_id in LIVE_QUALITY_RULES:
            required_count = required.get(count_metric, 0)
            if required_count <= 0 or gate_id not in gates:
                continue
            actual_count = int(metrics.get(count_metric, 0) or 0)
            if actual_count < required_count:
                continue

            try:
                minimum = float(gates[gate_id])
            except (TypeError, ValueError):
                continue
            actual = _number_metric(metrics, quality_metric)
            ok = actual is not None and actual >= minimum
            if not ok and quality_metric not in missing_metrics:
                missing_metrics.append(quality_metric)
            quality_checks.append(
                {
                    "metric": quality_metric,
                    "actual": None if actual is None else round(actual, 4),
                    "expected": f">={minimum:.2f}",
                    "ok": ok,
                    "source_gate": gate_id,
                }
            )

        updated = dict(target)
        updated["missing_metrics"] = missing_metrics
        if quality_checks:
            updated["quality_checks"] = quality_checks
        if missing_metrics:
            updated["ok"] = False
            updated["status"] = "gap"
        enriched.append(updated)
    return enriched


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
    best_answer_report = _best_report_by_count(report_summaries, "live_answer_result_count")
    if best_answer_report:
        for key in ("live_answer_pass_rate", "average_live_answer_term_coverage"):
            if best_answer_report.get(key) is not None:
                metrics[key] = float(best_answer_report[key])
    best_retrieval_report = _best_report_by_count(report_summaries, "live_retrieval_result_count")
    if best_retrieval_report and best_retrieval_report.get("live_retrieval_pass_rate") is not None:
        metrics["live_retrieval_pass_rate"] = float(best_retrieval_report["live_retrieval_pass_rate"])

    targets = _enrich_targets_with_live_quality(
        config,
        evaluate_quality_maturity_targets(config, metrics),
        metrics,
    )
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
            f"live_retrieval_pass_rate={_format_optional_float(report.get('live_retrieval_pass_rate'))}, "
            f"live_answer_pass_rate={_format_optional_float(report.get('live_answer_pass_rate'))}, "
            f"live_answer_term_coverage={_format_optional_float(report.get('average_live_answer_term_coverage'))}, "
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
