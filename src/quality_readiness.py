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
LIVE_EVIDENCE_METRICS = {
    "live_answer_result_count",
    "live_answer_pass_rate",
    "live_retrieval_result_count",
    "live_retrieval_pass_rate",
    "average_live_answer_term_coverage",
}
CORPUS_EVIDENCE_METRICS = {"seed_paper_count"}
EVIDENCE_SOURCE_ORDER = ("corpus_manifest", "eval_baseline", "live_eval_report")


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


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
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


def _empty_live_report_summary(reason: str) -> dict[str, Any]:
    return {
        "name": "live_report",
        "ok": False,
        "schema_version": None,
        "source_name_exported": False,
        "live_answer_result_count": 0,
        "live_retrieval_result_count": 0,
        "live_answer_pass_rate": None,
        "live_retrieval_pass_rate": None,
        "average_live_answer_term_coverage": None,
        "reason": reason,
    }


def _live_report_summary_from_data(data: Any) -> dict[str, Any]:
    """Return only maturity evidence from a parsed no-secret eval JSON report."""
    report = _empty_live_report_summary("ok")
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
    return report


def _live_report_summary(path: Path) -> dict[str, Any]:
    """Read a no-secret eval JSON report and return only maturity evidence."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _empty_live_report_summary("unreadable")
    return _live_report_summary_from_data(data)


def _live_report_summaries_from_data(reports: list[Any] | None) -> list[dict[str, Any]]:
    return [_live_report_summary_from_data(report) for report in (reports or [])]


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


def _target_gap_summary(targets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return no-secret count and quality gaps for each maturity target."""
    summaries: list[dict[str, Any]] = []
    for target in targets:
        checks = [
            check
            for check in target.get("checks", [])
            if isinstance(check, dict)
        ]
        count_gaps = [
            {
                "metric": str(check.get("metric", "")),
                "actual": _safe_int(check.get("actual", 0)),
                "expected": _safe_int(check.get("expected", 0)),
                "gap": _safe_int(check.get("gap", 0)),
            }
            for check in checks
            if _safe_int(check.get("gap", 0)) > 0
        ]
        quality_gaps = [
            {
                "metric": str(check.get("metric", "")),
                "actual": check.get("actual"),
                "expected": str(check.get("expected", "")),
                "source_gate": str(check.get("source_gate", "")),
            }
            for check in target.get("quality_checks", [])
            if isinstance(check, dict) and not check.get("ok")
        ]
        count_gaps.sort(key=lambda item: (-item["gap"], item["metric"]))
        summaries.append(
            {
                "target": str(target.get("id", "")),
                "status": str(target.get("status", "")),
                "required_metric_count": len(checks),
                "met_metric_count": sum(1 for check in checks if check.get("ok")),
                "missing_metric_count": len(count_gaps),
                "missing_metrics": [item["metric"] for item in count_gaps],
                "count_gaps": count_gaps,
                "quality_gaps": quality_gaps,
            }
        )
    return summaries


def _evidence_source_for_metric(metric: str) -> str:
    if metric in LIVE_EVIDENCE_METRICS:
        return "live_eval_report"
    if metric in CORPUS_EVIDENCE_METRICS:
        return "corpus_manifest"
    return "eval_baseline"


def _evidence_requests(gap_summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return no-secret evidence deltas needed for each maturity target."""
    requests: list[dict[str, Any]] = []
    for summary in gap_summary:
        count_items = [
            {
                "kind": "count",
                "metric": str(gap.get("metric", "")),
                "actual": _safe_int(gap.get("actual")),
                "expected": _safe_int(gap.get("expected")),
                "gap": _safe_int(gap.get("gap")),
                "evidence_source": _evidence_source_for_metric(str(gap.get("metric", ""))),
            }
            for gap in summary.get("count_gaps", []) or []
            if isinstance(gap, dict) and _safe_int(gap.get("gap")) > 0
        ]
        quality_items = [
            {
                "kind": "quality",
                "metric": str(gap.get("metric", "")),
                "actual": gap.get("actual"),
                "expected": str(gap.get("expected", "")),
                "evidence_source": _evidence_source_for_metric(str(gap.get("metric", ""))),
                "source_gate": str(gap.get("source_gate", "")),
            }
            for gap in summary.get("quality_gaps", []) or []
            if isinstance(gap, dict)
        ]
        items = sorted(
            count_items,
            key=lambda item: (
                item["evidence_source"],
                -item["gap"],
                item["metric"],
            ),
        ) + sorted(
            quality_items,
            key=lambda item: (item["evidence_source"], item["metric"]),
        )
        target = str(summary.get("target", ""))
        requests.append(
            {
                "target": target,
                "status": str(summary.get("status", "")),
                "ready": not items,
                "item_count": len(items),
                "evidence_sources": sorted(
                    {item["evidence_source"] for item in items}
                ),
                "items": items,
            }
        )
    return requests


def _next_evidence_request(requests: list[dict[str, Any]]) -> dict[str, Any]:
    for request in requests:
        if not request.get("ready"):
            return request
    return {
        "target": "none",
        "status": "met",
        "ready": True,
        "item_count": 0,
        "evidence_sources": [],
        "items": [],
    }


def _evidence_request_by_target(
    requests: list[dict[str, Any]],
    target: str,
) -> dict[str, Any]:
    for request in requests:
        if request.get("target") == target:
            return request
    return {
        "target": target,
        "status": "missing",
        "ready": False,
        "item_count": 1,
        "evidence_sources": ["eval_baseline"],
        "items": [
            {
                "kind": "count",
                "metric": "target_missing",
                "actual": 0,
                "expected": 1,
                "gap": 1,
                "evidence_source": "eval_baseline",
            }
        ],
    }


def _dedupe_sorted(values: list[str]) -> list[str]:
    return sorted({value for value in values if value})


def _live_eval_command(metrics: list[str]) -> str:
    flags: list[str] = []
    if any(metric.startswith("live_retrieval") for metric in metrics):
        flags.append("--retrieval-url <api-base-url>")
    if any(
        metric.startswith("live_answer")
        or metric == "average_live_answer_term_coverage"
        for metric in metrics
    ):
        flags.append("--live-url <api-base-url>")
    if not flags:
        flags = ["--retrieval-url <api-base-url>", "--live-url <api-base-url>"]
    return (
        ".venv/bin/python scripts/evaluate_rag.py "
        + " ".join(flags)
        + " --api-key-env FLUXMIND_API_TOKEN --json-report <report.json>"
    )


def _evidence_source_action(source: str, metrics: list[str]) -> tuple[str, str]:
    if source == "corpus_manifest":
        return (
            "Add curated open-access corpus entries until the corpus count gaps close.",
            "manual: update the curated corpus manifest, then run .venv/bin/python scripts/evaluate_rag.py --json-report <report.json>",
        )
    if source == "eval_baseline":
        return (
            "Expand the eval baseline with traceable answer, retrieval, recorded-answer, or topic coverage fixtures until the eval gaps close.",
            "manual: update the eval baseline, then run .venv/bin/python scripts/evaluate_rag.py --json-report <report.json>",
        )
    if source == "live_eval_report":
        return (
            "Run live retrieval or live answer scoring against an explicit API base URL and feed the no-secret JSON report back into readiness.",
            _live_eval_command(metrics),
        )
    return (
        "Provide the missing no-secret evidence for this metric source.",
        "manual: collect evidence, then run .venv/bin/python scripts/quality_readiness.py --format markdown",
    )


def _evidence_collection_plan(request: dict[str, Any]) -> dict[str, Any]:
    """Turn a target evidence request into an operator-facing no-secret plan."""
    target = str(request.get("target", ""))
    items = [
        item
        for item in request.get("items", []) or []
        if isinstance(item, dict)
    ]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        source = str(item.get("evidence_source", "eval_baseline"))
        grouped.setdefault(source, []).append(item)

    source_order = list(EVIDENCE_SOURCE_ORDER) + sorted(
        source for source in grouped if source not in EVIDENCE_SOURCE_ORDER
    )
    steps: list[dict[str, Any]] = []
    for source in source_order:
        source_items = grouped.get(source, [])
        if not source_items:
            continue
        metrics = _dedupe_sorted([str(item.get("metric", "")) for item in source_items])
        action, command = _evidence_source_action(source, metrics)
        steps.append(
            {
                "evidence_source": source,
                "item_count": len(source_items),
                "metrics": metrics,
                "action": action,
                "command": command,
                "verification_command": (
                    ".venv/bin/python scripts/quality_readiness.py "
                    f"--live-report <report.json> --require-target {target} --format markdown"
                ),
                "content_exported": False,
                "secrets_exported": False,
                "paths_exported": False,
            }
        )

    return {
        "target": target,
        "status": str(request.get("status", "")),
        "ready": bool(request.get("ready")),
        "item_count": len(items),
        "source_count": len(steps),
        "steps": steps,
        "content_exported": False,
        "secrets_exported": False,
        "paths_exported": False,
        "notes": [
            "Commands use placeholders only; operators must supply an API base URL and an ignored local report path.",
            "The plan does not embed prompts, answers, source text, raw report payloads, API tokens, or local paths.",
        ],
    }


def collect_quality_readiness(
    *,
    eval_file: Path | None = None,
    live_report_paths: list[Path] | None = None,
    live_reports: list[Any] | None = None,
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
    ] + _live_report_summaries_from_data(live_reports)

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
    gap_summary = _target_gap_summary(targets)
    evidence_requests = _evidence_requests(gap_summary)
    next_evidence_request = _next_evidence_request(evidence_requests)
    community_evidence_request = _evidence_request_by_target(
        evidence_requests,
        "community",
    )
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
        "gap_summary": gap_summary,
        "evidence_requests": evidence_requests,
        "next_evidence_request": next_evidence_request,
        "next_evidence_plan": _evidence_collection_plan(next_evidence_request),
        "community_evidence_plan": _evidence_collection_plan(
            community_evidence_request
        ),
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

    lines.extend(["", "## Target Gap Summary", ""])
    for summary in status.get("gap_summary", []) or []:
        count_gaps = summary.get("count_gaps", []) or []
        quality_gaps = summary.get("quality_gaps", []) or []
        count_gap_text = "; ".join(
            f"{item.get('metric', '')} {item.get('actual', 0)}/"
            f"{item.get('expected', 0)} gap={item.get('gap', 0)}"
            for item in count_gaps
        ) or "none"
        quality_gap_text = "; ".join(
            f"{item.get('metric', '')} actual={_format_optional_float(item.get('actual'))} "
            f"expected={item.get('expected', '')}"
            for item in quality_gaps
        ) or "none"
        lines.append(
            f"- {summary.get('target', '')}: status={summary.get('status', '')}, "
            f"met={summary.get('met_metric_count', 0)}/"
            f"{summary.get('required_metric_count', 0)}, "
            f"count_gaps={count_gap_text}, quality_gaps={quality_gap_text}"
        )

    evidence_requests = status.get("evidence_requests", []) or []
    lines.extend(["", "## Evidence Requests", ""])
    for request in evidence_requests:
        lines.append(
            f"- {request.get('target', '')}: ready={_format_bool(request.get('ready'))}, "
            f"items={request.get('item_count', 0)}, "
            f"sources={','.join(request.get('evidence_sources', [])) or 'none'}"
        )
        for item in request.get("items", []) or []:
            if item.get("kind") == "quality":
                lines.append(
                    f"  - quality {item.get('metric', '')}: actual="
                    f"{_format_optional_float(item.get('actual'))} "
                    f"expected={item.get('expected', '')} "
                    f"source={item.get('evidence_source', '')}"
                )
            else:
                lines.append(
                    f"  - count {item.get('metric', '')}: actual={item.get('actual', 0)} "
                    f"expected={item.get('expected', 0)} gap={item.get('gap', 0)} "
                    f"source={item.get('evidence_source', '')}"
                )

    lines.extend(["", "## Evidence Collection Plan", ""])
    for label, plan_key in (
        ("Next target", "next_evidence_plan"),
        ("Community target", "community_evidence_plan"),
    ):
        plan = status.get(plan_key, {}) or {}
        lines.append(
            f"- {label}: target={plan.get('target', '')}, "
            f"ready={_format_bool(plan.get('ready', False))}, "
            f"sources={plan.get('source_count', 0)}, items={plan.get('item_count', 0)}"
        )
        for step in plan.get("steps", []) or []:
            lines.append(
                f"  - {step.get('evidence_source', '')}: "
                f"metrics={','.join(step.get('metrics', [])) or 'none'}; "
                f"command=`{step.get('command', '')}`; "
                f"verify=`{step.get('verification_command', '')}`"
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
