"""No-secret local activation suite for FluxMind."""

from __future__ import annotations

import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from src import config
from src.collaboration_readiness import collect_collaboration_readiness
from src.openapi_contract import collect_openapi_contract
from src.product_activation_rehearsal import collect_product_activation_rehearsal
from src.product_readiness import collect_product_readiness
from src.provider_runtime_rehearsal import collect_provider_runtime_rehearsal
from src.quality_readiness import collect_quality_readiness
from src.storage_migration import run_storage_migration_rehearsal


ACTIVATION_SUITE_SCHEMA_VERSION = 1
ACTIVATION_SUITE_STATE_DIR = ".fluxmind-activation-suite"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _format_bool(value: Any) -> str:
    return "true" if bool(value) else "false"


@contextmanager
def _suite_root(root: Path | None) -> Iterator[Path]:
    if root is not None:
        root.mkdir(parents=True, exist_ok=True)
        state_root = root / ACTIVATION_SUITE_STATE_DIR
        state_root.mkdir(parents=True, exist_ok=True)
        yield state_root
        return
    with tempfile.TemporaryDirectory(prefix="fluxmind-activation-suite-") as temp_root:
        yield Path(temp_root)


def _string_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return [str(value) for value in values if str(value)]


def _product_summary(status: dict[str, Any]) -> dict[str, Any]:
    readiness = status.get("readiness", {}) or {}
    api_keys = status.get("api_key_lifecycle", {}) or {}
    registry = status.get("product_registry", {}) or {}
    rbac = status.get("rbac", {}) or {}
    workspace_isolation = status.get("workspace_isolation", {}) or {}
    quota = status.get("quota", {}) or {}
    return {
        "ok": bool(status.get("ok")),
        "local_foundation_ready": bool(readiness.get("local_foundation_ready")),
        "activation_ready": bool(readiness.get("activation_ready")),
        "active_key_count": int(api_keys.get("active_key_count", 0) or 0),
        "revoked_key_count": int(api_keys.get("revoked_key_count", 0) or 0),
        "workspace_count": int(registry.get("workspace_count", 0) or 0),
        "quota_limit_count": int(registry.get("quota_limit_count", 0) or 0),
        "rbac_ok": bool(rbac.get("ok")),
        "workspace_isolation_ok": bool(workspace_isolation.get("ok")),
        "quota_ok": bool(quota.get("ok")),
        "activation_blockers": _string_list(readiness.get("activation_blockers")),
        "local_foundation_blockers": _string_list(readiness.get("local_foundation_blockers")),
    }


def _product_readiness_summary(status: dict[str, Any]) -> dict[str, Any]:
    blockers = status.get("blockers", {}) or {}
    summary = status.get("summary", {}) or {}
    return {
        "ok": bool(status.get("local_foundation_ready")),
        "local_foundation_ready": bool(status.get("local_foundation_ready")),
        "activation_ready": bool(status.get("activation_ready")),
        "identity_quotas_billing_enabled": bool(
            status.get("identity_quotas_billing_enabled")
        ),
        "api_key_lifecycle_available": bool(
            summary.get("api_key_lifecycle_available")
        ),
        "product_registry_available": bool(summary.get("product_registry_available")),
        "workspace_identity_available": bool(summary.get("workspace_identity_available")),
        "quota_store_available": bool(summary.get("quota_store_available")),
        "billing_ledger_available": bool(summary.get("billing_ledger_available")),
        "product_quota_guard_enabled": bool(summary.get("product_quota_guard_enabled")),
        "product_rbac_guard_enabled": bool(summary.get("product_rbac_guard_enabled")),
        "activation_blockers": _string_list(blockers.get("activation")),
        "local_foundation_blockers": _string_list(blockers.get("local_foundation")),
        "advisories": _string_list(status.get("advisories")),
    }


def _collaboration_summary(status: dict[str, Any]) -> dict[str, Any]:
    summary = status.get("summary", {}) or {}
    blockers = status.get("blockers", {}) or {}
    return {
        "ok": bool(status.get("ok")),
        "local_foundation_ready": bool(status.get("local_foundation_ready")),
        "safe_default_ready": bool(status.get("safe_default_ready")),
        "activation_ready": bool(status.get("activation_ready")),
        "private_corpora_enabled": bool(summary.get("private_corpora_enabled")),
        "share_links_enabled": bool(summary.get("share_links_enabled")),
        "product_registry_available": bool(summary.get("product_registry_available")),
        "product_rbac_guard_enabled": bool(summary.get("product_rbac_guard_enabled")),
        "share_link_token_store_available": bool(
            summary.get("share_link_token_store_available")
        ),
        "policy_scenario_count": int(summary.get("policy_scenario_count", 0) or 0),
        "policy_denied_count": int(summary.get("policy_denied_count", 0) or 0),
        "activation_blockers": _string_list(blockers.get("activation")),
        "local_foundation_blockers": _string_list(blockers.get("local_foundation")),
    }


def _openapi_contract_summary(status: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": bool(status.get("local_contract_ready")),
        "local_contract_ready": bool(status.get("local_contract_ready")),
        "route_count": int(status.get("route_count", 0) or 0),
        "operation_count": int(status.get("operation_count", 0) or 0),
        "required_operation_count": int(status.get("required_operation_count", 0) or 0),
        "required_operation_missing_count": int(
            status.get("required_operation_missing_count", 0) or 0
        ),
        "protected_operation_count": int(status.get("protected_operation_count", 0) or 0),
        "protected_auth_header_operation_count": int(
            status.get("protected_auth_header_operation_count", 0) or 0
        ),
        "undocumented_operation_count": int(
            status.get("undocumented_operation_count", 0) or 0
        ),
        "response_missing_operation_count": int(
            status.get("response_missing_operation_count", 0) or 0
        ),
        "blockers": _string_list(status.get("blockers")),
        "missing_required_operations": _string_list(
            status.get("missing_required_operations")
        ),
        "protected_operations_missing_auth_headers": _string_list(
            status.get("protected_operations_missing_auth_headers")
        ),
        "raw_schema_exported": bool(status.get("raw_schema_exported")),
        "content_exported": bool(status.get("content_exported")),
        "secrets_exported": bool(status.get("secrets_exported")),
        "paths_exported": bool(status.get("paths_exported")),
    }


def _provider_summary(status: dict[str, Any]) -> dict[str, Any]:
    readiness = status.get("readiness", {}) or {}
    image = status.get("image_provider", {}) or {}
    python = status.get("python_execution", {}) or {}
    octave = status.get("octave_execution", {}) or {}
    quota = status.get("provider_quota_guard", {}) or {}
    docker = status.get("docker_execution", {}) or {}
    return {
        "ok": bool(status.get("ok")),
        "local_foundation_ready": bool(readiness.get("local_foundation_ready")),
        "external_activation_ready": bool(status.get("external_activation_ready")),
        "provider_activation_ready": bool(readiness.get("activation_ready")),
        "image_provider_ok": bool(image.get("ok")),
        "python_execution_ok": bool(python.get("ok")),
        "octave_execution_ok": bool(octave.get("ok")),
        "octave_runtime_available": bool(octave.get("runtime_available")),
        "docker_available": bool(docker.get("available")),
        "provider_quota_guard_ok": bool(quota.get("ok")),
        "activation_blockers": _string_list(readiness.get("activation_blockers")),
        "local_foundation_blockers": _string_list(readiness.get("local_foundation_blockers")),
    }


def _storage_summary(status: dict[str, Any]) -> dict[str, Any]:
    summary = status.get("summary", {}) or {}
    source = status.get("source_preflight", {}) or {}
    return {
        "ok": bool(status.get("rehearsal_ok")),
        "source_preflight_ok": bool(summary.get("source_preflight_ok")),
        "source_activation_ready": bool(summary.get("source_activation_ready")),
        "restore_check_ok": bool(summary.get("restore_check_ok")),
        "staged_storage_schema_ok": bool(summary.get("staged_storage_schema_ok")),
        "copy_group_count": int(summary.get("copy_group_count", 0) or 0),
        "copied_files": int(summary.get("copied_files", 0) or 0),
        "skipped_symlinks": int(summary.get("skipped_symlinks", 0) or 0),
        "object_manifest_ready": bool(summary.get("object_manifest_ready")),
        "job_store_manifest_ready": bool(summary.get("job_store_manifest_ready")),
        "job_store_manifest_jobs": int(summary.get("job_store_manifest_jobs", 0) or 0),
        "job_store_manifest_claims": int(summary.get("job_store_manifest_claims", 0) or 0),
        "blockers": _string_list(status.get("blockers")),
        "activation_blockers": _string_list(source.get("activation_blockers")),
        "local_foundation_blockers": _string_list(source.get("local_blockers")),
    }


def _safe_metric_name(value: Any) -> str:
    return str(value or "")


def _safe_gap_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _next_quality_evidence(status: dict[str, Any]) -> dict[str, Any]:
    """Summarize the next no-secret quality evidence target for operators."""
    request = status.get("next_evidence_request")
    if isinstance(request, dict):
        gaps = [
            {
                "kind": str(item.get("kind", "")),
                "metric": _safe_metric_name(item.get("metric")),
                "actual": item.get("actual"),
                "expected": item.get("expected"),
                "gap": _safe_gap_int(item.get("gap")),
                "evidence_source": str(item.get("evidence_source", "")),
            }
            for item in request.get("items", []) or []
            if isinstance(item, dict)
        ]
        return {
            "target": str(request.get("target", "none")),
            "gap_count": len(gaps),
            "count_gap_count": sum(1 for item in gaps if item.get("kind") == "count"),
            "quality_gap_count": sum(1 for item in gaps if item.get("kind") == "quality"),
            "evidence_sources": list(request.get("evidence_sources", []) or []),
            "gaps": gaps,
        }

    summaries = [
        item
        for item in status.get("gap_summary", []) or []
        if isinstance(item, dict)
    ]
    by_target = {str(item.get("target", "")): item for item in summaries}
    if not status.get("small_group_ready"):
        target = "small_group"
    elif not status.get("community_ready"):
        target = "community"
    else:
        target = "none"

    selected = by_target.get(target, {}) if target != "none" else {}
    count_gaps = [
        {
            "kind": "count",
            "metric": _safe_metric_name(gap.get("metric")),
            "actual": _safe_gap_int(gap.get("actual")),
            "expected": _safe_gap_int(gap.get("expected")),
            "gap": _safe_gap_int(gap.get("gap")),
        }
        for gap in selected.get("count_gaps", []) or []
        if isinstance(gap, dict) and _safe_gap_int(gap.get("gap")) > 0
    ]
    quality_gaps = [
        {
            "kind": "quality",
            "metric": _safe_metric_name(gap.get("metric")),
            "actual": gap.get("actual"),
            "expected": str(gap.get("expected", "")),
            "source_gate": str(gap.get("source_gate", "")),
        }
        for gap in selected.get("quality_gaps", []) or []
        if isinstance(gap, dict)
    ]
    gaps = sorted(count_gaps, key=lambda item: (-item["gap"], item["metric"])) + sorted(
        quality_gaps,
        key=lambda item: item["metric"],
    )
    return {
        "target": target,
        "gap_count": len(gaps),
        "count_gap_count": len(count_gaps),
        "quality_gap_count": len(quality_gaps),
        "evidence_sources": [],
        "gaps": gaps,
    }


def _quality_summary(status: dict[str, Any]) -> dict[str, Any]:
    blockers = status.get("blockers", {}) or {}
    next_evidence = _next_quality_evidence(status)
    community_evidence_plan = status.get("community_evidence_plan", {}) or {}
    return {
        "ok": bool(status.get("local_foundation_ready")),
        "local_foundation_ready": bool(status.get("local_foundation_ready")),
        "small_group_ready": bool(status.get("small_group_ready")),
        "community_ready": bool(status.get("community_ready")),
        "live_evidence_included": bool(status.get("live_evidence_included")),
        "target_gap_count": sum(
            int(item.get("missing_metric_count", 0) or 0)
            for item in status.get("gap_summary", []) or []
            if isinstance(item, dict)
        ),
        "next_evidence": next_evidence,
        "next_evidence_target": next_evidence["target"],
        "next_evidence_gap_count": next_evidence["gap_count"],
        "community_evidence_plan": _quality_plan_summary(community_evidence_plan),
        "local_foundation_blockers": _string_list(blockers.get("local_foundation")),
        "maturity_blockers": _string_list(blockers.get("maturity")),
    }


def _quality_plan_summary(plan: dict[str, Any]) -> dict[str, Any]:
    """Project the quality evidence plan without raw report payloads."""
    steps = [
        {
            "evidence_source": str(step.get("evidence_source", "")),
            "metrics": _string_list(step.get("metrics")),
            "command": str(step.get("command", "")),
            "verification_command": str(step.get("verification_command", "")),
            "content_exported": False,
            "secrets_exported": False,
            "paths_exported": False,
        }
        for step in plan.get("steps", []) or []
        if isinstance(step, dict)
    ]
    return {
        "target": str(plan.get("target", "")),
        "ready": bool(plan.get("ready")),
        "item_count": int(plan.get("item_count", 0) or 0),
        "source_count": len(steps),
        "steps": steps,
        "content_exported": False,
        "secrets_exported": False,
        "paths_exported": False,
    }


def _failed_check_names(checks: dict[str, dict[str, Any]]) -> list[str]:
    return sorted(name for name, check in checks.items() if not check.get("ok"))


def _full_activation_blockers(checks: dict[str, dict[str, Any]]) -> list[str]:
    blockers: list[str] = []
    if not checks["product_readiness"].get("activation_ready"):
        blockers.append("product_readiness_activation_not_ready")
    if not checks["product_activation"].get("activation_ready"):
        blockers.append("product_activation_not_ready")
    if not checks["collaboration_readiness"].get("activation_ready"):
        blockers.append("collaboration_activation_not_ready")
    if not checks["provider_runtime"].get("provider_activation_ready"):
        blockers.append("provider_activation_not_ready")
    if not checks["storage_migration"].get("source_activation_ready"):
        blockers.append("platform_migration_activation_not_ready")
    if not checks["quality_readiness"].get("community_ready"):
        blockers.append("quality_community_not_ready")
    return sorted(set(blockers))


def _activation_step(
    *,
    area: str,
    ready: bool,
    blockers: list[str],
    action: str,
    command: str,
    verification_command: str,
    substeps: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "area": area,
        "ready": ready,
        "blockers": blockers,
        "action": action,
        "command": command,
        "verification_command": verification_command,
        "substeps": substeps or [],
        "content_exported": False,
        "secrets_exported": False,
        "paths_exported": False,
    }


def _activation_action_plan(
    checks: dict[str, dict[str, Any]],
    *,
    full_activation_ready: bool,
) -> dict[str, Any]:
    """Return the no-secret operator handoff for remaining activation gates."""
    steps: list[dict[str, Any]] = []

    product_readiness = checks["product_readiness"]
    if not product_readiness.get("activation_ready"):
        steps.append(
            _activation_step(
                area="product_readiness",
                ready=False,
                blockers=_string_list(product_readiness.get("activation_blockers")),
                action="Configure and enable the actual identity, API-key registry, quota store, quota/RBAC guards, and billing-attribution readiness targets before claiming full product activation.",
                command=".venv/bin/python scripts/product_readiness.py --format markdown",
                verification_command=".venv/bin/python scripts/product_readiness.py --format markdown --require-activation",
            )
        )

    product = checks["product_activation"]
    if not product.get("activation_ready"):
        steps.append(
            _activation_step(
                area="product_activation",
                ready=False,
                blockers=_string_list(product.get("activation_blockers")),
                action="Enable and verify the local product registry/API-key/RBAC/quota/billing-attribution foundation before external identity or billing work.",
                command=".venv/bin/python scripts/product_activation_rehearsal.py --format markdown --require-activation",
                verification_command=".venv/bin/python scripts/product_readiness.py --format markdown --require-activation",
            )
        )

    collaboration = checks["collaboration_readiness"]
    if not collaboration.get("activation_ready"):
        steps.append(
            _activation_step(
                area="collaboration_readiness",
                ready=False,
                blockers=_string_list(collaboration.get("activation_blockers")),
                action="Enable private-corpus and share-link collaboration only after the product registry, RBAC guard, and share-link token-store gates are configured.",
                command=".venv/bin/python scripts/collaboration_readiness.py --format markdown",
                verification_command=".venv/bin/python scripts/collaboration_readiness.py --format markdown --require-activation",
            )
        )

    openapi_contract = checks.get("openapi_contract")
    if openapi_contract and not openapi_contract.get("ok"):
        steps.append(
            _activation_step(
                area="openapi_contract",
                ready=False,
                blockers=_string_list(openapi_contract.get("blockers")),
                action="Restore required API route/method coverage, operation metadata, responses, and protected header declarations before claiming local foundation readiness.",
                command=".venv/bin/python scripts/openapi_contract.py --format markdown --require-local-contract",
                verification_command=".venv/bin/python scripts/activation_suite.py --format markdown --require-target local_foundation",
            )
        )

    provider = checks["provider_runtime"]
    if not provider.get("provider_activation_ready"):
        steps.append(
            _activation_step(
                area="provider_activation",
                ready=False,
                blockers=_string_list(provider.get("activation_blockers")),
                action="Configure real external image/provider execution/MATLAB/provider quota settings only after the local provider rehearsal stays green.",
                command=".venv/bin/python scripts/provider_runtime_rehearsal.py --format markdown --require-local-foundation",
                verification_command=".venv/bin/python scripts/provider_readiness.py --format markdown --require-activation",
            )
        )

    storage = checks["storage_migration"]
    if not storage.get("source_activation_ready"):
        steps.append(
            _activation_step(
                area="platform_migration",
                ready=False,
                blockers=_string_list(storage.get("activation_blockers")),
                action="Choose and configure production metadata database, object storage, and distributed job-store targets, then rerun the no-secret migration preflight.",
                command=".venv/bin/python scripts/platform_migration_rehearsal.py --include-object-manifest --include-job-store-manifest --format markdown",
                verification_command=".venv/bin/python scripts/platform_migration_preflight.py --format markdown --require-activation",
            )
        )

    quality = checks["quality_readiness"]
    if not quality.get("community_ready"):
        community_plan = quality.get("community_evidence_plan", {}) or {}
        steps.append(
            _activation_step(
                area="community_quality",
                ready=False,
                blockers=_string_list(quality.get("maturity_blockers")),
                action="Close the community corpus/eval/live-answer evidence gaps before public community release claims.",
                command=".venv/bin/python scripts/quality_readiness.py --format markdown",
                verification_command=".venv/bin/python scripts/quality_readiness.py --live-report <report.json> --require-target community --format markdown",
                substeps=[
                    {
                        "evidence_source": str(step.get("evidence_source", "")),
                        "metrics": _string_list(step.get("metrics")),
                        "command": str(step.get("command", "")),
                        "verification_command": str(step.get("verification_command", "")),
                    }
                    for step in community_plan.get("steps", []) or []
                    if isinstance(step, dict)
                ],
            )
        )

    return {
        "target": "full_activation",
        "ready": full_activation_ready,
        "step_count": len(steps),
        "steps": steps,
        "content_exported": False,
        "secrets_exported": False,
        "paths_exported": False,
        "connectivity_checked": False,
        "notes": [
            "Commands are local or placeholder-based operator handoffs.",
            "The plan does not activate external providers, object stores, distributed queues, identity providers, or billing systems by itself.",
        ],
    }


def collect_activation_suite(
    *,
    root: Path | None = None,
    project_root: Path = config.PROJECT_ROOT,
    eval_file: Path | None = None,
    live_report_paths: list[Path] | None = None,
    live_reports: list[dict[str, Any]] | None = None,
    openapi_schema: dict[str, Any] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Run the local no-secret activation suite.

    The suite composes the existing product, provider, migration, and quality
    checks into one operator-facing report. It keeps detailed reports out of the
    top-level payload so no local paths, raw tokens, prompts, job payloads, or
    artifact URIs are exported through this aggregate surface.
    """
    timestamp = generated_at or _utc_now()
    with _suite_root(root) as state_root:
        product = collect_product_activation_rehearsal(
            root=state_root / "product",
            generated_at=timestamp,
        )
        product_readiness = collect_product_readiness(generated_at=timestamp)
        collaboration = collect_collaboration_readiness(generated_at=timestamp)
        provider = collect_provider_runtime_rehearsal(
            root=state_root / "provider",
            generated_at=timestamp,
        )
        storage = run_storage_migration_rehearsal(
            project_root=project_root,
            staging_root=state_root / "migration-staging",
            overwrite_staging=True,
            include_runtime_dependencies=False,
            include_object_manifest=False,
            include_job_store_manifest=True,
            generated_at=timestamp,
        )
        quality = collect_quality_readiness(
            eval_file=eval_file,
            live_report_paths=live_report_paths or [],
            live_reports=live_reports or [],
            generated_at=timestamp,
            project_root=project_root,
        )

    checks = {
        "product_readiness": _product_readiness_summary(product_readiness),
        "product_activation": _product_summary(product),
        "collaboration_readiness": _collaboration_summary(collaboration),
        "provider_runtime": _provider_summary(provider),
        "storage_migration": _storage_summary(storage),
        "quality_readiness": _quality_summary(quality),
    }
    if openapi_schema is not None:
        openapi_contract = collect_openapi_contract(
            openapi_schema,
            generated_at=timestamp,
        )
        checks["openapi_contract"] = _openapi_contract_summary(openapi_contract)

    local_foundation_blockers = _failed_check_names(checks)
    full_activation_blockers = _full_activation_blockers(checks)
    local_foundation_ready = not local_foundation_blockers
    full_activation_ready = local_foundation_ready and not full_activation_blockers
    small_group_ready = local_foundation_ready and checks["quality_readiness"]["small_group_ready"]
    community_ready = local_foundation_ready and checks["quality_readiness"]["community_ready"]
    activation_action_plan = _activation_action_plan(
        checks,
        full_activation_ready=full_activation_ready,
    )
    local_foundation_gates = [
        "product_readiness.ok",
        "product_activation.ok",
        "collaboration_readiness.ok",
        "provider_runtime.ok",
        "storage_migration.ok",
        "quality_readiness.local_foundation_ready",
    ]
    if "openapi_contract" in checks:
        local_foundation_gates.append("openapi_contract.ok")

    return {
        "mode": "activation_suite",
        "schema_version": ACTIVATION_SUITE_SCHEMA_VERSION,
        "generated_at": timestamp,
        "ok": local_foundation_ready,
        "local_foundation_ready": local_foundation_ready,
        "small_group_ready": small_group_ready,
        "community_ready": community_ready,
        "full_activation_ready": full_activation_ready,
        "check_count": len(checks),
        "failed_check_count": len(local_foundation_blockers),
        "full_activation_blocker_count": len(full_activation_blockers),
        "activation_step_count": int(activation_action_plan.get("step_count", 0) or 0),
        "local_only": True,
        "raw_reports_included": False,
        "content_exported": False,
        "secrets_exported": False,
        "paths_exported": False,
        "connectivity_checked": False,
        "checks": checks,
        "activation_action_plan": activation_action_plan,
        "blockers": {
            "local_foundation": local_foundation_blockers,
            "full_activation": full_activation_blockers,
        },
        "required_gates": {
            "default": "local_foundation",
            "local_foundation": local_foundation_gates,
            "full_activation": [
                "product_readiness.activation_ready",
                "product_activation.activation_ready",
                "collaboration_readiness.activation_ready",
                "provider_runtime.provider_activation_ready",
                "storage_migration.source_activation_ready",
                "quality_readiness.community_ready",
            ],
        },
        "notes": [
            "The suite aggregates no-secret summaries from existing local checks.",
            "OpenAPI contract readiness is included when a generated schema is supplied by CLI/API/UI.",
            "External provider, hosted execution, MATLAB, and community readiness remain explicit activation gates.",
        ],
    }


def _check_line(name: str, check: dict[str, Any]) -> str:
    extra: list[str] = []
    if "activation_ready" in check:
        extra.append(f"activation_ready={_format_bool(check.get('activation_ready'))}")
    if "local_contract_ready" in check:
        extra.append(f"local_contract_ready={_format_bool(check.get('local_contract_ready'))}")
    if "required_operation_missing_count" in check:
        extra.append(
            "missing_required_operations="
            f"{int(check.get('required_operation_missing_count', 0) or 0)}"
        )
    if "provider_activation_ready" in check:
        extra.append(
            f"provider_activation_ready={_format_bool(check.get('provider_activation_ready'))}"
        )
    if "source_activation_ready" in check:
        extra.append(f"source_activation_ready={_format_bool(check.get('source_activation_ready'))}")
    if "small_group_ready" in check:
        extra.append(f"small_group_ready={_format_bool(check.get('small_group_ready'))}")
        extra.append(f"community_ready={_format_bool(check.get('community_ready'))}")
    if check.get("next_evidence_target"):
        extra.append(f"next_evidence_target={check.get('next_evidence_target')}")
        extra.append(f"next_evidence_gap_count={int(check.get('next_evidence_gap_count', 0) or 0)}")
    blockers = (
        check.get("blockers")
        or check.get("local_foundation_blockers")
        or check.get("activation_blockers")
        or check.get("maturity_blockers")
        or []
    )
    extra.append(f"blockers={','.join(blockers) or 'none'}")
    return f"- {name}: ok={_format_bool(check.get('ok'))}; " + "; ".join(extra)


def format_activation_suite_markdown(status: dict[str, Any]) -> str:
    """Render the activation suite as no-secret Markdown."""
    lines = [
        "# FluxMind Activation Suite",
        "",
        f"- Mode: {status.get('mode', '')}",
        f"- Schema version: {status.get('schema_version', '')}",
        f"- Generated at: {status.get('generated_at', '')}",
        f"- OK: {_format_bool(status.get('ok', False))}",
        f"- Local foundation ready: {_format_bool(status.get('local_foundation_ready', False))}",
        f"- Small-group ready: {_format_bool(status.get('small_group_ready', False))}",
        f"- Community ready: {_format_bool(status.get('community_ready', False))}",
        f"- Full activation ready: {_format_bool(status.get('full_activation_ready', False))}",
        f"- Checks: {int(status.get('check_count', 0) or 0)}",
        f"- Failed checks: {int(status.get('failed_check_count', 0) or 0)}",
        f"- Full activation blockers: {int(status.get('full_activation_blocker_count', 0) or 0)}",
        f"- Activation steps: {int(status.get('activation_step_count', 0) or 0)}",
        f"- Local only: {_format_bool(status.get('local_only', False))}",
        f"- Raw reports included: {_format_bool(status.get('raw_reports_included', False))}",
        f"- Content exported: {_format_bool(status.get('content_exported', False))}",
        f"- Secrets exported: {_format_bool(status.get('secrets_exported', False))}",
        f"- Paths exported: {_format_bool(status.get('paths_exported', False))}",
        f"- Connectivity checked: {_format_bool(status.get('connectivity_checked', False))}",
        "",
        "## Checks",
        "",
    ]
    checks = status.get("checks", {}) or {}
    check_order = [
        "product_readiness",
        "product_activation",
        "collaboration_readiness",
        "openapi_contract",
        "provider_runtime",
        "storage_migration",
        "quality_readiness",
    ]
    for name in check_order:
        if name not in checks:
            continue
        lines.append(_check_line(name, checks.get(name, {}) or {}))

    quality_evidence = (checks.get("quality_readiness", {}) or {}).get("next_evidence", {}) or {}
    evidence_gaps = quality_evidence.get("gaps", []) or []
    lines.extend(["", "## Next Quality Evidence", ""])
    lines.append(f"- Target: {quality_evidence.get('target', 'none')}")
    lines.append(
        f"- Sources: {', '.join(quality_evidence.get('evidence_sources', [])) or 'none'}"
    )
    if not evidence_gaps:
        lines.append("- Gaps: none")
    for gap in evidence_gaps:
        if gap.get("kind") == "quality":
            lines.append(
                f"- quality {gap.get('metric', '')}: actual={gap.get('actual')} "
                f"expected={gap.get('expected', '')} "
                f"source={gap.get('evidence_source', '')}"
            )
        else:
            lines.append(
                f"- count {gap.get('metric', '')}: actual={gap.get('actual', 0)} "
                f"expected={gap.get('expected', 0)} gap={gap.get('gap', 0)} "
                f"source={gap.get('evidence_source', '')}"
            )

    action_plan = status.get("activation_action_plan", {}) or {}
    lines.extend(["", "## Activation Action Plan", ""])
    lines.append(
        f"- Target: {action_plan.get('target', '')}; "
        f"ready={_format_bool(action_plan.get('ready', False))}; "
        f"steps={int(action_plan.get('step_count', 0) or 0)}"
    )
    for step in action_plan.get("steps", []) or []:
        lines.append(
            f"- {step.get('area', '')}: ready={_format_bool(step.get('ready', False))}; "
            f"blockers={','.join(step.get('blockers', [])) or 'none'}; "
            f"command=`{step.get('command', '')}`; "
            f"verify=`{step.get('verification_command', '')}`"
        )
        for substep in step.get("substeps", []) or []:
            lines.append(
                f"  - {substep.get('evidence_source', '')}: "
                f"metrics={','.join(substep.get('metrics', [])) or 'none'}; "
                f"command=`{substep.get('command', '')}`"
            )

    blockers = status.get("blockers", {}) or {}
    lines.extend(
        [
            "",
            "## Blockers",
            "",
            f"- Local foundation: {', '.join(blockers.get('local_foundation', [])) or 'none'}",
            f"- Full activation: {', '.join(blockers.get('full_activation', [])) or 'none'}",
        ]
    )
    return "\n".join(lines)
