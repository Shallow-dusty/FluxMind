"""No-secret local product activation rehearsal for FluxMind."""

from __future__ import annotations

import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from src.api_keys import LocalApiKeyRegistry
from src.product_readiness import collect_product_readiness
from src.product_registry import LocalProductRegistry


PRODUCT_ACTIVATION_REHEARSAL_SCHEMA_VERSION = 1
PRODUCT_ACTIVATION_REHEARSAL_STATE_DIR = ".fluxmind-product-activation-rehearsal"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _format_bool(value: Any) -> str:
    return "true" if bool(value) else "false"


@contextmanager
def _rehearsal_root(root: Path | None) -> Iterator[Path]:
    if root is not None:
        root.mkdir(parents=True, exist_ok=True)
        state_root = root / PRODUCT_ACTIVATION_REHEARSAL_STATE_DIR
        state_root.mkdir(parents=True, exist_ok=True)
        yield state_root
        return
    with tempfile.TemporaryDirectory(prefix="fluxmind-product-rehearsal-") as temp_root:
        yield Path(temp_root)


def _reset_rehearsal_sqlite(db_path: Path) -> None:
    for candidate in (db_path, Path(f"{db_path}-wal"), Path(f"{db_path}-shm")):
        if candidate.exists():
            candidate.unlink()


def collect_product_activation_rehearsal(
    *,
    root: Path | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Run a local SQLite-only product activation rehearsal.

    The rehearsal creates disposable local API-key and product registry stores,
    exercises lifecycle/RBAC/quota/billing contracts, and summarizes only
    no-secret booleans/counts/reason codes. Raw tokens, file paths, prompts,
    answers, and external connectivity are never returned.
    """
    with _rehearsal_root(root) as rehearsal_root:
        api_key_db = rehearsal_root / "api_keys.sqlite3"
        product_db = rehearsal_root / "product_registry.sqlite3"
        _reset_rehearsal_sqlite(api_key_db)
        _reset_rehearsal_sqlite(product_db)
        api_keys = LocalApiKeyRegistry(api_key_db)
        product_registry = LocalProductRegistry(product_db)

        owner_key = api_keys.create_key(
            owner_id="rehearsal-owner",
            owner_label="Rehearsal owner",
            description="local product activation rehearsal",
        )
        viewer_key = api_keys.create_key(
            owner_id="rehearsal-viewer",
            owner_label="Rehearsal viewer",
            description="local product activation rehearsal",
        )
        revoked_key = api_keys.create_key(
            owner_id="rehearsal-revoked",
            owner_label="Rehearsal revoked",
            description="local product activation rehearsal",
        )
        owner_verified = api_keys.verify_token(owner_key["token"], update_usage=True)
        viewer_verified = api_keys.verify_token(viewer_key["token"], update_usage=True)
        revoked_record = api_keys.revoke_key(revoked_key["key"]["key_id"])
        revoked_verified = api_keys.verify_token(revoked_key["token"], update_usage=True)

        workspace = product_registry.create_workspace(
            workspace_id="rehearsal-workspace",
            label="Rehearsal workspace",
            owner_user_id="rehearsal-owner",
            owner_label="Rehearsal owner",
        )
        other_workspace = product_registry.create_workspace(
            workspace_id="rehearsal-other-workspace",
            label="Rehearsal other workspace",
            owner_user_id="rehearsal-outsider",
            owner_label="Rehearsal outsider",
        )
        product_registry.add_member(
            workspace_id=workspace.workspace_id,
            user_id="rehearsal-viewer",
            label="Rehearsal viewer",
            role="viewer",
        )
        product_registry.set_quota(
            workspace_id=workspace.workspace_id,
            metric="requests",
            limit_value=1,
            window_s=86400,
        )
        product_registry.set_billing_account(
            workspace_id=workspace.workspace_id,
            billing_mode="local-ledger",
            attribution_enabled=True,
        )

        owner_admin = product_registry.permission_decision(
            workspace_id=workspace.workspace_id,
            user_id="rehearsal-owner",
            action="admin_write",
        )
        viewer_query = product_registry.permission_decision(
            workspace_id=workspace.workspace_id,
            user_id="rehearsal-viewer",
            action="query",
        )
        viewer_job = product_registry.permission_decision(
            workspace_id=workspace.workspace_id,
            user_id="rehearsal-viewer",
            action="job_submit",
        )
        viewer_other_query = product_registry.permission_decision(
            workspace_id=other_workspace.workspace_id,
            user_id="rehearsal-viewer",
            action="query",
        )
        owner_other_admin = product_registry.permission_decision(
            workspace_id=other_workspace.workspace_id,
            user_id="rehearsal-owner",
            action="admin_write",
        )
        outsider_primary_query = product_registry.permission_decision(
            workspace_id=workspace.workspace_id,
            user_id="rehearsal-outsider",
            action="query",
        )
        outsider_other_query = product_registry.permission_decision(
            workspace_id=other_workspace.workspace_id,
            user_id="rehearsal-outsider",
            action="query",
        )
        quota_first = product_registry.quota_decision(
            workspace_id=workspace.workspace_id,
            user_id="rehearsal-owner",
            metric="requests",
            amount=1,
            source="rehearsal",
            record=True,
        )
        quota_second = product_registry.quota_decision(
            workspace_id=workspace.workspace_id,
            user_id="rehearsal-owner",
            metric="requests",
            amount=1,
            source="rehearsal",
            record=True,
        )

        readiness = collect_product_readiness(
            generated_at=generated_at,
            api_token_configured=True,
            api_access_audit_enabled=True,
            api_rate_limit_enabled=True,
            api_rate_limit_max_requests=120,
            api_rate_limit_window_s=60,
            query_cost_provider="local-rehearsal",
            query_cost_prompt_usd_per_1m="0.01",
            query_cost_completion_usd_per_1m="0.01",
            identity_provider="local-registry",
            api_key_registry_backend="sqlite",
            quota_store_backend="sqlite",
            billing_provider="local-ledger",
            billing_attribution_enabled=True,
            identity_quotas_billing_enabled=True,
            product_quota_guard_enabled=True,
            product_rbac_guard_enabled=True,
            product_registry_backend="sqlite",
            api_key_registry_file=api_key_db,
            product_registry_file=product_db,
        )
        api_key_status = api_keys.status()
        product_status = product_registry.status()

    api_lifecycle_ok = (
        owner_verified is not None
        and viewer_verified is not None
        and revoked_record is not None
        and revoked_verified is None
        and api_key_status.get("active_key_count") == 2
        and api_key_status.get("revoked_key_count") == 1
    )
    rbac_ok = (
        owner_admin.get("allowed") is True
        and viewer_query.get("allowed") is True
        and viewer_job.get("allowed") is False
        and viewer_job.get("reason") == "product_role_forbidden"
    )
    quota_ok = (
        quota_first.get("allowed") is True
        and quota_second.get("limited") is True
        and quota_second.get("reason") == "quota_exceeded"
    )
    registry_ok = (
        product_status.get("workspace_count") == 2
        and product_status.get("quota_limit_count") == 1
        and product_status.get("billing_attribution_count") == 1
    )
    workspace_isolation_ok = (
        viewer_other_query.get("allowed") is False
        and viewer_other_query.get("reason") == "product_workspace_not_found"
        and owner_other_admin.get("allowed") is False
        and owner_other_admin.get("reason") == "product_workspace_not_found"
        and outsider_primary_query.get("allowed") is False
        and outsider_primary_query.get("reason") == "product_workspace_not_found"
        and outsider_other_query.get("allowed") is True
    )
    readiness_ok = bool(readiness.get("activation_ready"))
    ok = (
        api_lifecycle_ok
        and rbac_ok
        and quota_ok
        and registry_ok
        and workspace_isolation_ok
        and readiness_ok
    )

    return {
        "mode": "product_activation_rehearsal",
        "schema_version": PRODUCT_ACTIVATION_REHEARSAL_SCHEMA_VERSION,
        "generated_at": generated_at or _utc_now(),
        "ok": ok,
        "local_only": True,
        "content_exported": False,
        "secrets_exported": False,
        "paths_exported": False,
        "connectivity_checked": False,
        "api_key_lifecycle": {
            "ok": api_lifecycle_ok,
            "active_key_count": api_key_status.get("active_key_count", 0),
            "revoked_key_count": api_key_status.get("revoked_key_count", 0),
            "owner_verify_ok": owner_verified is not None,
            "viewer_verify_ok": viewer_verified is not None,
            "revoked_verify_blocked": revoked_verified is None,
            "token_exported": False,
            "hash_only_persisted": True,
        },
        "product_registry": {
            "ok": registry_ok,
            "user_count": product_status.get("user_count", 0),
            "workspace_count": product_status.get("workspace_count", 0),
            "member_count": product_status.get("member_count", 0),
            "quota_limit_count": product_status.get("quota_limit_count", 0),
            "usage_event_count": product_status.get("usage_event_count", 0),
            "billing_account_count": product_status.get("billing_account_count", 0),
            "billing_attribution_count": product_status.get("billing_attribution_count", 0),
        },
        "rbac": {
            "ok": rbac_ok,
            "owner_admin_write_allowed": owner_admin.get("allowed") is True,
            "viewer_query_allowed": viewer_query.get("allowed") is True,
            "viewer_job_submit_denied": viewer_job.get("allowed") is False,
            "viewer_job_submit_reason": viewer_job.get("reason", ""),
        },
        "workspace_isolation": {
            "ok": workspace_isolation_ok,
            "workspace_count": product_status.get("workspace_count", 0),
            "viewer_cross_workspace_query_denied": viewer_other_query.get("allowed") is False,
            "viewer_cross_workspace_query_reason": viewer_other_query.get("reason", ""),
            "owner_cross_workspace_admin_denied": owner_other_admin.get("allowed") is False,
            "owner_cross_workspace_admin_reason": owner_other_admin.get("reason", ""),
            "outsider_primary_workspace_query_denied": outsider_primary_query.get("allowed") is False,
            "outsider_primary_workspace_query_reason": outsider_primary_query.get("reason", ""),
            "outsider_own_workspace_query_allowed": outsider_other_query.get("allowed") is True,
            "identifiers_exported": False,
            "share_links_enabled": False,
            "private_corpora_enabled": False,
        },
        "quota": {
            "ok": quota_ok,
            "first_request_allowed": quota_first.get("allowed") is True,
            "second_request_limited": quota_second.get("limited") is True,
            "second_request_reason": quota_second.get("reason", ""),
            "limit_value": quota_second.get("limit_value", 0),
            "used": quota_second.get("used", 0),
            "remaining": quota_second.get("remaining", 0),
        },
        "readiness": {
            "ok": readiness_ok,
            "local_foundation_ready": readiness.get("local_foundation_ready", False),
            "activation_ready": readiness.get("activation_ready", False),
            "activation_blockers": readiness.get("blockers", {}).get("activation", []),
            "local_foundation_blockers": readiness.get("blockers", {}).get("local_foundation", []),
        },
        "notes": [
            "Rehearsal uses disposable local SQLite stores only.",
            "Raw API key tokens are verified internally and omitted from the report.",
        ],
    }


def format_product_activation_rehearsal_markdown(status: dict[str, Any]) -> str:
    """Render product activation rehearsal as no-secret Markdown."""
    api_keys = status.get("api_key_lifecycle", {}) or {}
    registry = status.get("product_registry", {}) or {}
    rbac = status.get("rbac", {}) or {}
    isolation = status.get("workspace_isolation", {}) or {}
    quota = status.get("quota", {}) or {}
    readiness = status.get("readiness", {}) or {}
    lines = [
        "# FluxMind Product Activation Rehearsal",
        "",
        f"- Mode: {status.get('mode', '')}",
        f"- Schema version: {status.get('schema_version', '')}",
        f"- Generated at: {status.get('generated_at', '')}",
        f"- OK: {_format_bool(status.get('ok', False))}",
        f"- Local only: {_format_bool(status.get('local_only', False))}",
        f"- Content exported: {_format_bool(status.get('content_exported', False))}",
        f"- Secrets exported: {_format_bool(status.get('secrets_exported', False))}",
        f"- Paths exported: {_format_bool(status.get('paths_exported', False))}",
        f"- Connectivity checked: {_format_bool(status.get('connectivity_checked', False))}",
        "",
        "## API Key Lifecycle",
        "",
        f"- OK: {_format_bool(api_keys.get('ok', False))}",
        f"- Active keys: {api_keys.get('active_key_count', 0)}",
        f"- Revoked keys: {api_keys.get('revoked_key_count', 0)}",
        f"- Owner verify OK: {_format_bool(api_keys.get('owner_verify_ok', False))}",
        f"- Viewer verify OK: {_format_bool(api_keys.get('viewer_verify_ok', False))}",
        f"- Revoked verify blocked: {_format_bool(api_keys.get('revoked_verify_blocked', False))}",
        f"- Token exported: {_format_bool(api_keys.get('token_exported', False))}",
        "",
        "## Product Registry",
        "",
        f"- OK: {_format_bool(registry.get('ok', False))}",
        f"- Users: {registry.get('user_count', 0)}",
        f"- Workspaces: {registry.get('workspace_count', 0)}",
        f"- Members: {registry.get('member_count', 0)}",
        f"- Quota limits: {registry.get('quota_limit_count', 0)}",
        f"- Usage events: {registry.get('usage_event_count', 0)}",
        f"- Billing accounts: {registry.get('billing_account_count', 0)}",
        "",
        "## RBAC",
        "",
        f"- OK: {_format_bool(rbac.get('ok', False))}",
        f"- Owner admin write allowed: {_format_bool(rbac.get('owner_admin_write_allowed', False))}",
        f"- Viewer query allowed: {_format_bool(rbac.get('viewer_query_allowed', False))}",
        f"- Viewer job submit denied: {_format_bool(rbac.get('viewer_job_submit_denied', False))}",
        f"- Viewer job submit reason: {rbac.get('viewer_job_submit_reason', '')}",
        "",
        "## Workspace Isolation",
        "",
        f"- OK: {_format_bool(isolation.get('ok', False))}",
        f"- Workspaces checked: {isolation.get('workspace_count', 0)}",
        f"- Viewer cross-workspace query denied: {_format_bool(isolation.get('viewer_cross_workspace_query_denied', False))}",
        f"- Viewer cross-workspace query reason: {isolation.get('viewer_cross_workspace_query_reason', '')}",
        f"- Owner cross-workspace admin denied: {_format_bool(isolation.get('owner_cross_workspace_admin_denied', False))}",
        f"- Owner cross-workspace admin reason: {isolation.get('owner_cross_workspace_admin_reason', '')}",
        f"- Outsider primary-workspace query denied: {_format_bool(isolation.get('outsider_primary_workspace_query_denied', False))}",
        f"- Outsider primary-workspace query reason: {isolation.get('outsider_primary_workspace_query_reason', '')}",
        f"- Outsider own-workspace query allowed: {_format_bool(isolation.get('outsider_own_workspace_query_allowed', False))}",
        f"- Identifiers exported: {_format_bool(isolation.get('identifiers_exported', False))}",
        f"- Share links enabled: {_format_bool(isolation.get('share_links_enabled', False))}",
        f"- Private corpora enabled: {_format_bool(isolation.get('private_corpora_enabled', False))}",
        "",
        "## Quota",
        "",
        f"- OK: {_format_bool(quota.get('ok', False))}",
        f"- First request allowed: {_format_bool(quota.get('first_request_allowed', False))}",
        f"- Second request limited: {_format_bool(quota.get('second_request_limited', False))}",
        f"- Second request reason: {quota.get('second_request_reason', '')}",
        f"- Limit value: {quota.get('limit_value', 0)}",
        f"- Used: {quota.get('used', 0)}",
        f"- Remaining: {quota.get('remaining', 0)}",
        "",
        "## Readiness",
        "",
        f"- OK: {_format_bool(readiness.get('ok', False))}",
        f"- Local foundation ready: {_format_bool(readiness.get('local_foundation_ready', False))}",
        f"- Activation ready: {_format_bool(readiness.get('activation_ready', False))}",
        f"- Activation blockers: {', '.join(readiness.get('activation_blockers', [])) or 'none'}",
        f"- Local foundation blockers: {', '.join(readiness.get('local_foundation_blockers', [])) or 'none'}",
    ]
    return "\n".join(lines)
