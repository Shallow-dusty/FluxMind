"""No-secret readiness for private corpora and share-link collaboration."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src import config
from src.product_registry import product_registry_backend_status
from src.share_links import share_link_registry_backend_status


COLLABORATION_READINESS_SCHEMA_VERSION = 1
ROLE_ORDER = ("owner", "admin", "member", "viewer")
WRITE_ROLES = {"owner", "admin"}
READ_ROLES = {"owner", "admin", "member", "viewer"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _format_bool(value: Any) -> str:
    return "true" if bool(value) else "false"


def _safe_backend(value: str | None) -> str:
    backend = (value or "none").strip().lower() or "none"
    if len(backend) > 64:
        return "custom"
    if any(char not in "abcdefghijklmnopqrstuvwxyz0123456789_.-" for char in backend):
        return "custom"
    return backend


def _role_decision(
    *,
    role: str,
    action_roles: set[str],
    feature_enabled: bool,
    guard_ready: bool,
    disabled_reason: str,
) -> dict[str, Any]:
    if not feature_enabled:
        return {"role": role, "allowed": False, "reason": disabled_reason}
    if not guard_ready:
        return {"role": role, "allowed": False, "reason": "collaboration_guard_not_ready"}
    allowed = role in action_roles
    return {
        "role": role,
        "allowed": allowed,
        "reason": "allowed" if allowed else "product_role_forbidden",
    }


def _role_matrix(
    *,
    feature_enabled: bool,
    guard_ready: bool,
    disabled_reason: str,
    action_roles: set[str],
) -> list[dict[str, Any]]:
    return [
        _role_decision(
            role=role,
            action_roles=action_roles,
            feature_enabled=feature_enabled,
            guard_ready=guard_ready,
            disabled_reason=disabled_reason,
        )
        for role in ROLE_ORDER
    ]


def _share_create_matrix(
    *,
    feature_enabled: bool,
    product_guard_ready: bool,
    token_store_available: bool,
    token_store_reason: str,
) -> list[dict[str, Any]]:
    if not feature_enabled:
        reason = "share_links_disabled"
        guard_ready = False
    elif not product_guard_ready:
        reason = "collaboration_guard_not_ready"
        guard_ready = False
    elif not token_store_available:
        reason = token_store_reason
        guard_ready = False
    else:
        reason = ""
        guard_ready = True
    return [
        {
            "role": role,
            "allowed": role in WRITE_ROLES if guard_ready else False,
            "reason": (
                "allowed"
                if guard_ready and role in WRITE_ROLES
                else (
                    "product_role_forbidden"
                    if guard_ready
                    else reason
                )
            ),
        }
        for role in ROLE_ORDER
    ]


def _denied_count(items: list[dict[str, Any]]) -> int:
    return sum(1 for item in items if item.get("allowed") is not True)


def collect_collaboration_readiness(
    *,
    generated_at: str | None = None,
    private_corpora_enabled: bool | None = None,
    share_links_enabled: bool | None = None,
    product_rbac_guard_enabled: bool | None = None,
    product_registry_backend: str | None = None,
    product_registry_file: Path | None = None,
    share_link_token_store_backend: str | None = None,
    share_link_token_store_file: Path | None = None,
    owner_metadata_supported: bool = True,
) -> dict[str, Any]:
    """Collect collaboration readiness without exporting identifiers or links."""
    private_enabled = (
        bool(config.PRIVATE_CORPORA_ENABLED)
        if private_corpora_enabled is None
        else bool(private_corpora_enabled)
    )
    share_enabled = (
        bool(config.SHARE_LINKS_ENABLED)
        if share_links_enabled is None
        else bool(share_links_enabled)
    )
    rbac_guard = (
        bool(config.PRODUCT_RBAC_GUARD_ENABLED)
        if product_rbac_guard_enabled is None
        else bool(product_rbac_guard_enabled)
    )
    registry_backend = (
        product_registry_backend
        if product_registry_backend is not None
        else config.PRODUCT_REGISTRY_BACKEND
    )
    registry = product_registry_backend_status(
        backend=registry_backend,
        db_path=product_registry_file,
    )
    token_store = share_link_registry_backend_status(
        backend=share_link_token_store_backend,
        db_path=share_link_token_store_file,
    )
    token_store_backend = _safe_backend(str(token_store.get("backend", "none")))
    token_store_configured = bool(token_store.get("configured", False))
    token_store_available = bool(token_store.get("available", False))
    token_store_reason = str(token_store.get("reason", "share_link_token_store_unavailable"))
    product_guard_ready = bool(registry.get("available")) and rbac_guard
    private_ready = private_enabled and product_guard_ready
    share_ready = share_enabled and product_guard_ready and token_store_available

    private_read = _role_matrix(
        feature_enabled=private_enabled,
        guard_ready=product_guard_ready,
        disabled_reason="private_corpora_disabled",
        action_roles=READ_ROLES,
    )
    private_write = _role_matrix(
        feature_enabled=private_enabled,
        guard_ready=product_guard_ready,
        disabled_reason="private_corpora_disabled",
        action_roles=WRITE_ROLES,
    )
    share_create = _share_create_matrix(
        feature_enabled=share_enabled,
        product_guard_ready=product_guard_ready,
        token_store_available=token_store_available,
        token_store_reason=token_store_reason,
    )
    anonymous_reason = (
        "share_links_disabled"
            if not share_enabled
            else (
            token_store_reason
            if not token_store_available
            else "share_link_token_required"
        )
    )

    local_blockers: list[str] = []
    if not owner_metadata_supported:
        local_blockers.append("owner_metadata_contract_missing")

    activation_blockers: list[str] = []
    if not private_enabled:
        activation_blockers.append("private_corpora_disabled")
    if not share_enabled:
        activation_blockers.append("share_links_disabled")
    if not registry.get("configured"):
        activation_blockers.append("product_registry_not_configured")
    elif not registry.get("available"):
        activation_blockers.append("product_registry_unavailable")
    if not rbac_guard:
        activation_blockers.append("product_rbac_guard_not_enabled")
    if share_enabled and not token_store_configured:
        activation_blockers.append("share_link_token_store_not_configured")
    elif share_enabled and not token_store_available:
        activation_blockers.append("share_link_token_store_unavailable")

    safe_default_ready = not private_enabled and not share_enabled
    local_foundation_ready = not local_blockers
    activation_ready = private_ready and share_ready and not activation_blockers
    ok = local_foundation_ready and (safe_default_ready or activation_ready)

    denied_total = (
        _denied_count(private_read)
        + _denied_count(private_write)
        + _denied_count(share_create)
        + 1
    )
    return {
        "mode": "collaboration_readiness",
        "schema_version": COLLABORATION_READINESS_SCHEMA_VERSION,
        "generated_at": generated_at or _utc_now(),
        "ok": ok,
        "local_foundation_ready": local_foundation_ready,
        "safe_default_ready": safe_default_ready,
        "activation_ready": activation_ready,
        "local_only": True,
        "content_exported": False,
        "secrets_exported": False,
        "paths_exported": False,
        "identifiers_exported": False,
        "share_tokens_exported": False,
        "share_urls_exported": False,
        "connectivity_checked": False,
        "summary": {
            "private_corpora_enabled": private_enabled,
            "share_links_enabled": share_enabled,
            "product_registry_available": bool(registry.get("available")),
            "product_rbac_guard_enabled": rbac_guard,
            "share_link_token_store_available": token_store_available,
            "owner_metadata_supported": bool(owner_metadata_supported),
            "policy_scenario_count": 13,
            "policy_denied_count": denied_total,
        },
        "checks": {
            "feature_flags": {
                "private_corpora_enabled": private_enabled,
                "share_links_enabled": share_enabled,
                "reason": "safe_default" if safe_default_ready else "operator_enabled",
            },
            "product_registry": {
                "backend": registry.get("backend", "none"),
                "configured": bool(registry.get("configured")),
                "available": bool(registry.get("available")),
                "reason": registry.get("reason", ""),
            },
            "product_rbac_guard": {
                "enabled": rbac_guard,
                "reason": "enabled" if rbac_guard else "product_rbac_guard_not_enabled",
            },
            "share_link_token_store": {
                "backend": token_store_backend,
                "configured": token_store_configured,
                "supported": bool(token_store.get("supported", False)),
                "available": token_store_available,
                "reason": token_store_reason,
                "active_link_count": int(token_store.get("active_link_count", 0) or 0),
                "total_link_count": int(token_store.get("total_link_count", 0) or 0),
            },
            "private_corpus_policy": {
                "ready": private_ready,
                "read": private_read,
                "write": private_write,
                "cross_workspace_read_denied": True,
                "cross_workspace_read_reason": "product_workspace_not_found",
                "cross_workspace_write_denied": True,
                "cross_workspace_write_reason": "product_workspace_not_found",
            },
            "share_link_policy": {
                "ready": share_ready,
                "create": share_create,
                "anonymous_redeem_allowed": False,
                "anonymous_redeem_reason": anonymous_reason,
                "revocation_required": True,
                "expiry_required": True,
            },
        },
        "blockers": {
            "local_foundation": local_blockers,
            "activation": activation_blockers,
        },
        "notes": [
            "This check is a pre-activation gate; it does not create corpora or share links.",
            "Role names and blocker codes are reported, but workspace IDs, user IDs, corpus IDs, share tokens, and URLs are omitted.",
        ],
    }


def _format_role_matrix(items: list[dict[str, Any]]) -> list[str]:
    return [
        f"- {item.get('role', '')}: allowed={_format_bool(item.get('allowed', False))}; "
        f"reason={item.get('reason', '')}"
        for item in items
    ]


def format_collaboration_readiness_markdown(status: dict[str, Any]) -> str:
    """Render collaboration readiness as no-secret Markdown."""
    summary = status.get("summary", {}) or {}
    checks = status.get("checks", {}) or {}
    blockers = status.get("blockers", {}) or {}
    private_policy = checks.get("private_corpus_policy", {}) or {}
    share_policy = checks.get("share_link_policy", {}) or {}
    registry = checks.get("product_registry", {}) or {}
    rbac = checks.get("product_rbac_guard", {}) or {}
    token_store = checks.get("share_link_token_store", {}) or {}
    lines = [
        "# FluxMind Collaboration Readiness",
        "",
        f"- Mode: {status.get('mode', '')}",
        f"- Schema version: {status.get('schema_version', '')}",
        f"- Generated at: {status.get('generated_at', '')}",
        f"- OK: {_format_bool(status.get('ok', False))}",
        f"- Local foundation ready: {_format_bool(status.get('local_foundation_ready', False))}",
        f"- Safe default ready: {_format_bool(status.get('safe_default_ready', False))}",
        f"- Activation ready: {_format_bool(status.get('activation_ready', False))}",
        f"- Local only: {_format_bool(status.get('local_only', False))}",
        f"- Content exported: {_format_bool(status.get('content_exported', False))}",
        f"- Secrets exported: {_format_bool(status.get('secrets_exported', False))}",
        f"- Paths exported: {_format_bool(status.get('paths_exported', False))}",
        f"- Identifiers exported: {_format_bool(status.get('identifiers_exported', False))}",
        f"- Share tokens exported: {_format_bool(status.get('share_tokens_exported', False))}",
        f"- Share URLs exported: {_format_bool(status.get('share_urls_exported', False))}",
        "",
        "## Summary",
        "",
        f"- Private corpora enabled: {_format_bool(summary.get('private_corpora_enabled', False))}",
        f"- Share links enabled: {_format_bool(summary.get('share_links_enabled', False))}",
        f"- Product registry available: {_format_bool(summary.get('product_registry_available', False))}",
        f"- Product RBAC guard enabled: {_format_bool(summary.get('product_rbac_guard_enabled', False))}",
        f"- Share-link token store available: {_format_bool(summary.get('share_link_token_store_available', False))}",
        f"- Policy scenarios: {summary.get('policy_scenario_count', 0)}",
        f"- Denied scenarios: {summary.get('policy_denied_count', 0)}",
        "",
        "## Prerequisites",
        "",
        f"- Product registry: {registry.get('backend', '')} ({registry.get('reason', '')})",
        f"- Product RBAC guard: {_format_bool(rbac.get('enabled', False))} ({rbac.get('reason', '')})",
        f"- Share-link token store: {token_store.get('backend', '')} ({token_store.get('reason', '')})",
        "",
        "## Private Corpus Policy",
        "",
        f"- Ready: {_format_bool(private_policy.get('ready', False))}",
        "- Read roles:",
    ]
    lines.extend(_format_role_matrix(private_policy.get("read", []) or []))
    lines.append("- Write roles:")
    lines.extend(_format_role_matrix(private_policy.get("write", []) or []))
    lines.extend(
        [
            f"- Cross-workspace read denied: {_format_bool(private_policy.get('cross_workspace_read_denied', False))}",
            f"- Cross-workspace write denied: {_format_bool(private_policy.get('cross_workspace_write_denied', False))}",
            "",
            "## Share-Link Policy",
            "",
            f"- Ready: {_format_bool(share_policy.get('ready', False))}",
            "- Create roles:",
        ]
    )
    lines.extend(_format_role_matrix(share_policy.get("create", []) or []))
    lines.extend(
        [
            f"- Anonymous redeem allowed: {_format_bool(share_policy.get('anonymous_redeem_allowed', False))}",
            f"- Anonymous redeem reason: {share_policy.get('anonymous_redeem_reason', '')}",
            f"- Revocation required: {_format_bool(share_policy.get('revocation_required', False))}",
            f"- Expiry required: {_format_bool(share_policy.get('expiry_required', False))}",
            "",
            "## Blockers",
            "",
            f"- Local foundation: {', '.join(blockers.get('local_foundation', [])) or 'none'}",
            f"- Activation: {', '.join(blockers.get('activation', [])) or 'none'}",
        ]
    )
    return "\n".join(lines)
