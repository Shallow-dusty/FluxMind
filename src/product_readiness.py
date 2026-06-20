"""No-secret productization readiness for identity, quotas, and billing."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src import config
from src.api_keys import api_key_registry_backend_status
from src.costs import query_pricing_status
from src.product_registry import product_registry_backend_status


PRODUCT_READINESS_SCHEMA_VERSION = 1
DISABLED_BACKENDS = {"", "none", "disabled", "local", "local-disabled", "mock"}
SAFE_BACKEND_CHARS = set("abcdefghijklmnopqrstuvwxyz0123456789_.-")
SUPPORTED_IDENTITY_PROVIDERS = {
    "auth0",
    "authentik",
    "clerk",
    "external",
    "local-registry",
    "oauth2",
    "oidc",
    "supabase",
    "zitadel",
}
SUPPORTED_API_KEY_REGISTRIES = {"external", "postgres", "postgresql", "sqlite"}
SUPPORTED_QUOTA_STORES = {"external", "postgres", "postgresql", "redis", "sqlite"}
SUPPORTED_BILLING_PROVIDERS = {"external", "lemonsqueezy", "paddle", "stripe", "local-ledger"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_backend_name(value: str | None) -> str:
    backend = (value or "none").strip().lower() or "none"
    if len(backend) > 64:
        return "custom"
    if any(char not in SAFE_BACKEND_CHARS for char in backend):
        return "custom"
    return backend


def _backend_status(
    *,
    value: str | None,
    supported: set[str],
    missing_reason: str,
    unsupported_reason: str,
) -> dict[str, Any]:
    backend = _safe_backend_name(value)
    configured = backend not in DISABLED_BACKENDS
    supported_backend = configured and backend in supported
    if not configured:
        reason = missing_reason
    elif supported_backend:
        reason = "configured_not_connected"
    else:
        reason = unsupported_reason
    return {
        "backend": backend,
        "configured": configured,
        "supported": supported_backend,
        "available": supported_backend,
        "reason": reason,
    }


def _positive_int(value: int | str | None) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def collect_product_readiness(
    *,
    generated_at: str | None = None,
    api_token_configured: bool | None = None,
    api_access_audit_enabled: bool | None = None,
    api_rate_limit_enabled: bool | None = None,
    api_rate_limit_max_requests: int | None = None,
    api_rate_limit_window_s: int | None = None,
    query_cost_provider: str | None = None,
    query_cost_prompt_usd_per_1m: str | None = None,
    query_cost_completion_usd_per_1m: str | None = None,
    identity_provider: str | None = None,
    api_key_registry_backend: str | None = None,
    quota_store_backend: str | None = None,
    billing_provider: str | None = None,
    billing_attribution_enabled: bool | None = None,
    identity_quotas_billing_enabled: bool | None = None,
    product_quota_guard_enabled: bool | None = None,
    product_rbac_guard_enabled: bool | None = None,
    product_registry_backend: str | None = None,
    api_key_registry_file: Path | None = None,
    product_registry_file: Path | None = None,
    owner_metadata_supported: bool = True,
) -> dict[str, Any]:
    """Collect productization readiness without exposing secrets or identities."""
    token_configured = (
        config.FLUXMIND_API_TOKEN_CONFIGURED
        if api_token_configured is None
        else bool(api_token_configured)
    )
    audit_enabled = (
        config.API_ACCESS_AUDIT_ENABLED
        if api_access_audit_enabled is None
        else bool(api_access_audit_enabled)
    )
    rate_limit_enabled = (
        config.API_RATE_LIMIT_ENABLED
        if api_rate_limit_enabled is None
        else bool(api_rate_limit_enabled)
    )
    rate_limit_max = _positive_int(
        config.API_RATE_LIMIT_MAX_REQUESTS
        if api_rate_limit_max_requests is None
        else api_rate_limit_max_requests
    )
    rate_limit_window = _positive_int(
        config.API_RATE_LIMIT_WINDOW_S
        if api_rate_limit_window_s is None
        else api_rate_limit_window_s
    )
    billing_attribution = (
        config.BILLING_ATTRIBUTION_ENABLED
        if billing_attribution_enabled is None
        else bool(billing_attribution_enabled)
    )
    product_enabled = (
        config.IDENTITY_QUOTAS_BILLING_ENABLED
        if identity_quotas_billing_enabled is None
        else bool(identity_quotas_billing_enabled)
    )
    quota_guard_enabled = (
        config.PRODUCT_QUOTA_GUARD_ENABLED
        if product_quota_guard_enabled is None
        else bool(product_quota_guard_enabled)
    )
    rbac_guard_enabled = (
        config.PRODUCT_RBAC_GUARD_ENABLED
        if product_rbac_guard_enabled is None
        else bool(product_rbac_guard_enabled)
    )

    pricing = query_pricing_status(
        provider=query_cost_provider if query_cost_provider is not None else config.QUERY_COST_PROVIDER,
        prompt_usd_per_1m=(
            query_cost_prompt_usd_per_1m
            if query_cost_prompt_usd_per_1m is not None
            else config.QUERY_COST_PROMPT_USD_PER_1M
        ),
        completion_usd_per_1m=(
            query_cost_completion_usd_per_1m
            if query_cost_completion_usd_per_1m is not None
            else config.QUERY_COST_COMPLETION_USD_PER_1M
        ),
    )
    pricing_rates_valid = pricing["reason"] != "invalid_rate"
    rate_limit_configured = rate_limit_max > 0 and rate_limit_window > 0
    product_registry = product_registry_backend_status(
        backend=(
            product_registry_backend
            if product_registry_backend is not None
            else config.PRODUCT_REGISTRY_BACKEND
        ),
        db_path=product_registry_file,
    )

    identity = _backend_status(
        value=identity_provider if identity_provider is not None else config.IDENTITY_PROVIDER,
        supported=SUPPORTED_IDENTITY_PROVIDERS,
        missing_reason="identity_provider_not_configured",
        unsupported_reason="unsupported_identity_provider",
    )
    if identity["backend"] == "local-registry":
        identity.update(
            {
                "available": bool(product_registry.get("available", False)),
                "reason": (
                    "available"
                    if product_registry.get("available", False)
                    else "product_registry_unavailable"
                ),
                "user_count": product_registry.get("user_count", 0),
                "workspace_count": product_registry.get("workspace_count", 0),
            }
        )
    api_key_registry = _backend_status(
        value=(
            api_key_registry_backend
            if api_key_registry_backend is not None
            else config.API_KEY_REGISTRY_BACKEND
        ),
        supported=SUPPORTED_API_KEY_REGISTRIES,
        missing_reason="api_key_registry_not_configured",
        unsupported_reason="unsupported_api_key_registry",
    )
    if api_key_registry["backend"] == "sqlite":
        local_registry = api_key_registry_backend_status(
            backend=api_key_registry["backend"],
            db_path=api_key_registry_file,
        )
        api_key_registry.update(
            {
                "available": bool(local_registry.get("available", False)),
                "reason": local_registry.get("reason", api_key_registry["reason"]),
                "active_key_count": local_registry.get("active_key_count", 0),
                "revoked_key_count": local_registry.get("revoked_key_count", 0),
            }
        )
    quota_store = _backend_status(
        value=quota_store_backend if quota_store_backend is not None else config.QUOTA_STORE_BACKEND,
        supported=SUPPORTED_QUOTA_STORES,
        missing_reason="quota_store_not_configured",
        unsupported_reason="unsupported_quota_store",
    )
    if quota_store["backend"] == "sqlite":
        quota_store.update(
            {
                "available": bool(product_registry.get("available", False)),
                "reason": (
                    "available"
                    if product_registry.get("available", False)
                    else "product_registry_unavailable"
                ),
                "quota_limit_count": product_registry.get("quota_limit_count", 0),
                "usage_event_count": product_registry.get("usage_event_count", 0),
            }
        )
    billing = _backend_status(
        value=billing_provider if billing_provider is not None else config.BILLING_PROVIDER,
        supported=SUPPORTED_BILLING_PROVIDERS,
        missing_reason="billing_provider_not_configured",
        unsupported_reason="unsupported_billing_provider",
    )
    if billing["backend"] == "local-ledger":
        billing.update(
            {
                "available": bool(product_registry.get("available", False)),
                "reason": (
                    "available"
                    if product_registry.get("available", False)
                    else "product_registry_unavailable"
                ),
                "billing_account_count": product_registry.get("billing_account_count", 0),
                "billing_attribution_count": product_registry.get("billing_attribution_count", 0),
            }
        )

    local_blockers: list[str] = []
    if not audit_enabled:
        local_blockers.append("api_access_audit_disabled")
    if not rate_limit_configured:
        local_blockers.append("local_rate_limit_config_invalid")
    if not owner_metadata_supported:
        local_blockers.append("owner_metadata_contract_missing")
    if not pricing_rates_valid:
        local_blockers.append("query_cost_rates_invalid")

    activation_blockers: list[str] = []
    if not identity["configured"]:
        activation_blockers.append("multi_user_identity_not_configured")
    elif not identity["supported"]:
        activation_blockers.append("identity_provider_unsupported")
    elif not identity["available"]:
        activation_blockers.append("identity_provider_unavailable")
    if not api_key_registry["configured"]:
        activation_blockers.append("api_key_lifecycle_not_configured")
    elif not api_key_registry["supported"]:
        activation_blockers.append("api_key_registry_unsupported")
    elif not api_key_registry["available"]:
        activation_blockers.append("api_key_registry_unavailable")
    if not quota_store["configured"]:
        activation_blockers.append("identity_quota_store_not_configured")
    elif not quota_store["supported"]:
        activation_blockers.append("quota_store_unsupported")
    elif quota_store["backend"] == "sqlite" and not quota_store["available"]:
        activation_blockers.append("quota_store_unavailable")
    if not billing["configured"]:
        activation_blockers.append("billing_provider_not_configured")
    elif not billing["supported"]:
        activation_blockers.append("billing_provider_unsupported")
    elif billing["backend"] == "local-ledger" and not billing["available"]:
        activation_blockers.append("billing_provider_unavailable")
    if not billing_attribution:
        activation_blockers.append("billing_attribution_not_enabled")
    if product_enabled and not quota_guard_enabled:
        activation_blockers.append("product_quota_guard_not_enabled")
    if product_enabled and not rbac_guard_enabled:
        activation_blockers.append("product_rbac_guard_not_enabled")

    advisories: list[str] = []
    if not token_configured:
        advisories.append("single_api_token_not_configured")
    if not rate_limit_enabled:
        advisories.append("local_rate_limit_disabled")
    if not pricing["configured"]:
        advisories.append("query_cost_pricing_not_configured")
    if not product_enabled:
        advisories.append("identity_quotas_billing_runtime_disabled")
    if not quota_guard_enabled:
        advisories.append("product_quota_guard_disabled")
    if not rbac_guard_enabled:
        advisories.append("product_rbac_guard_disabled")

    local_foundation_ready = not local_blockers
    activation_ready = local_foundation_ready and not activation_blockers

    return {
        "mode": "product_readiness",
        "schema_version": PRODUCT_READINESS_SCHEMA_VERSION,
        "generated_at": generated_at or _utc_now(),
        "local_foundation_ready": local_foundation_ready,
        "activation_ready": activation_ready,
        "identity_quotas_billing_enabled": product_enabled,
        "content_exported": False,
        "secrets_exported": False,
        "connectivity_checked": False,
        "summary": {
            "api_access_audit_enabled": audit_enabled,
            "single_api_token_configured": token_configured,
            "local_rate_limit_configured": rate_limit_configured,
            "local_rate_limit_enabled": rate_limit_enabled,
            "owner_metadata_supported": bool(owner_metadata_supported),
            "query_cost_estimation_available": True,
            "query_cost_pricing_configured": bool(pricing["configured"]),
            "api_key_lifecycle_available": bool(api_key_registry["available"]),
            "product_registry_available": bool(product_registry["available"]),
            "workspace_identity_available": bool(identity["available"]),
            "quota_store_available": bool(quota_store["available"]),
            "billing_ledger_available": bool(billing["available"]),
            "product_quota_guard_enabled": quota_guard_enabled,
            "product_rbac_guard_enabled": rbac_guard_enabled,
            "external_billing_enabled": False,
            "identity_quotas_billing_enabled": product_enabled,
        },
        "checks": {
            "api_access_audit": {
                "enabled": audit_enabled,
                "reason": "enabled" if audit_enabled else "disabled",
            },
            "api_token": {
                "configured": token_configured,
                "scope": "single_shared_token",
                "secret_exported": False,
            },
            "local_rate_limit": {
                "enabled": rate_limit_enabled,
                "configured": rate_limit_configured,
                "max_requests": rate_limit_max,
                "window_s": rate_limit_window,
                "scope": "local_in_memory",
            },
            "owner_metadata": {
                "supported": bool(owner_metadata_supported),
                "scope": "metadata_only_not_identity",
            },
            "query_cost_estimation": {
                "available": True,
                "pricing_configured": bool(pricing["configured"]),
                "pricing_reason": pricing["reason"],
                "rates_valid": pricing_rates_valid,
                "external_billing_enabled": False,
            },
            "identity_provider": identity,
            "product_registry": product_registry,
            "api_key_registry": api_key_registry,
            "quota_store": quota_store,
            "billing_provider": billing,
            "billing_attribution": {
                "enabled": billing_attribution,
                "reason": "enabled" if billing_attribution else "billing_attribution_not_enabled",
            },
            "product_quota_guard": {
                "enabled": quota_guard_enabled,
                "metric": getattr(config, "PRODUCT_QUOTA_METRIC", "requests"),
                "reason": "enabled" if quota_guard_enabled else "product_quota_guard_disabled",
            },
            "product_rbac_guard": {
                "enabled": rbac_guard_enabled,
                "role_source": "local_product_registry",
                "reason": "enabled" if rbac_guard_enabled else "product_rbac_guard_disabled",
            },
        },
        "blockers": {
            "local_foundation": local_blockers,
            "activation": activation_blockers,
        },
        "advisories": advisories,
        "notes": [
            "Local owner fields, audit events, rate limits, and cost estimates are readiness foundations only.",
            "Local product registry can satisfy local workspace, RBAC, quota, and billing-attribution contracts but is not an external identity or payment provider.",
        ],
    }


def _format_bool(value: Any) -> str:
    return "true" if bool(value) else "false"


def format_product_readiness_markdown(status: dict[str, Any]) -> str:
    """Render product readiness as no-secret Markdown."""
    summary = status.get("summary", {}) or {}
    checks = status.get("checks", {}) or {}
    blockers = status.get("blockers", {}) or {}
    lines = [
        "# FluxMind Product Readiness",
        "",
        f"- Mode: {status.get('mode', '')}",
        f"- Schema version: {status.get('schema_version', '')}",
        f"- Generated at: {status.get('generated_at', '')}",
        f"- Local foundation ready: {_format_bool(status.get('local_foundation_ready', False))}",
        f"- Activation ready: {_format_bool(status.get('activation_ready', False))}",
        f"- Identity/quotas/billing enabled: {_format_bool(status.get('identity_quotas_billing_enabled', False))}",
        f"- Content exported: {_format_bool(status.get('content_exported', False))}",
        f"- Secrets exported: {_format_bool(status.get('secrets_exported', False))}",
        f"- Connectivity checked: {_format_bool(status.get('connectivity_checked', False))}",
        "",
        "## Local Foundation",
        "",
        f"- API access audit enabled: {_format_bool(summary.get('api_access_audit_enabled', False))}",
        f"- Single API token configured: {_format_bool(summary.get('single_api_token_configured', False))}",
        f"- Local rate limit enabled: {_format_bool(summary.get('local_rate_limit_enabled', False))}",
        f"- Local rate limit configured: {_format_bool(summary.get('local_rate_limit_configured', False))}",
        f"- Owner metadata supported: {_format_bool(summary.get('owner_metadata_supported', False))}",
        f"- Query cost pricing configured: {_format_bool(summary.get('query_cost_pricing_configured', False))}",
        f"- Product registry available: {_format_bool(summary.get('product_registry_available', False))}",
        f"- Product quota guard enabled: {_format_bool(summary.get('product_quota_guard_enabled', False))}",
        f"- Product RBAC guard enabled: {_format_bool(summary.get('product_rbac_guard_enabled', False))}",
        f"- External billing enabled: {_format_bool(summary.get('external_billing_enabled', False))}",
        "",
        "## Activation Targets",
        "",
        f"- Identity provider: {checks.get('identity_provider', {}).get('backend', '')} ({checks.get('identity_provider', {}).get('reason', '')})",
        f"- Product registry: {checks.get('product_registry', {}).get('backend', '')} ({checks.get('product_registry', {}).get('reason', '')})",
        f"- API key registry: {checks.get('api_key_registry', {}).get('backend', '')} ({checks.get('api_key_registry', {}).get('reason', '')})",
        f"- Quota store: {checks.get('quota_store', {}).get('backend', '')} ({checks.get('quota_store', {}).get('reason', '')})",
        f"- Billing provider: {checks.get('billing_provider', {}).get('backend', '')} ({checks.get('billing_provider', {}).get('reason', '')})",
        f"- Billing attribution: {_format_bool(checks.get('billing_attribution', {}).get('enabled', False))}",
        f"- Product quota guard: {_format_bool(checks.get('product_quota_guard', {}).get('enabled', False))}",
        f"- Product RBAC guard: {_format_bool(checks.get('product_rbac_guard', {}).get('enabled', False))}",
        "",
        "## Blockers",
        "",
        f"- Local foundation: {', '.join(blockers.get('local_foundation', [])) or 'none'}",
        f"- Activation: {', '.join(blockers.get('activation', [])) or 'none'}",
        f"- Advisories: {', '.join(status.get('advisories', [])) or 'none'}",
    ]
    return "\n".join(lines)
