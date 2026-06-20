"""No-secret provider quota and cost guard decisions."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any

from src import config
from src.costs import estimate_query_cost_usd, query_pricing_status


PROVIDER_QUOTA_GUARD_SCHEMA_VERSION = 1
SAFE_LABEL_CHARS = set("abcdefghijklmnopqrstuvwxyz0123456789_.-")
MIN_SAFE_DECIMAL_EXPONENT = -18
MAX_SAFE_DECIMAL_EXPONENT = 18


def _decimal_is_safe(value: Decimal) -> bool:
    if not value.is_finite():
        return False
    if value.is_zero():
        return True
    adjusted = value.adjusted()
    return MIN_SAFE_DECIMAL_EXPONENT <= adjusted <= MAX_SAFE_DECIMAL_EXPONENT


def _safe_label(value: Any, *, fallback: str = "custom") -> str:
    label = str(value or "").strip().lower() or fallback
    if len(label) > 64:
        return fallback
    if any(char not in SAFE_LABEL_CHARS for char in label):
        return fallback
    return label


def _positive_int(value: Any, *, default: int = 0) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return max(0, default)
    return max(0, parsed)


def _nonnegative_decimal(value: Any) -> Decimal:
    try:
        raw = str(value if value is not None else "0").strip() or "0"
        parsed = Decimal(raw)
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")
    try:
        if not _decimal_is_safe(parsed) or parsed < 0:
            return Decimal("0")
    except InvalidOperation:
        return Decimal("0")
    return parsed


def _format_decimal(value: Decimal) -> str:
    if not _decimal_is_safe(value):
        return "0"
    text = format(value.normalize(), "f")
    if "." not in text:
        return text
    return text.rstrip("0").rstrip(".") or "0"


def provider_quota_policy(
    *,
    provider_quota_guard_enabled: bool | None = None,
    max_prompt_tokens: int | None = None,
    max_completion_tokens: int | None = None,
    max_cost_usd: str | Decimal | None = None,
    pricing_provider: str | None = None,
    prompt_usd_per_1m: str | None = None,
    completion_usd_per_1m: str | None = None,
) -> dict[str, Any]:
    """Return no-secret provider quota/cost guard policy metadata."""
    enabled = (
        config.PROVIDER_QUOTA_GUARD_ENABLED
        if provider_quota_guard_enabled is None
        else bool(provider_quota_guard_enabled)
    )
    prompt_limit = _positive_int(
        config.PROVIDER_QUOTA_MAX_PROMPT_TOKENS_PER_REQUEST
        if max_prompt_tokens is None
        else max_prompt_tokens
    )
    completion_limit = _positive_int(
        config.PROVIDER_QUOTA_MAX_COMPLETION_TOKENS_PER_REQUEST
        if max_completion_tokens is None
        else max_completion_tokens
    )
    cost_limit = _nonnegative_decimal(
        config.PROVIDER_QUOTA_MAX_COST_USD_PER_REQUEST
        if max_cost_usd is None
        else max_cost_usd
    )
    pricing = query_pricing_status(
        provider=pricing_provider or config.QUERY_COST_PROVIDER or config.LLM_MODEL,
        prompt_usd_per_1m=(
            config.QUERY_COST_PROMPT_USD_PER_1M
            if prompt_usd_per_1m is None
            else prompt_usd_per_1m
        ),
        completion_usd_per_1m=(
            config.QUERY_COST_COMPLETION_USD_PER_1M
            if completion_usd_per_1m is None
            else completion_usd_per_1m
        ),
    )
    return {
        "schema_version": PROVIDER_QUOTA_GUARD_SCHEMA_VERSION,
        "enabled": enabled,
        "max_prompt_tokens_per_request": prompt_limit,
        "max_completion_tokens_per_request": completion_limit,
        "max_cost_usd_per_request": _format_decimal(cost_limit),
        "cost_limit_configured": cost_limit > 0,
        "pricing_configured": bool(pricing.get("configured", False)),
        "pricing_reason": pricing.get("reason", ""),
        "content_exported": False,
        "secrets_exported": False,
    }


def provider_quota_guard_decision(
    *,
    operation: str,
    provider: str,
    estimated_prompt_tokens: int,
    requested_completion_tokens: int,
    provider_quota_guard_enabled: bool | None = None,
    max_prompt_tokens: int | None = None,
    max_completion_tokens: int | None = None,
    max_cost_usd: str | Decimal | None = None,
    prompt_usd_per_1m: str | None = None,
    completion_usd_per_1m: str | None = None,
) -> dict[str, Any]:
    """Decide whether a provider call is allowed before external execution.

    The decision intentionally contains only operation/provider labels, counts,
    thresholds, and reason codes. It never receives or returns prompts, answers,
    file paths, provider URLs, or credential material.
    """
    prompt_tokens = _positive_int(estimated_prompt_tokens)
    completion_tokens = _positive_int(requested_completion_tokens)
    policy = provider_quota_policy(
        provider_quota_guard_enabled=provider_quota_guard_enabled,
        max_prompt_tokens=max_prompt_tokens,
        max_completion_tokens=max_completion_tokens,
        max_cost_usd=max_cost_usd,
        pricing_provider=provider,
        prompt_usd_per_1m=prompt_usd_per_1m,
        completion_usd_per_1m=completion_usd_per_1m,
    )
    base = {
        "enabled": policy["enabled"],
        "operation": _safe_label(operation),
        "provider": _safe_label(provider, fallback="unspecified"),
        "estimated_prompt_tokens": prompt_tokens,
        "requested_completion_tokens": completion_tokens,
        "estimated_total_tokens": prompt_tokens + completion_tokens,
        "max_prompt_tokens_per_request": policy["max_prompt_tokens_per_request"],
        "max_completion_tokens_per_request": policy["max_completion_tokens_per_request"],
        "max_cost_usd_per_request": policy["max_cost_usd_per_request"],
        "estimated_cost_usd": "0",
        "cost_limit_configured": policy["cost_limit_configured"],
        "pricing_configured": policy["pricing_configured"],
        "content_exported": False,
        "secrets_exported": False,
    }
    if not policy["enabled"]:
        return {
            **base,
            "allowed": True,
            "limited": False,
            "reason": "provider_quota_guard_disabled",
            "status_code": 200,
        }

    prompt_limit = int(policy["max_prompt_tokens_per_request"])
    completion_limit = int(policy["max_completion_tokens_per_request"])
    if prompt_limit <= 0 or completion_limit <= 0:
        return {
            **base,
            "allowed": False,
            "limited": False,
            "reason": "provider_quota_guard_invalid_limit",
            "status_code": 503,
        }
    if prompt_tokens > prompt_limit:
        return {
            **base,
            "allowed": False,
            "limited": True,
            "reason": "provider_prompt_token_limit_exceeded",
            "status_code": 429,
        }
    if completion_tokens > completion_limit:
        return {
            **base,
            "allowed": False,
            "limited": True,
            "reason": "provider_completion_token_limit_exceeded",
            "status_code": 429,
        }

    cost_limit = _nonnegative_decimal(policy["max_cost_usd_per_request"])
    if cost_limit > 0:
        if not policy["pricing_configured"]:
            return {
                **base,
                "allowed": False,
                "limited": False,
                "reason": "provider_cost_pricing_not_configured",
                "status_code": 503,
            }
        estimated_cost = _nonnegative_decimal(
            estimate_query_cost_usd(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                prompt_usd_per_1m=(
                    config.QUERY_COST_PROMPT_USD_PER_1M
                    if prompt_usd_per_1m is None
                    else prompt_usd_per_1m
                ),
                completion_usd_per_1m=(
                    config.QUERY_COST_COMPLETION_USD_PER_1M
                    if completion_usd_per_1m is None
                    else completion_usd_per_1m
                ),
            )
        )
        base["estimated_cost_usd"] = _format_decimal(estimated_cost)
        if estimated_cost > cost_limit:
            return {
                **base,
                "allowed": False,
                "limited": True,
                "reason": "provider_cost_limit_exceeded",
                "status_code": 429,
            }

    return {
        **base,
        "allowed": True,
        "limited": False,
        "reason": "allowed",
        "status_code": 200,
    }
