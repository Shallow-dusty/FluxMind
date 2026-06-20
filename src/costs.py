"""No-secret local cost estimation helpers for query usage."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any

ONE_MILLION = Decimal("1000000")
USD_QUANTUM = Decimal("0.000001")
MIN_SAFE_DECIMAL_EXPONENT = -18
MAX_SAFE_DECIMAL_EXPONENT = 18


def _decimal_is_safe(value: Decimal) -> bool:
    if not value.is_finite():
        return False
    if value.is_zero():
        return True
    adjusted = value.adjusted()
    return MIN_SAFE_DECIMAL_EXPONENT <= adjusted <= MAX_SAFE_DECIMAL_EXPONENT


def _parse_decimal(value: str | int | float | Decimal | None) -> tuple[Decimal, bool]:
    try:
        raw = str(value if value is not None else "0").strip() or "0"
        parsed = Decimal(raw)
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0"), False
    try:
        if not _decimal_is_safe(parsed) or parsed < 0:
            return Decimal("0"), False
    except InvalidOperation:
        return Decimal("0"), False
    return parsed, True


def _format_decimal(value: Decimal) -> str:
    if not _decimal_is_safe(value):
        return "0"
    text = format(value.normalize(), "f")
    if "." not in text:
        return text
    return text.rstrip("0").rstrip(".") or "0"


def format_usd(value: Decimal) -> str:
    if not _decimal_is_safe(value):
        return "0"
    try:
        rounded = value.quantize(USD_QUANTUM, rounding=ROUND_HALF_UP)
    except InvalidOperation:
        return "0"
    return _format_decimal(rounded)


def query_pricing_status(
    *,
    provider: str,
    prompt_usd_per_1m: str,
    completion_usd_per_1m: str,
) -> dict[str, Any]:
    prompt_rate, prompt_valid = _parse_decimal(prompt_usd_per_1m)
    completion_rate, completion_valid = _parse_decimal(completion_usd_per_1m)
    rates_valid = prompt_valid and completion_valid
    configured = rates_valid and (prompt_rate > 0 or completion_rate > 0)
    if not rates_valid:
        reason = "invalid_rate"
    elif configured:
        reason = "configured"
    else:
        reason = "not_configured"

    return {
        "configured": configured,
        "reason": reason,
        "provider": provider.strip() or "unspecified",
        "currency": "USD",
        "prompt_usd_per_1m": _format_decimal(prompt_rate),
        "completion_usd_per_1m": _format_decimal(completion_rate),
        "external_billing_enabled": False,
    }


def estimate_query_cost_usd(
    *,
    prompt_tokens: int,
    completion_tokens: int,
    prompt_usd_per_1m: str,
    completion_usd_per_1m: str,
) -> str:
    prompt_rate, prompt_valid = _parse_decimal(prompt_usd_per_1m)
    completion_rate, completion_valid = _parse_decimal(completion_usd_per_1m)
    if not (prompt_valid and completion_valid):
        return "0"
    cost = (
        Decimal(max(0, int(prompt_tokens))) * prompt_rate
        + Decimal(max(0, int(completion_tokens))) * completion_rate
    ) / ONE_MILLION
    return format_usd(cost)


def summarize_query_cost(
    *,
    estimated_prompt_tokens: int,
    estimated_completion_tokens: int,
    provider_prompt_tokens: int = 0,
    provider_completion_tokens: int = 0,
    provider_usage_events: int = 0,
    total_events: int = 0,
    cost_prompt_tokens: int | None = None,
    cost_completion_tokens: int | None = None,
    provider: str,
    prompt_usd_per_1m: str,
    completion_usd_per_1m: str,
) -> dict[str, Any]:
    pricing = query_pricing_status(
        provider=provider,
        prompt_usd_per_1m=prompt_usd_per_1m,
        completion_usd_per_1m=completion_usd_per_1m,
    )
    use_provider_tokens = provider_usage_events > 0
    prompt_tokens = (
        cost_prompt_tokens
        if cost_prompt_tokens is not None
        else provider_prompt_tokens
        if use_provider_tokens
        else estimated_prompt_tokens
    )
    completion_tokens = (
        cost_completion_tokens
        if cost_completion_tokens is not None
        else provider_completion_tokens
        if use_provider_tokens
        else estimated_completion_tokens
    )
    if pricing["configured"]:
        if use_provider_tokens and total_events and provider_usage_events < total_events:
            cost_source = "mixed_tokens"
        elif use_provider_tokens:
            cost_source = "provider_tokens"
        else:
            cost_source = "estimated_tokens"
        estimated_cost_usd = estimate_query_cost_usd(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            prompt_usd_per_1m=prompt_usd_per_1m,
            completion_usd_per_1m=completion_usd_per_1m,
        )
    else:
        cost_source = "not_configured"
        estimated_cost_usd = "0"

    return {
        "estimated_cost_usd": estimated_cost_usd,
        "cost_source": cost_source,
        "cost_prompt_tokens": max(0, int(prompt_tokens)),
        "cost_completion_tokens": max(0, int(completion_tokens)),
        "pricing": pricing,
    }
