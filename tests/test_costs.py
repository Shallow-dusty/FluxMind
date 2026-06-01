from src.costs import summarize_query_cost


def test_summarize_query_cost_uses_provider_tokens_when_available():
    summary = summarize_query_cost(
        estimated_prompt_tokens=30,
        estimated_completion_tokens=60,
        provider_prompt_tokens=4,
        provider_completion_tokens=8,
        provider_usage_events=1,
        provider="mimo-deepseek",
        prompt_usd_per_1m="2",
        completion_usd_per_1m="5",
    )

    assert summary["estimated_cost_usd"] == "0.000048"
    assert summary["cost_source"] == "provider_tokens"
    assert summary["cost_prompt_tokens"] == 4
    assert summary["cost_completion_tokens"] == 8
    assert summary["pricing"] == {
        "configured": True,
        "reason": "configured",
        "provider": "mimo-deepseek",
        "currency": "USD",
        "prompt_usd_per_1m": "2",
        "completion_usd_per_1m": "5",
        "external_billing_enabled": False,
    }


def test_summarize_query_cost_stays_zero_until_rates_are_configured():
    summary = summarize_query_cost(
        estimated_prompt_tokens=30,
        estimated_completion_tokens=60,
        provider="local",
        prompt_usd_per_1m="0",
        completion_usd_per_1m="0",
    )

    assert summary["estimated_cost_usd"] == "0"
    assert summary["cost_source"] == "not_configured"
    assert summary["pricing"]["configured"] is False
    assert summary["pricing"]["external_billing_enabled"] is False


def test_summarize_query_cost_supports_mixed_provider_and_estimated_tokens():
    summary = summarize_query_cost(
        estimated_prompt_tokens=30,
        estimated_completion_tokens=60,
        provider_prompt_tokens=4,
        provider_completion_tokens=8,
        provider_usage_events=1,
        total_events=2,
        cost_prompt_tokens=14,
        cost_completion_tokens=28,
        provider="mimo-deepseek",
        prompt_usd_per_1m="2",
        completion_usd_per_1m="5",
    )

    assert summary["estimated_cost_usd"] == "0.000168"
    assert summary["cost_source"] == "mixed_tokens"
    assert summary["cost_prompt_tokens"] == 14
    assert summary["cost_completion_tokens"] == 28
