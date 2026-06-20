import json

from src.provider_guard import provider_quota_guard_decision, provider_quota_policy


def test_provider_quota_guard_disabled_allows_without_secrets():
    decision = provider_quota_guard_decision(
        operation="rag_generation",
        provider="https://provider.example.test/hunter2",
        estimated_prompt_tokens=999999,
        requested_completion_tokens=999999,
        provider_quota_guard_enabled=False,
        max_prompt_tokens=1,
        max_completion_tokens=1,
    )

    payload = json.dumps(decision, ensure_ascii=False, sort_keys=True)
    assert decision["allowed"] is True
    assert decision["limited"] is False
    assert decision["reason"] == "provider_quota_guard_disabled"
    assert decision["content_exported"] is False
    assert decision["secrets_exported"] is False
    assert "hunter2" not in payload
    assert "example.test" not in payload
    assert "sk-test" not in payload
    assert decision["provider"] == "unspecified"


def test_provider_quota_guard_blocks_prompt_token_limit():
    decision = provider_quota_guard_decision(
        operation="rag_generation",
        provider="test-provider",
        estimated_prompt_tokens=101,
        requested_completion_tokens=10,
        provider_quota_guard_enabled=True,
        max_prompt_tokens=100,
        max_completion_tokens=50,
    )

    assert decision["allowed"] is False
    assert decision["limited"] is True
    assert decision["reason"] == "provider_prompt_token_limit_exceeded"
    assert decision["status_code"] == 429
    assert decision["estimated_prompt_tokens"] == 101
    assert decision["max_prompt_tokens_per_request"] == 100


def test_provider_quota_guard_blocks_completion_token_limit():
    decision = provider_quota_guard_decision(
        operation="rag_generation",
        provider="test-provider",
        estimated_prompt_tokens=50,
        requested_completion_tokens=51,
        provider_quota_guard_enabled=True,
        max_prompt_tokens=100,
        max_completion_tokens=50,
    )

    assert decision["allowed"] is False
    assert decision["limited"] is True
    assert decision["reason"] == "provider_completion_token_limit_exceeded"
    assert decision["status_code"] == 429


def test_provider_quota_guard_blocks_cost_limit():
    decision = provider_quota_guard_decision(
        operation="rag_generation",
        provider="test-provider",
        estimated_prompt_tokens=1_000_000,
        requested_completion_tokens=1_000_000,
        provider_quota_guard_enabled=True,
        max_prompt_tokens=2_000_000,
        max_completion_tokens=2_000_000,
        max_cost_usd="1",
        prompt_usd_per_1m="1",
        completion_usd_per_1m="1",
    )

    assert decision["allowed"] is False
    assert decision["limited"] is True
    assert decision["reason"] == "provider_cost_limit_exceeded"
    assert decision["estimated_cost_usd"] == "2"


def test_provider_quota_guard_requires_pricing_when_cost_limit_enabled():
    decision = provider_quota_guard_decision(
        operation="rag_generation",
        provider="test-provider",
        estimated_prompt_tokens=10,
        requested_completion_tokens=10,
        provider_quota_guard_enabled=True,
        max_prompt_tokens=100,
        max_completion_tokens=100,
        max_cost_usd="1",
        prompt_usd_per_1m="0",
        completion_usd_per_1m="0",
    )

    assert decision["allowed"] is False
    assert decision["limited"] is False
    assert decision["reason"] == "provider_cost_pricing_not_configured"
    assert decision["status_code"] == 503


def test_provider_quota_guard_sanitizes_invalid_cost_limit():
    for invalid_limit in ("NaN", "sNaN", "Infinity", "1e999999"):
        decision = provider_quota_guard_decision(
            operation="rag_generation",
            provider="test-provider",
            estimated_prompt_tokens=10,
            requested_completion_tokens=10,
            provider_quota_guard_enabled=True,
            max_prompt_tokens=100,
            max_completion_tokens=100,
            max_cost_usd=invalid_limit,
            prompt_usd_per_1m="1",
            completion_usd_per_1m="1",
        )

        assert decision["allowed"] is True
        assert decision["limited"] is False
        assert decision["reason"] == "allowed"
        assert decision["max_cost_usd_per_request"] == "0"
        assert decision["cost_limit_configured"] is False


def test_provider_quota_guard_handles_nonfinite_pricing_when_cost_limit_enabled():
    decision = provider_quota_guard_decision(
        operation="rag_generation",
        provider="test-provider",
        estimated_prompt_tokens=10,
        requested_completion_tokens=10,
        provider_quota_guard_enabled=True,
        max_prompt_tokens=100,
        max_completion_tokens=100,
        max_cost_usd="1",
        prompt_usd_per_1m="NaN",
        completion_usd_per_1m="1",
    )

    assert decision["allowed"] is False
    assert decision["limited"] is False
    assert decision["reason"] == "provider_cost_pricing_not_configured"
    assert decision["pricing_configured"] is False


def test_provider_quota_policy_reports_no_secret_thresholds():
    policy = provider_quota_policy(
        provider_quota_guard_enabled=True,
        max_prompt_tokens=100,
        max_completion_tokens=50,
        max_cost_usd="0.25",
        pricing_provider="provider:hunter2@example.test",
        prompt_usd_per_1m="1",
        completion_usd_per_1m="2",
    )

    payload = json.dumps(policy, ensure_ascii=False, sort_keys=True)
    assert policy["enabled"] is True
    assert policy["max_prompt_tokens_per_request"] == 100
    assert policy["max_completion_tokens_per_request"] == 50
    assert policy["max_cost_usd_per_request"] == "0.25"
    assert policy["cost_limit_configured"] is True
    assert policy["pricing_configured"] is True
    assert policy["content_exported"] is False
    assert policy["secrets_exported"] is False
    assert "hunter2" not in payload
