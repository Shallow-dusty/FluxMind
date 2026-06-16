import json

from src.api_keys import LocalApiKeyRegistry
from src.product_registry import LocalProductRegistry
from src.product_readiness import (
    collect_product_readiness,
    format_product_readiness_markdown,
)


def test_product_readiness_reports_local_foundation_without_activation():
    status = collect_product_readiness(
        generated_at="2026-06-16T00:00:00+00:00",
        api_token_configured=False,
        api_access_audit_enabled=True,
        api_rate_limit_enabled=False,
        api_rate_limit_max_requests=300,
        api_rate_limit_window_s=60,
        query_cost_provider="",
        query_cost_prompt_usd_per_1m="0",
        query_cost_completion_usd_per_1m="0",
        identity_provider="none",
        api_key_registry_backend="none",
        quota_store_backend="none",
        billing_provider="none",
        billing_attribution_enabled=False,
        identity_quotas_billing_enabled=False,
    )

    assert status["mode"] == "product_readiness"
    assert status["local_foundation_ready"] is True
    assert status["activation_ready"] is False
    assert status["content_exported"] is False
    assert status["secrets_exported"] is False
    assert status["connectivity_checked"] is False
    assert status["blockers"]["local_foundation"] == []
    assert "multi_user_identity_not_configured" in status["blockers"]["activation"]
    assert "api_key_lifecycle_not_configured" in status["blockers"]["activation"]
    assert "identity_quota_store_not_configured" in status["blockers"]["activation"]
    assert "billing_provider_not_configured" in status["blockers"]["activation"]
    assert "billing_attribution_not_enabled" in status["blockers"]["activation"]
    assert "single_api_token_not_configured" in status["advisories"]
    assert "local_rate_limit_disabled" in status["advisories"]
    assert "product_quota_guard_disabled" in status["advisories"]
    assert "product_rbac_guard_disabled" in status["advisories"]
    assert status["summary"]["owner_metadata_supported"] is True


def test_product_readiness_can_report_activation_ready_without_secrets():
    status = collect_product_readiness(
        generated_at="2026-06-16T00:00:00+00:00",
        api_token_configured=True,
        api_access_audit_enabled=True,
        api_rate_limit_enabled=True,
        api_rate_limit_max_requests=120,
        api_rate_limit_window_s=60,
        query_cost_provider="lab-model",
        query_cost_prompt_usd_per_1m="0.10",
        query_cost_completion_usd_per_1m="0.20",
        identity_provider="oidc",
        api_key_registry_backend="postgres",
        quota_store_backend="redis",
        billing_provider="stripe",
        billing_attribution_enabled=True,
        identity_quotas_billing_enabled=True,
        product_quota_guard_enabled=True,
        product_rbac_guard_enabled=True,
    )

    payload = json.dumps(status, ensure_ascii=False, sort_keys=True)
    assert status["local_foundation_ready"] is True
    assert status["activation_ready"] is True
    assert status["identity_quotas_billing_enabled"] is True
    assert status["blockers"]["activation"] == []
    assert status["checks"]["identity_provider"]["backend"] == "oidc"
    assert status["checks"]["billing_provider"]["backend"] == "stripe"
    assert "api_key" in payload
    assert "hunter2" not in payload


def test_product_readiness_checks_local_sqlite_api_key_registry(tmp_path, monkeypatch):
    registry_path = tmp_path / "api_keys.sqlite3"
    token = LocalApiKeyRegistry(registry_path).create_key(owner_id="lab-product")["token"]
    monkeypatch.setattr("src.api_keys.config.API_KEY_REGISTRY_FILE", registry_path)

    status = collect_product_readiness(
        generated_at="2026-06-16T00:00:00+00:00",
        api_access_audit_enabled=True,
        api_rate_limit_enabled=True,
        api_rate_limit_max_requests=120,
        api_rate_limit_window_s=60,
        identity_provider="none",
        api_key_registry_backend="sqlite",
        quota_store_backend="none",
        billing_provider="none",
        billing_attribution_enabled=False,
    )

    payload = json.dumps(status, ensure_ascii=False, sort_keys=True)
    assert status["checks"]["api_key_registry"]["backend"] == "sqlite"
    assert status["checks"]["api_key_registry"]["available"] is True
    assert status["summary"]["api_key_lifecycle_available"] is True
    assert "api_key_lifecycle_not_configured" not in status["blockers"]["activation"]
    assert "api_key_registry_unavailable" not in status["blockers"]["activation"]
    assert "identity_quota_store_not_configured" in status["blockers"]["activation"]
    assert token not in payload


def test_product_readiness_can_use_local_product_registry(tmp_path, monkeypatch):
    api_key_path = tmp_path / "api_keys.sqlite3"
    product_registry_path = tmp_path / "product_registry.sqlite3"
    LocalApiKeyRegistry(api_key_path).create_key(owner_id="lab-product")
    product_registry = LocalProductRegistry(product_registry_path)
    workspace = product_registry.create_workspace(
        workspace_id="lab-workspace",
        owner_user_id="lab-owner",
    )
    product_registry.set_quota(
        workspace_id=workspace.workspace_id,
        metric="requests",
        limit_value=100,
        window_s=3600,
    )
    product_registry.set_billing_account(workspace_id=workspace.workspace_id)
    monkeypatch.setattr("src.api_keys.config.API_KEY_REGISTRY_FILE", api_key_path)
    monkeypatch.setattr("src.product_registry.config.PRODUCT_REGISTRY_FILE", product_registry_path)

    status = collect_product_readiness(
        generated_at="2026-06-16T00:00:00+00:00",
        api_token_configured=True,
        api_access_audit_enabled=True,
        api_rate_limit_enabled=True,
        api_rate_limit_max_requests=120,
        api_rate_limit_window_s=60,
        query_cost_provider="lab-model",
        query_cost_prompt_usd_per_1m="0.10",
        query_cost_completion_usd_per_1m="0.20",
        identity_provider="local-registry",
        api_key_registry_backend="sqlite",
        quota_store_backend="sqlite",
        billing_provider="local-ledger",
        billing_attribution_enabled=True,
        identity_quotas_billing_enabled=True,
        product_quota_guard_enabled=True,
        product_rbac_guard_enabled=True,
        product_registry_backend="sqlite",
    )

    payload = json.dumps(status, ensure_ascii=False, sort_keys=True)
    assert status["local_foundation_ready"] is True
    assert status["activation_ready"] is True
    assert status["blockers"]["activation"] == []
    assert status["summary"]["product_registry_available"] is True
    assert status["summary"]["workspace_identity_available"] is True
    assert status["summary"]["quota_store_available"] is True
    assert status["summary"]["billing_ledger_available"] is True
    assert status["summary"]["product_quota_guard_enabled"] is True
    assert status["summary"]["product_rbac_guard_enabled"] is True
    assert status["checks"]["identity_provider"]["backend"] == "local-registry"
    assert status["checks"]["identity_provider"]["workspace_count"] == 1
    assert status["checks"]["quota_store"]["quota_limit_count"] == 1
    assert status["checks"]["billing_provider"]["billing_account_count"] == 1
    assert "hunter2" not in payload


def test_product_readiness_blocks_local_product_registry_when_unavailable(tmp_path, monkeypatch):
    bad_path = tmp_path / "product_registry.sqlite3"
    bad_path.write_text("not sqlite", encoding="utf-8")
    monkeypatch.setattr("src.product_registry.config.PRODUCT_REGISTRY_FILE", bad_path)

    status = collect_product_readiness(
        generated_at="2026-06-16T00:00:00+00:00",
        api_access_audit_enabled=True,
        api_rate_limit_enabled=True,
        api_rate_limit_max_requests=120,
        api_rate_limit_window_s=60,
        identity_provider="local-registry",
        quota_store_backend="sqlite",
        billing_provider="local-ledger",
        billing_attribution_enabled=True,
        product_registry_backend="sqlite",
    )

    assert "identity_provider_unavailable" in status["blockers"]["activation"]
    assert "quota_store_unavailable" in status["blockers"]["activation"]
    assert "billing_provider_unavailable" in status["blockers"]["activation"]


def test_product_readiness_blocks_enabled_runtime_without_quota_guard():
    status = collect_product_readiness(
        generated_at="2026-06-16T00:00:00+00:00",
        api_access_audit_enabled=True,
        api_rate_limit_enabled=True,
        api_rate_limit_max_requests=120,
        api_rate_limit_window_s=60,
        identity_provider="oidc",
        api_key_registry_backend="postgres",
        quota_store_backend="redis",
        billing_provider="stripe",
        billing_attribution_enabled=True,
        identity_quotas_billing_enabled=True,
        product_quota_guard_enabled=False,
        product_rbac_guard_enabled=True,
    )

    assert "product_quota_guard_not_enabled" in status["blockers"]["activation"]
    assert status["summary"]["product_quota_guard_enabled"] is False


def test_product_readiness_blocks_enabled_runtime_without_rbac_guard():
    status = collect_product_readiness(
        generated_at="2026-06-16T00:00:00+00:00",
        api_access_audit_enabled=True,
        api_rate_limit_enabled=True,
        api_rate_limit_max_requests=120,
        api_rate_limit_window_s=60,
        identity_provider="oidc",
        api_key_registry_backend="postgres",
        quota_store_backend="redis",
        billing_provider="stripe",
        billing_attribution_enabled=True,
        identity_quotas_billing_enabled=True,
        product_quota_guard_enabled=True,
        product_rbac_guard_enabled=False,
    )

    assert "product_rbac_guard_not_enabled" in status["blockers"]["activation"]
    assert status["summary"]["product_rbac_guard_enabled"] is False


def test_product_readiness_sanitizes_secret_like_backend_values():
    status = collect_product_readiness(
        generated_at="2026-06-16T00:00:00+00:00",
        api_access_audit_enabled=True,
        api_rate_limit_max_requests=300,
        api_rate_limit_window_s=60,
        identity_provider="oidc:hunter2@example.test",
        api_key_registry_backend="postgres://hunter2@example.test/db",
        quota_store_backend="redis://:hunter2@example.test/0",
        billing_provider="stripe:hunter2",
        billing_attribution_enabled=True,
    )

    payload = json.dumps(status, ensure_ascii=False, sort_keys=True)
    assert "identity_provider_unsupported" in status["blockers"]["activation"]
    assert "api_key_registry_unsupported" in status["blockers"]["activation"]
    assert "quota_store_unsupported" in status["blockers"]["activation"]
    assert "billing_provider_unsupported" in status["blockers"]["activation"]
    assert "hunter2" not in payload
    assert "example.test" not in payload
    assert status["checks"]["identity_provider"]["backend"] == "custom"


def test_product_readiness_blocks_invalid_local_foundation():
    status = collect_product_readiness(
        generated_at="2026-06-16T00:00:00+00:00",
        api_access_audit_enabled=False,
        api_rate_limit_max_requests=0,
        api_rate_limit_window_s=0,
        query_cost_prompt_usd_per_1m="bad-rate",
        query_cost_completion_usd_per_1m="0",
        owner_metadata_supported=False,
    )

    assert status["local_foundation_ready"] is False
    assert "api_access_audit_disabled" in status["blockers"]["local_foundation"]
    assert "local_rate_limit_config_invalid" in status["blockers"]["local_foundation"]
    assert "owner_metadata_contract_missing" in status["blockers"]["local_foundation"]
    assert "query_cost_rates_invalid" in status["blockers"]["local_foundation"]


def test_format_product_readiness_markdown_is_no_secret():
    status = collect_product_readiness(
        generated_at="2026-06-16T00:00:00+00:00",
        api_token_configured=True,
        identity_provider="oidc:hunter2@example.test",
        api_key_registry_backend="postgres://hunter2@example.test/db",
        quota_store_backend="redis://:hunter2@example.test/0",
        billing_provider="stripe:hunter2",
    )

    markdown = format_product_readiness_markdown(status)

    assert "# FluxMind Product Readiness" in markdown
    assert "Local foundation ready:" in markdown
    assert "Activation ready:" in markdown
    assert "Product quota guard enabled:" in markdown
    assert "Product RBAC guard enabled:" in markdown
    assert "hunter2" not in markdown
    assert "example.test" not in markdown
