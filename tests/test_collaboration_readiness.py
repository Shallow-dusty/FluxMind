import json

from src.collaboration_readiness import (
    collect_collaboration_readiness,
    format_collaboration_readiness_markdown,
)


def test_collaboration_readiness_defaults_to_safe_no_secret_gate(tmp_path):
    status = collect_collaboration_readiness(
        generated_at="2026-06-20T00:00:00+00:00",
        product_registry_file=tmp_path / "product_registry.sqlite3",
    )

    payload = json.dumps(status, ensure_ascii=False, sort_keys=True)
    assert status["mode"] == "collaboration_readiness"
    assert status["ok"] is True
    assert status["local_foundation_ready"] is True
    assert status["safe_default_ready"] is True
    assert status["activation_ready"] is False
    assert status["local_only"] is True
    assert status["content_exported"] is False
    assert status["secrets_exported"] is False
    assert status["paths_exported"] is False
    assert status["identifiers_exported"] is False
    assert status["share_tokens_exported"] is False
    assert status["share_urls_exported"] is False
    assert status["summary"]["private_corpora_enabled"] is False
    assert status["summary"]["share_links_enabled"] is False
    assert status["summary"]["policy_scenario_count"] == 13
    assert status["summary"]["policy_denied_count"] == 13
    assert "private_corpora_disabled" in status["blockers"]["activation"]
    assert "share_links_disabled" in status["blockers"]["activation"]
    assert status["checks"]["private_corpus_policy"]["cross_workspace_read_denied"] is True
    assert status["checks"]["share_link_policy"]["anonymous_redeem_allowed"] is False
    assert status["checks"]["share_link_policy"]["anonymous_redeem_reason"] == "share_links_disabled"
    for sensitive in (
        "hunter2",
        "fmk_",
        "share-token",
        "workspace-",
        "user-",
        "corpus-",
        "https://",
        str(tmp_path),
    ):
        assert sensitive not in payload


def test_collaboration_readiness_can_pass_activation_with_local_prerequisites(tmp_path):
    status = collect_collaboration_readiness(
        generated_at="2026-06-20T00:00:00+00:00",
        private_corpora_enabled=True,
        share_links_enabled=True,
        product_rbac_guard_enabled=True,
        product_registry_backend="sqlite",
        product_registry_file=tmp_path / "product_registry.sqlite3",
        share_link_token_store_backend="sqlite",
        share_link_token_store_file=tmp_path / "share_links.sqlite3",
    )

    assert status["ok"] is True
    assert status["safe_default_ready"] is False
    assert status["activation_ready"] is True
    assert status["blockers"]["activation"] == []
    private = status["checks"]["private_corpus_policy"]
    assert private["ready"] is True
    assert {item["role"] for item in private["read"] if item["allowed"]} == {
        "owner",
        "admin",
        "member",
        "viewer",
    }
    assert {item["role"] for item in private["write"] if item["allowed"]} == {
        "owner",
        "admin",
    }
    share = status["checks"]["share_link_policy"]
    assert share["ready"] is True
    assert {item["role"] for item in share["create"] if item["allowed"]} == {
        "owner",
        "admin",
    }
    assert share["anonymous_redeem_allowed"] is False
    assert share["anonymous_redeem_reason"] == "share_link_token_required"


def test_collaboration_readiness_blocks_enabled_features_without_guards(tmp_path):
    status = collect_collaboration_readiness(
        generated_at="2026-06-20T00:00:00+00:00",
        private_corpora_enabled=True,
        share_links_enabled=True,
        product_rbac_guard_enabled=False,
        product_registry_backend="none",
        product_registry_file=tmp_path / "product_registry.sqlite3",
    )

    assert status["ok"] is False
    assert status["local_foundation_ready"] is True
    assert status["safe_default_ready"] is False
    assert status["activation_ready"] is False
    assert "product_registry_not_configured" in status["blockers"]["activation"]
    assert "product_rbac_guard_not_enabled" in status["blockers"]["activation"]
    assert "share_link_token_store_not_configured" in status["blockers"]["activation"]
    assert all(
        item["reason"] == "collaboration_guard_not_ready"
        for item in status["checks"]["private_corpus_policy"]["read"]
    )


def test_collaboration_readiness_reports_share_token_store_reason_after_rbac_ready(tmp_path):
    status = collect_collaboration_readiness(
        generated_at="2026-06-20T00:00:00+00:00",
        private_corpora_enabled=True,
        share_links_enabled=True,
        product_rbac_guard_enabled=True,
        product_registry_backend="sqlite",
        product_registry_file=tmp_path / "product_registry.sqlite3",
        share_link_token_store_file=tmp_path / "share_links.sqlite3",
    )

    assert status["ok"] is False
    assert status["checks"]["private_corpus_policy"]["ready"] is True
    assert status["checks"]["share_link_policy"]["ready"] is False
    assert status["checks"]["share_link_token_store"]["reason"] == (
        "share_link_token_store_not_configured"
    )
    assert {
        item["reason"] for item in status["checks"]["share_link_policy"]["create"]
    } == {"share_link_token_store_not_configured"}
    assert status["checks"]["share_link_policy"]["anonymous_redeem_reason"] == (
        "share_link_token_store_not_configured"
    )


def test_collaboration_readiness_reports_unsupported_share_token_store(tmp_path):
    status = collect_collaboration_readiness(
        generated_at="2026-06-20T00:00:00+00:00",
        private_corpora_enabled=True,
        share_links_enabled=True,
        product_rbac_guard_enabled=True,
        product_registry_backend="sqlite",
        product_registry_file=tmp_path / "product_registry.sqlite3",
        share_link_token_store_backend="s3",
        share_link_token_store_file=tmp_path / "share_links.sqlite3",
    )

    assert "share_link_token_store_unavailable" in status["blockers"]["activation"]
    assert status["checks"]["share_link_token_store"]["reason"] == (
        "share_link_token_store_unavailable"
    )
    assert {
        item["reason"] for item in status["checks"]["share_link_policy"]["create"]
    } == {"share_link_token_store_unavailable"}


def test_collaboration_readiness_markdown_is_no_secret(tmp_path):
    status = collect_collaboration_readiness(
        generated_at="2026-06-20T00:00:00+00:00",
        product_registry_file=tmp_path / "product_registry.sqlite3",
    )

    markdown = format_collaboration_readiness_markdown(status)

    assert "# FluxMind Collaboration Readiness" in markdown
    assert "Safe default ready: true" in markdown
    assert "Activation ready: false" in markdown
    assert "Private corpora enabled: false" in markdown
    assert "Share links enabled: false" in markdown
    assert "Anonymous redeem reason: share_links_disabled" in markdown
    assert "Activation: private_corpora_disabled" in markdown
    assert "Share URLs exported: false" in markdown
    for sensitive in (
        "hunter2",
        "fmk_",
        "share-token",
        "workspace-",
        "user-",
        "corpus-",
        str(tmp_path),
    ):
        assert sensitive not in markdown
