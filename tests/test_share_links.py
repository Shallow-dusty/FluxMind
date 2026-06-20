import json
import sqlite3

from src.share_links import (
    LocalShareLinkRegistry,
    share_link_registry_backend_status,
)


def test_share_link_registry_lifecycle_is_hash_only(tmp_path):
    registry = LocalShareLinkRegistry(tmp_path / "share_links.sqlite3")

    created = registry.create_link(
        workspace_id="lab-ws",
        created_by_user_id="owner-user",
        resource_kind="corpus_profile",
        resource_ref="private-profile-id",
        description="pilot share /private/hunter2 token=sk-secret-value",
        max_redemptions=1,
    )

    token = created["token"]
    link = created["share_link"]
    assert token.startswith("fms_")
    assert link["link_id"].startswith("share_")
    assert link["resource_kind"] == "corpus_profile"
    assert link["workspace_present"] is True
    assert link["workspace_fingerprint"]
    assert link["created_by_user_present"] is True
    assert link["created_by_user_fingerprint"]
    assert link["resource_ref_present"] is True
    assert link["resource_ref_fingerprint"]
    assert link["description_present"] is True
    assert link["description_fingerprint"]
    assert "workspace_id" not in link
    assert "lab-ws" not in json.dumps(link, sort_keys=True)
    assert "private-profile-id" not in json.dumps(link, sort_keys=True)
    assert "owner-user" not in json.dumps(link, sort_keys=True)
    assert "/private/hunter2" not in json.dumps(link, sort_keys=True)
    assert "sk-secret-value" not in json.dumps(link, sort_keys=True)

    listed = [record.to_public_dict() for record in registry.list_links()]
    rendered = json.dumps(listed, sort_keys=True)
    assert len(listed) == 1
    assert token not in rendered
    assert "workspace_id" not in rendered
    assert "lab-ws" not in rendered
    assert "private-profile-id" not in rendered
    assert "owner-user" not in rendered
    assert "/private/hunter2" not in rendered
    assert "sk-secret-value" not in rendered

    first_resolution = registry.resolve_token(token, record_redeem=True)
    assert first_resolution["valid"] is True
    assert first_resolution["share_link"]["redeem_count"] == 1
    assert token not in json.dumps(first_resolution, sort_keys=True)
    assert "workspace_id" not in json.dumps(first_resolution, sort_keys=True)
    assert "lab-ws" not in json.dumps(first_resolution, sort_keys=True)
    assert "private-profile-id" not in json.dumps(first_resolution, sort_keys=True)
    assert "owner-user" not in json.dumps(first_resolution, sort_keys=True)
    assert "/private/hunter2" not in json.dumps(first_resolution, sort_keys=True)
    assert "sk-secret-value" not in json.dumps(first_resolution, sort_keys=True)

    exhausted = registry.resolve_token(token)
    assert exhausted["valid"] is False
    assert exhausted["reason"] == "share_link_redemption_limit_exceeded"

    revoked = registry.revoke_link(link["link_id"])
    assert revoked is not None
    assert revoked.revoked_at is not None
    revoked_resolution = registry.resolve_token(token)
    assert revoked_resolution["valid"] is False
    assert revoked_resolution["reason"] == "share_link_revoked"


def test_share_link_registry_status_counts_terminal_states(tmp_path):
    registry = LocalShareLinkRegistry(tmp_path / "share_links.sqlite3")
    active = registry.create_link(
        workspace_id="lab-ws",
        created_by_user_id="owner-user",
        resource_ref="active-profile",
    )
    expired = registry.create_link(
        workspace_id="lab-ws",
        created_by_user_id="owner-user",
        resource_ref="expired-profile",
    )
    exhausted = registry.create_link(
        workspace_id="lab-ws",
        created_by_user_id="owner-user",
        resource_ref="exhausted-profile",
        max_redemptions=1,
    )
    revoked = registry.create_link(
        workspace_id="lab-ws",
        created_by_user_id="owner-user",
        resource_ref="revoked-profile",
    )

    with sqlite3.connect(registry.db_path) as conn:
        conn.execute(
            "UPDATE share_links SET expires_at = ? WHERE link_id = ?",
            ("2000-01-01T00:00:00+00:00", expired["share_link"]["link_id"]),
        )
    assert registry.resolve_token(exhausted["token"], record_redeem=True)["valid"] is True
    assert registry.revoke_link(revoked["share_link"]["link_id"]) is not None

    status = registry.status()
    listed = [record.to_public_dict() for record in registry.list_links()]
    all_links = [
        record.to_public_dict()
        for record in registry.list_links(include_revoked=True, limit=200)
    ]
    rendered = json.dumps({"listed": listed, "all": all_links, "status": status}, sort_keys=True)

    assert status["total_link_count"] == 4
    assert status["active_link_count"] == 1
    assert status["expired_link_count"] == 1
    assert status["revoked_link_count"] == 1
    assert status["redeem_count"] == 1
    assert all(link["link_id"] != revoked["share_link"]["link_id"] for link in listed)
    assert any(link["link_id"] == active["share_link"]["link_id"] for link in listed)
    assert any(link["exhausted"] is True for link in all_links)
    assert any(link["expired"] is True for link in all_links)
    assert any(link["active"] is False for link in all_links)
    assert "workspace_id" not in rendered
    assert "lab-ws" not in rendered
    assert "active-profile" not in rendered
    assert "expired-profile" not in rendered
    assert "exhausted-profile" not in rendered
    assert "revoked-profile" not in rendered


def test_share_link_registry_rejects_empty_resource_and_sanitizes_status_errors(tmp_path):
    db_path = tmp_path / "share_links.sqlite3"
    registry = LocalShareLinkRegistry(db_path)

    try:
        registry.create_link(
            workspace_id="lab-ws",
            created_by_user_id="owner-user",
            resource_ref="",
        )
    except ValueError as exc:
        assert str(exc) == "resource_ref_required"
    else:
        raise AssertionError("empty resource_ref should be rejected")

    db_path.write_text("not sqlite /private/hunter2 token=sk-secret-value", encoding="utf-8")
    status = share_link_registry_backend_status(backend="sqlite", db_path=db_path)
    rendered = json.dumps(status, sort_keys=True)

    assert status["configured"] is True
    assert status["supported"] is True
    assert status["available"] is False
    assert status["reason"] == "share_link_token_store_unavailable"
    assert str(db_path) not in rendered
    assert "/private/hunter2" not in rendered
    assert "sk-secret-value" not in rendered


def test_share_link_registry_status_reports_configured_sqlite(tmp_path):
    status = share_link_registry_backend_status(
        backend="sqlite",
        db_path=tmp_path / "share_links.sqlite3",
    )

    assert status["configured"] is True
    assert status["supported"] is True
    assert status["available"] is True
    assert status["active_link_count"] == 0
    assert status["secrets_exported"] is False
    assert status["share_tokens_exported"] is False
    assert status["share_urls_exported"] is False


def test_share_link_registry_status_reports_disabled_and_unsupported(tmp_path):
    disabled = share_link_registry_backend_status(
        backend="none",
        db_path=tmp_path / "share_links.sqlite3",
    )
    unsupported = share_link_registry_backend_status(
        backend="redis",
        db_path=tmp_path / "share_links.sqlite3",
    )

    assert disabled["available"] is False
    assert disabled["reason"] == "share_link_token_store_not_configured"
    assert unsupported["available"] is False
    assert unsupported["reason"] == "share_link_token_store_unavailable"
