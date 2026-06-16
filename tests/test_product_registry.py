import json

from src.product_registry import LocalProductRegistry, product_registry_backend_status


def test_local_product_registry_bootstraps_workspace_quota_usage_and_billing(tmp_path):
    db_path = tmp_path / "product_registry.sqlite3"
    registry = LocalProductRegistry(db_path)

    workspace = registry.create_workspace(
        workspace_id="lab-ws",
        label="Lab Workspace",
        owner_user_id="lab-owner",
        owner_label="Lab Owner",
    )
    quota = registry.set_quota(
        workspace_id=workspace.workspace_id,
        metric="requests",
        limit_value=120,
        window_s=3600,
    )
    usage = registry.record_usage(
        workspace_id=workspace.workspace_id,
        user_id="lab-owner",
        metric="requests",
        amount=3,
        source="test",
    )
    billing = registry.set_billing_account(workspace_id=workspace.workspace_id)
    status = registry.status()

    payload = json.dumps(status, ensure_ascii=False, sort_keys=True)
    assert workspace.workspace_id == "lab-ws"
    assert workspace.owner_user_id == "lab-owner"
    assert quota["limit_value"] == 120
    assert usage["amount"] == 3
    assert billing["billing_mode"] == "local-ledger"
    assert status["available"] is True
    assert status["user_count"] == 1
    assert status["workspace_count"] == 1
    assert status["member_count"] == 1
    assert status["quota_limit_count"] == 1
    assert status["usage_event_count"] == 1
    assert status["billing_account_count"] == 1
    assert status["billing_attribution_count"] == 1
    assert "hunter2" not in payload


def test_product_registry_backend_status_is_no_secret_and_handles_disabled_backend(tmp_path):
    disabled = product_registry_backend_status(backend="none", db_path=tmp_path / "registry.sqlite3")
    assert disabled["configured"] is False
    assert disabled["available"] is False
    assert disabled["reason"] == "product_registry_not_configured"
    assert disabled["secrets_exported"] is False
    assert disabled["paths_exported"] is False


def test_product_registry_backend_status_reports_sqlite_available(tmp_path):
    db_path = tmp_path / "product_registry.sqlite3"
    LocalProductRegistry(db_path).create_workspace(workspace_id="ws", owner_user_id="owner")

    status = product_registry_backend_status(backend="sqlite", db_path=db_path)

    assert status["configured"] is True
    assert status["supported"] is True
    assert status["available"] is True
    assert status["reason"] == "available"
    assert status["workspace_count"] == 1
    assert status["content_exported"] is False
    assert status["secrets_exported"] is False


def test_product_registry_sanitizes_invalid_ids(tmp_path):
    registry = LocalProductRegistry(tmp_path / "product_registry.sqlite3")

    workspace = registry.create_workspace(
        workspace_id="ws/secret@example.test",
        owner_user_id="owner/secret@example.test",
    )

    assert "secret" not in workspace.workspace_id
    assert "example.test" not in workspace.owner_user_id


def test_product_registry_finds_workspace_for_member(tmp_path):
    registry = LocalProductRegistry(tmp_path / "product_registry.sqlite3")
    workspace = registry.create_workspace(workspace_id="ws-main", owner_user_id="owner")
    registry.add_member(workspace_id=workspace.workspace_id, user_id="member", role="member")

    membership = registry.workspace_for_user(user_id="member")
    requested = registry.workspace_for_user(user_id="member", workspace_id="ws-main")
    missing = registry.workspace_for_user(user_id="member", workspace_id="ws-other")

    assert membership["workspace_id"] == "ws-main"
    assert membership["role"] == "member"
    assert requested["workspace_id"] == "ws-main"
    assert requested["secrets_exported"] is False
    assert missing is None


def test_product_registry_quota_decision_records_and_blocks_over_limit(tmp_path):
    registry = LocalProductRegistry(tmp_path / "product_registry.sqlite3")
    workspace = registry.create_workspace(workspace_id="quota-ws", owner_user_id="owner")
    registry.set_quota(
        workspace_id=workspace.workspace_id,
        metric="requests",
        limit_value=2,
        window_s=3600,
    )

    first = registry.quota_decision(
        workspace_id=workspace.workspace_id,
        user_id="owner",
        metric="requests",
        amount=1,
        source="test",
    )
    second = registry.quota_decision(
        workspace_id=workspace.workspace_id,
        user_id="owner",
        metric="requests",
        amount=1,
        source="test",
    )
    third = registry.quota_decision(
        workspace_id=workspace.workspace_id,
        user_id="owner",
        metric="requests",
        amount=1,
        source="test",
    )
    status = registry.status()

    assert first["allowed"] is True
    assert first["remaining"] == 1
    assert second["allowed"] is True
    assert second["remaining"] == 0
    assert third["allowed"] is False
    assert third["limited"] is True
    assert third["reason"] == "quota_exceeded"
    assert third["usage_event_id"] is None
    assert status["usage_event_count"] == 2
    assert "hunter2" not in json.dumps(third, sort_keys=True)


def test_product_registry_quota_decision_allows_and_records_without_limit(tmp_path):
    registry = LocalProductRegistry(tmp_path / "product_registry.sqlite3")
    workspace = registry.create_workspace(workspace_id="unlimited-ws", owner_user_id="owner")

    decision = registry.quota_decision(
        workspace_id=workspace.workspace_id,
        user_id="owner",
        metric="requests",
        amount=3,
        source="test",
    )

    assert decision["allowed"] is True
    assert decision["quota_configured"] is False
    assert decision["reason"] == "quota_not_configured"
    assert decision["usage_event_id"]
    assert registry.status()["usage_event_count"] == 1
