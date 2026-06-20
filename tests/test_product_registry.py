import json

import pytest

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
    member = registry.add_member(workspace_id=workspace.workspace_id, user_id="member", role="member")

    membership = registry.workspace_for_user(user_id="member")
    requested = registry.workspace_for_user(user_id="member", workspace_id="ws-main")
    missing = registry.workspace_for_user(user_id="member", workspace_id="ws-other")

    assert member["workspace_id"] == "ws-main"
    assert member["user_id"] == "member"
    assert member["secrets_exported"] is False
    assert membership["workspace_id"] == "ws-main"
    assert membership["role"] == "member"
    assert requested["workspace_id"] == "ws-main"
    assert requested["secrets_exported"] is False
    assert missing is None


def test_product_registry_rejects_orphan_workspace_writes(tmp_path):
    registry = LocalProductRegistry(tmp_path / "product_registry.sqlite3")

    with pytest.raises(ValueError, match="Product workspace not found"):
        registry.add_member(workspace_id="missing-ws", user_id="member", role="member")
    with pytest.raises(ValueError, match="Product workspace not found"):
        registry.set_quota(
            workspace_id="missing-ws",
            metric="requests",
            limit_value=1,
            window_s=60,
        )
    with pytest.raises(ValueError, match="Product workspace not found"):
        registry.record_usage(
            workspace_id="missing-ws",
            user_id="member",
            metric="requests",
            amount=1,
        )
    with pytest.raises(ValueError, match="Product workspace not found"):
        registry.set_billing_account(workspace_id="missing-ws")
    with pytest.raises(ValueError, match="Product workspace not found"):
        registry.quota_decision(
            workspace_id="missing-ws",
            user_id="member",
            metric="requests",
        )

    status = registry.status()
    assert status["workspace_count"] == 0
    assert status["member_count"] == 0
    assert status["quota_limit_count"] == 0
    assert status["usage_event_count"] == 0
    assert status["billing_account_count"] == 0


def test_product_registry_rejects_orphan_usage_user(tmp_path):
    registry = LocalProductRegistry(tmp_path / "product_registry.sqlite3")
    workspace = registry.create_workspace(workspace_id="usage-ws", owner_user_id="owner")

    with pytest.raises(ValueError, match="Product user not found"):
        registry.record_usage(
            workspace_id=workspace.workspace_id,
            user_id="missing-user",
            metric="requests",
            amount=1,
        )
    with pytest.raises(ValueError, match="Product user not found"):
        registry.quota_decision(
            workspace_id=workspace.workspace_id,
            user_id="missing-user",
            metric="requests",
        )

    status = registry.status()
    assert status["workspace_count"] == 1
    assert status["user_count"] == 1
    assert status["usage_event_count"] == 0


def test_product_registry_permission_decision_enforces_local_roles(tmp_path):
    registry = LocalProductRegistry(tmp_path / "product_registry.sqlite3")
    workspace = registry.create_workspace(workspace_id="ws-main", owner_user_id="owner")
    registry.add_member(workspace_id=workspace.workspace_id, user_id="admin", role="admin")
    registry.add_member(workspace_id=workspace.workspace_id, user_id="member", role="member")
    registry.add_member(workspace_id=workspace.workspace_id, user_id="viewer", role="viewer")

    owner_corpus = registry.permission_decision(
        user_id="owner",
        workspace_id=workspace.workspace_id,
        action="corpus_write",
    )
    member_job = registry.permission_decision(
        user_id="member",
        workspace_id=workspace.workspace_id,
        action="job_submit",
    )
    viewer_job = registry.permission_decision(
        user_id="viewer",
        workspace_id=workspace.workspace_id,
        action="job_submit",
    )
    viewer_query = registry.permission_decision(
        user_id="viewer",
        workspace_id=workspace.workspace_id,
        action="query",
    )
    unsupported = registry.permission_decision(
        user_id="admin",
        workspace_id=workspace.workspace_id,
        action="billing/delete",
    )

    assert owner_corpus["allowed"] is True
    assert owner_corpus["required_roles"] == ["owner", "admin"]
    assert member_job["allowed"] is True
    assert viewer_job["allowed"] is False
    assert viewer_job["reason"] == "product_role_forbidden"
    assert viewer_query["allowed"] is True
    assert unsupported["allowed"] is False
    assert unsupported["reason"] == "unsupported_product_action"
    assert "billing/delete" not in json.dumps(unsupported, sort_keys=True)


def test_product_registry_workspace_detail_and_summary_are_no_secret(tmp_path):
    registry = LocalProductRegistry(tmp_path / "product_registry.sqlite3")
    workspace = registry.create_workspace(
        workspace_id="ws-detail",
        label="Detail workspace",
        owner_user_id="owner",
        owner_label="Owner",
    )
    registry.add_member(
        workspace_id=workspace.workspace_id,
        user_id="member",
        label="Member",
        role="member",
    )
    registry.set_quota(
        workspace_id=workspace.workspace_id,
        metric="requests",
        limit_value=10,
        window_s=60,
    )
    registry.record_usage(
        workspace_id=workspace.workspace_id,
        user_id="member",
        metric="requests",
        amount=2,
    )
    registry.set_billing_account(workspace_id=workspace.workspace_id)

    detail = registry.workspace_detail(workspace_id=workspace.workspace_id)
    summaries = registry.list_workspace_summaries()
    payload = json.dumps({"detail": detail, "summaries": summaries}, sort_keys=True)

    assert detail["workspace"]["workspace_id"] == "ws-detail"
    assert [member["role"] for member in detail["members"]] == ["owner", "member"]
    assert detail["quota_limits"][0]["limit_value"] == 10
    assert detail["billing"]["configured"] is True
    assert detail["usage_event_count"] == 1
    assert summaries[0]["member_count"] == 2
    assert summaries[0]["quota_limit_count"] == 1
    assert summaries[0]["usage_event_count"] == 1
    assert summaries[0]["billing_configured"] is True
    assert "admin-token" not in payload
    assert "sk-" not in payload


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
