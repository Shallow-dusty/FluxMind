import json

from src.product_activation_rehearsal import (
    PRODUCT_ACTIVATION_REHEARSAL_STATE_DIR,
    collect_product_activation_rehearsal,
    format_product_activation_rehearsal_markdown,
)


def test_product_activation_rehearsal_proves_local_activation_without_secrets(tmp_path):
    status = collect_product_activation_rehearsal(
        root=tmp_path / "rehearsal",
        generated_at="2026-06-19T00:00:00+08:00",
    )

    payload = json.dumps(status, ensure_ascii=False, sort_keys=True)
    assert status["mode"] == "product_activation_rehearsal"
    assert status["ok"] is True
    assert status["local_only"] is True
    assert status["content_exported"] is False
    assert status["secrets_exported"] is False
    assert status["paths_exported"] is False
    assert status["connectivity_checked"] is False
    assert status["api_key_lifecycle"]["active_key_count"] == 2
    assert status["api_key_lifecycle"]["revoked_key_count"] == 1
    assert status["api_key_lifecycle"]["revoked_verify_blocked"] is True
    assert status["rbac"]["owner_admin_write_allowed"] is True
    assert status["rbac"]["viewer_query_allowed"] is True
    assert status["rbac"]["viewer_job_submit_denied"] is True
    assert status["workspace_isolation"]["ok"] is True
    assert status["workspace_isolation"]["workspace_count"] == 2
    assert status["workspace_isolation"]["viewer_cross_workspace_query_denied"] is True
    assert status["workspace_isolation"]["owner_cross_workspace_admin_denied"] is True
    assert status["workspace_isolation"]["outsider_primary_workspace_query_denied"] is True
    assert status["workspace_isolation"]["outsider_own_workspace_query_allowed"] is True
    assert status["workspace_isolation"]["identifiers_exported"] is False
    assert status["workspace_isolation"]["share_links_enabled"] is False
    assert status["workspace_isolation"]["private_corpora_enabled"] is False
    assert status["quota"]["first_request_allowed"] is True
    assert status["quota"]["second_request_limited"] is True
    assert status["readiness"]["activation_ready"] is True
    assert status["readiness"]["activation_blockers"] == []
    assert "fmk_" not in payload
    assert str(tmp_path) not in payload
    assert "hunter2" not in payload
    for sensitive in (
        "rehearsal-owner",
        "Rehearsal owner",
        "rehearsal-viewer",
        "rehearsal-workspace",
        "rehearsal-other-workspace",
        "rehearsal-outsider",
        "api_keys.sqlite3",
        "product_registry.sqlite3",
        "local product activation rehearsal",
    ):
        assert sensitive not in payload


def test_product_activation_rehearsal_markdown_is_no_secret(tmp_path):
    status = collect_product_activation_rehearsal(
        root=tmp_path / "rehearsal",
        generated_at="2026-06-19T00:00:00+08:00",
    )

    markdown = format_product_activation_rehearsal_markdown(status)

    assert "# FluxMind Product Activation Rehearsal" in markdown
    assert "Activation ready: true" in markdown
    assert "Viewer job submit reason: product_role_forbidden" in markdown
    assert "Workspace Isolation" in markdown
    assert "Viewer cross-workspace query denied: true" in markdown
    assert "Owner cross-workspace admin denied: true" in markdown
    assert "Outsider own-workspace query allowed: true" in markdown
    assert "Share links enabled: false" in markdown
    assert "Private corpora enabled: false" in markdown
    assert "Second request reason: quota_exceeded" in markdown
    assert "fmk_" not in markdown
    assert str(tmp_path) not in markdown
    assert "hunter2" not in markdown
    for sensitive in (
        "rehearsal-owner",
        "Rehearsal owner",
        "rehearsal-viewer",
        "rehearsal-workspace",
        "rehearsal-other-workspace",
        "rehearsal-outsider",
        "api_keys.sqlite3",
        "product_registry.sqlite3",
        "local product activation rehearsal",
    ):
        assert sensitive not in markdown


def test_product_activation_rehearsal_reuses_root_without_state_leakage(tmp_path):
    root = tmp_path / "rehearsal"

    first = collect_product_activation_rehearsal(
        root=root,
        generated_at="2026-06-19T00:00:00+08:00",
    )
    second = collect_product_activation_rehearsal(
        root=root,
        generated_at="2026-06-19T00:00:01+08:00",
    )

    assert first["ok"] is True
    assert second["ok"] is True
    assert second["api_key_lifecycle"]["active_key_count"] == 2
    assert second["workspace_isolation"]["ok"] is True
    assert second["workspace_isolation"]["workspace_count"] == 2
    assert second["api_key_lifecycle"]["revoked_key_count"] == 1
    assert second["quota"]["first_request_allowed"] is True
    assert second["quota"]["second_request_limited"] is True


def test_product_activation_rehearsal_preserves_root_sqlite_files(tmp_path):
    root = tmp_path / "rehearsal"
    root.mkdir()
    existing_files = {
        root / "api_keys.sqlite3": "existing-api-db",
        root / "api_keys.sqlite3-wal": "existing-api-wal",
        root / "api_keys.sqlite3-shm": "existing-api-shm",
        root / "product_registry.sqlite3": "existing-product-db",
        root / "product_registry.sqlite3-wal": "existing-product-wal",
        root / "product_registry.sqlite3-shm": "existing-product-shm",
    }
    for path, content in existing_files.items():
        path.write_text(content, encoding="utf-8")

    status = collect_product_activation_rehearsal(
        root=root,
        generated_at="2026-06-19T00:00:00+08:00",
    )

    assert status["ok"] is True
    for path, content in existing_files.items():
        assert path.read_text(encoding="utf-8") == content
    assert (root / PRODUCT_ACTIVATION_REHEARSAL_STATE_DIR / "api_keys.sqlite3").is_file()
    assert (root / PRODUCT_ACTIVATION_REHEARSAL_STATE_DIR / "product_registry.sqlite3").is_file()
