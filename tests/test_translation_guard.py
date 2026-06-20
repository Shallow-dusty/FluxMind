from pathlib import Path


APP_SOURCE = Path("app.py").read_text(encoding="utf-8")


def test_browser_translation_guard_is_installed():
    assert '<meta name="google" content="notranslate">' in APP_SOURCE
    assert 'node.setAttribute("translate", "no")' in APP_SOURCE
    assert 'node.classList.add("notranslate")' in APP_SOURCE
    assert "MutationObserver" in APP_SOURCE


def test_streamlit_write_stream_is_not_used_for_chat_streaming():
    assert "st.write_stream" not in APP_SOURCE
    assert "render_streaming_response(prompt, answer_mode=st.session_state.answer_mode)" in APP_SOURCE


def test_streamlit_local_job_panel_is_installed():
    assert "get_async_job_manager" in APP_SOURCE
    assert "enqueue_index_rebuild(selected)" in APP_SOURCE
    assert "enqueue_mock_image" in APP_SOURCE
    assert "mock_image_template" in APP_SOURCE
    assert "enqueue_local_python" in APP_SOURCE
    assert "PYTHON_EXECUTION_TEMPLATES" in APP_SOURCE
    assert "python_execution_template" in APP_SOURCE
    assert "OCTAVE_EXECUTION_TEMPLATES" in APP_SOURCE
    assert "octave_execution_template" in APP_SOURCE
    assert "cancel_job" in APP_SOURCE
    assert "job_search" in APP_SOURCE
    assert "job_status_filter" in APP_SOURCE
    assert "job_kind_filter" in APP_SOURCE
    assert "retry_job" in APP_SOURCE
    assert "schedule_retry" in APP_SOURCE
    assert "LocalJobRunner().retry" in APP_SOURCE
    assert "get_async_job_manager().schedule_retry" in APP_SOURCE
    assert "set_active_paper_source_paths(selected)" in APP_SOURCE


def test_streamlit_latest_jobs_uses_no_secret_summary():
    assert "job_search_projection" in APP_SOURCE
    assert "job_sidebar_summary(job)" in APP_SOURCE
    assert "st.json(job.result)" not in APP_SOURCE
    assert "st.json(job.error)" not in APP_SOURCE


def test_streamlit_artifact_gallery_is_installed():
    assert "LocalArtifactRegistry" in APP_SOURCE
    assert "render_latest_artifacts()" in APP_SOURCE
    assert "artifact_search" in APP_SOURCE
    assert "artifact_kind_filter" in APP_SOURCE
    assert "artifact_job_kind_filter" in APP_SOURCE
    assert "st.download_button" in APP_SOURCE
    assert "artifact_to_public_dict" in APP_SOURCE
    assert "job_artifact_to_public_dict" in APP_SOURCE
    assert "safe_artifact_download_filename" in APP_SOURCE
    assert 'st.code(artifact.get("uri", "")' not in APP_SOURCE
    assert "st.code(artifact.uri" not in APP_SOURCE


def test_streamlit_admin_status_panel_is_installed():
    assert "collect_admin_status" in APP_SOURCE
    assert "collect_retention_preview" in APP_SOURCE
    assert "apply_retention_delete" in APP_SOURCE
    assert "list_runtime_events" in APP_SOURCE
    assert "render_admin_status()" in APP_SOURCE
    assert "render_retention_preview()" in APP_SOURCE
    assert "render_runtime_events()" in APP_SOURCE
    assert "admin_status" in APP_SOURCE
    assert "retention_preview" in APP_SOURCE
    assert "retention_delete" in APP_SOURCE
    assert "runtime_events" in APP_SOURCE
    assert "event_kind_filter" in APP_SOURCE
    assert "delete_enabled" in APP_SOURCE
    assert "worker_leases" in APP_SOURCE
    assert "status_cost_pricing" in APP_SOURCE
    assert "status_storage" in APP_SOURCE
    assert "status_storage_inventory" in APP_SOURCE
    assert "status_runtime_manifest" in APP_SOURCE
    assert "download_runtime_manifest" in APP_SOURCE
    assert "collect_runtime_restore_check" in APP_SOURCE
    assert "format_runtime_restore_check_markdown" in APP_SOURCE
    assert "runtime_restore_manifest_upload" in APP_SOURCE
    assert "download_runtime_restore_check" in APP_SOURCE
    assert "status_storage_paths" in APP_SOURCE
    assert "storage_readiness" in APP_SOURCE
    assert "external_storage_configured" in APP_SOURCE
    assert "status_api_access" in APP_SOURCE
    assert "api_access" in APP_SOURCE
    assert "status_admin_checks" in APP_SOURCE
    assert "admin_checks" in APP_SOURCE
    assert "rate_limited_recent" in APP_SOURCE
    assert "status_upload_scan" in APP_SOURCE
    assert "upload_scans" in APP_SOURCE
    assert '"upload_scan"' in APP_SOURCE
    assert "format_admin_metrics" in APP_SOURCE
    assert "download_admin_metrics" in APP_SOURCE
    assert "runtime_event_to_safe_dict(event, include_request_id=False)" in APP_SOURCE
    assert "event.__dict__" not in APP_SOURCE


def test_streamlit_product_registry_management_requires_explicit_flag():
    assert "STREAMLIT_PRODUCT_REGISTRY_MANAGEMENT_ENABLED" in APP_SOURCE
    assert '"management_enabled": STREAMLIT_PRODUCT_REGISTRY_MANAGEMENT_ENABLED' in APP_SOURCE
    assert "if not STREAMLIT_PRODUCT_REGISTRY_MANAGEMENT_ENABLED:" in APP_SOURCE
    assert "product_registry_management_disabled" in APP_SOURCE
    guard_index = APP_SOURCE.index("if not STREAMLIT_PRODUCT_REGISTRY_MANAGEMENT_ENABLED:")
    registry_index = APP_SOURCE.index("registry = LocalProductRegistry()")
    assert guard_index < registry_index


def test_streamlit_share_link_management_requires_explicit_flag():
    assert "STREAMLIT_SHARE_LINK_MANAGEMENT_ENABLED" in APP_SOURCE
    assert '"management_enabled": STREAMLIT_SHARE_LINK_MANAGEMENT_ENABLED' in APP_SOURCE
    assert "if not STREAMLIT_SHARE_LINK_MANAGEMENT_ENABLED:" in APP_SOURCE
    assert "share_link_management_disabled" in APP_SOURCE
    assert "type=\"password\"" in APP_SOURCE
    assert "safe_streamlit_error_message" in APP_SOURCE
    guard_index = APP_SOURCE.index("if not STREAMLIT_SHARE_LINK_MANAGEMENT_ENABLED:")
    registry_index = APP_SOURCE.index("registry = LocalShareLinkRegistry()")
    assert guard_index < registry_index


def test_streamlit_share_link_management_sanitizes_errors():
    share_link_block = APP_SOURCE.split("def render_share_link_registry_management()", 1)[
        1
    ].split("def render_admin_status()", 1)[0]

    assert "from scripts._safe_cli import format_os_error, sanitize_cli_error_message" in APP_SOURCE
    assert "def safe_streamlit_error_message" in APP_SOURCE
    assert "st.error(str(exc))" not in share_link_block
    assert share_link_block.count("st.error(safe_streamlit_error_message(exc))") == 5


def test_streamlit_runtime_event_filter_covers_guard_events():
    assert "RUNTIME_EVENT_KIND_FILTER_OPTIONS" in APP_SOURCE
    for event_kind in [
        "provider_failure",
        "provider_quota_guard",
        "product_quota",
        "product_rbac",
        "product_registry_admin",
        "share_link_admin",
        "query_usage",
        "retrieval_trace",
        "code_execution",
        "api_access",
        "admin_check",
        "upload_scan",
        "retention_delete",
    ]:
        assert f'"{event_kind}"' in APP_SOURCE
    assert "options=RUNTIME_EVENT_KIND_FILTER_OPTIONS" in APP_SOURCE


def test_streamlit_corpus_profile_report_download_is_installed():
    assert "collect_corpus_profile_status" in APP_SOURCE
    assert "format_corpus_profile_status_report" in APP_SOURCE
    assert "safe_corpus_profile_report_filename" in APP_SOURCE
    assert "download_profile_report" in APP_SOURCE
    assert "corpus_profile_report_download" in APP_SOURCE
    assert 'file_name=f"fluxmind-corpus-profile-{selected_profile}.md"' not in APP_SOURCE
