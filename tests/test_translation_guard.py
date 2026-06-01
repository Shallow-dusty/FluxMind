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


def test_streamlit_artifact_gallery_is_installed():
    assert "LocalArtifactRegistry" in APP_SOURCE
    assert "render_latest_artifacts()" in APP_SOURCE
    assert "artifact_search" in APP_SOURCE
    assert "artifact_kind_filter" in APP_SOURCE
    assert "artifact_job_kind_filter" in APP_SOURCE
    assert "st.download_button" in APP_SOURCE


def test_streamlit_admin_status_panel_is_installed():
    assert "collect_admin_status" in APP_SOURCE
    assert "collect_retention_preview" in APP_SOURCE
    assert "list_runtime_events" in APP_SOURCE
    assert "render_admin_status()" in APP_SOURCE
    assert "render_retention_preview()" in APP_SOURCE
    assert "render_runtime_events()" in APP_SOURCE
    assert "admin_status" in APP_SOURCE
    assert "retention_preview" in APP_SOURCE
    assert "runtime_events" in APP_SOURCE
    assert "event_kind_filter" in APP_SOURCE
    assert "delete_enabled" in APP_SOURCE
    assert "worker_leases" in APP_SOURCE
    assert "status_cost_pricing" in APP_SOURCE
    assert "status_storage" in APP_SOURCE
    assert "status_storage_inventory" in APP_SOURCE
    assert "status_runtime_manifest" in APP_SOURCE
    assert "download_runtime_manifest" in APP_SOURCE
    assert "status_storage_paths" in APP_SOURCE
    assert "storage_readiness" in APP_SOURCE
    assert "external_storage_configured" in APP_SOURCE


def test_streamlit_corpus_profile_report_download_is_installed():
    assert "collect_corpus_profile_status" in APP_SOURCE
    assert "format_corpus_profile_status_report" in APP_SOURCE
    assert "download_profile_report" in APP_SOURCE
    assert "corpus_profile_report_download" in APP_SOURCE
