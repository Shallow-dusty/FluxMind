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
    assert "enqueue_local_python" in APP_SOURCE
    assert "cancel_job" in APP_SOURCE
    assert "retry_job" in APP_SOURCE
    assert "LocalJobRunner().retry" in APP_SOURCE


def test_streamlit_artifact_gallery_is_installed():
    assert "LocalArtifactRegistry" in APP_SOURCE
    assert "render_latest_artifacts()" in APP_SOURCE
    assert "st.download_button" in APP_SOURCE


def test_streamlit_admin_status_panel_is_installed():
    assert "collect_admin_status" in APP_SOURCE
    assert "render_admin_status()" in APP_SOURCE
    assert "admin_status" in APP_SOURCE
