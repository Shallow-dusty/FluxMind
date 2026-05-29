from pathlib import Path


APP_SOURCE = Path("app.py").read_text(encoding="utf-8")


def test_browser_translation_guard_is_installed():
    assert '<meta name="google" content="notranslate">' in APP_SOURCE
    assert 'node.setAttribute("translate", "no")' in APP_SOURCE
    assert 'node.classList.add("notranslate")' in APP_SOURCE
    assert "MutationObserver" in APP_SOURCE


def test_streamlit_write_stream_is_not_used_for_chat_streaming():
    assert "st.write_stream" not in APP_SOURCE
    assert "render_streaming_response(prompt)" in APP_SOURCE
