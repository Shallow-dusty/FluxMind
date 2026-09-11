from pathlib import Path


APP_SOURCE = Path("app.py").read_text(encoding="utf-8")


def test_page_requests_browser_translation_opt_out():
    assert '<meta name="google" content="notranslate">' in APP_SOURCE


def test_chat_streaming_avoids_translation_sensitive_write_stream():
    assert "st.write_stream" not in APP_SOURCE
