import scripts.health_check as health_check


def test_http_status_retries_after_transient_error(monkeypatch):
    calls = {"count": 0}

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def fake_urlopen(_request, timeout):
        calls["count"] += 1
        if calls["count"] == 1:
            raise TimeoutError("transient")
        assert timeout == 3
        return Response()

    monkeypatch.setattr(health_check.urllib.request, "urlopen", fake_urlopen)

    assert health_check.http_status("https://example.test", 3, 2) == 200
    assert calls["count"] == 2


def test_directory_size_bytes_counts_nested_files(tmp_path):
    (tmp_path / "a").write_text("123", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "b").write_text("45", encoding="utf-8")

    assert health_check.directory_size_bytes(tmp_path) == 5
