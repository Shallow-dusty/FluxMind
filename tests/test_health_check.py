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
