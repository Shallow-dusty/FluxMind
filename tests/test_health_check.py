import scripts.health_check as health_check
import subprocess


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
    monkeypatch.setattr(health_check.time, "sleep", lambda _seconds: None)

    assert health_check.http_status("https://example.test", 3, 2) == 200
    assert calls["count"] == 2


def test_http_status_retries_after_warmup_gateway_error(monkeypatch):
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
            raise health_check.urllib.error.HTTPError(
                "https://example.test",
                502,
                "Bad Gateway",
                hdrs=None,
                fp=None,
            )
        assert timeout == 3
        return Response()

    monkeypatch.setattr(health_check.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(health_check.time, "sleep", lambda _seconds: None)

    assert health_check.http_status("https://example.test", 3, 2) == 200
    assert calls["count"] == 2


def test_directory_size_bytes_counts_nested_files(tmp_path):
    (tmp_path / "a").write_text("123", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "b").write_text("45", encoding="utf-8")

    assert health_check.directory_size_bytes(tmp_path) == 5


def test_run_ssh_reports_timeout_without_traceback(monkeypatch):
    def fake_run(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd=["ssh"], timeout=45, output="partial")

    monkeypatch.setattr(health_check.subprocess, "run", fake_run)

    code, output = health_check.run_ssh("root@example.test", "true", 10)

    assert code == 124
    assert "partial" in output
    assert "timed out after 45.0s" in output


def test_run_ssh_uses_minimum_command_timeout(monkeypatch):
    calls = {}

    class Result:
        returncode = 0
        stdout = "ok\n"

    def fake_run(*_args, **kwargs):
        calls["timeout"] = kwargs["timeout"]
        return Result()

    monkeypatch.setattr(health_check.subprocess, "run", fake_run)

    code, output = health_check.run_ssh("root@example.test", "true", 10)

    assert code == 0
    assert output == "ok\n"
    assert calls["timeout"] == 45.0
