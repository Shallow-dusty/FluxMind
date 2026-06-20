import scripts.health_check as health_check
import subprocess


def test_remote_ssh_checks_include_recent_safety_anchors():
    source = (health_check.PROJECT_ROOT / "scripts" / "health_check.py").read_text(
        encoding="utf-8"
    )

    assert (
        "grep -q 'STREAMLIT_PRODUCT_REGISTRY_MANAGEMENT_ENABLED' "
        "/opt/fluxmind/app.py"
    ) in source
    assert (
        "grep -q 'product_registry_management_disabled' /opt/fluxmind/app.py"
    ) in source
    assert (
        "grep -q 'STREAMLIT_SHARE_LINK_MANAGEMENT_ENABLED' "
        "/opt/fluxmind/app.py"
    ) in source
    assert (
        "grep -q 'share_link_management_disabled' /opt/fluxmind/app.py"
    ) in source
    assert (
        "grep -q 'enforce_product_registry_admin_read' /opt/fluxmind/api.py"
    ) in source
    assert (
        "grep -q 'endpoint=\\\"/admin/product-registry/workspaces\\\"' "
        "/opt/fluxmind/api.py"
    ) in source
    assert (
        "grep -q 'endpoint=\\\"/admin/product-registry/permissions/check\\\"' "
        "/opt/fluxmind/api.py"
    ) in source
    assert (
        "grep -q 'requires --format json' /opt/fluxmind/scripts/api_key_registry.py"
    ) in source
    assert (
        "grep -q 'share_link_registry_backend_status' "
        "/opt/fluxmind/src/share_links.py"
    ) in source
    assert (
        "grep -q '/admin/share-links/status' /opt/fluxmind/api.py"
    ) in source
    assert (
        "grep -q 'share_link_registry_sqlite' /opt/fluxmind/src/storage_schema.py"
    ) in source
    assert (
        "grep -q 'provider_quota_guard_invalid_limit' "
        "/opt/fluxmind/src/provider_readiness.py"
    ) in source


def test_main_local_health_check_passes(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["health_check.py"])

    assert health_check.main() == 0
    output = capsys.readouterr().out
    assert "ok   required file: app.py" in output
    assert "ok   no-secret readiness CLI OS errors omit raw paths" in output
    assert "ok   API startup warmup readiness route installed" in output
    assert "skip local FAISS index is absent" in output or "ok   local FAISS index is non-empty" in output


def test_main_reports_url_failures(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["health_check.py", "--url", "https://example.test"])
    monkeypatch.setattr(health_check, "http_status", lambda url, timeout, retries: 503)

    assert health_check.main() == 1
    output = capsys.readouterr().out
    assert "fail https://example.test returns 200 (got 503)" in output
    assert "Failed checks:" in output


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
    assert "timed out after 180.0s" in output


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
    assert calls["timeout"] == 180.0
