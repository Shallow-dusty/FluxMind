import json

from src.provider_readiness import (
    collect_provider_readiness,
    format_provider_readiness_markdown,
)


def test_provider_readiness_reports_local_foundation_without_activation():
    status = collect_provider_readiness(
        generated_at="2026-06-16T00:00:00+00:00",
        external_providers_enabled=False,
        llm_base_url_configured=True,
        llm_api_key_configured=True,
        image_provider_backend="local-mock",
        image_provider_api_configured=False,
        hosted_execution_backend="none",
        hosted_execution_configured=False,
        matlab_backend="none",
        matlab_license_configured=False,
        provider_quota_guard_enabled=False,
        code_execution_backend="local",
        docker_status={"configured": False, "available": False, "reason": "not_configured"},
        octave_available=False,
    )

    assert status["mode"] == "provider_readiness"
    assert status["local_foundation_ready"] is True
    assert status["activation_ready"] is False
    assert status["external_providers_enabled"] is False
    assert status["content_exported"] is False
    assert status["secrets_exported"] is False
    assert status["connectivity_checked"] is False
    assert status["blockers"]["local_foundation"] == []
    assert "external_providers_disabled" in status["blockers"]["activation"]
    assert "external_image_provider_not_configured" in status["blockers"]["activation"]
    assert "hosted_execution_provider_not_configured" in status["blockers"]["activation"]
    assert "matlab_backend_not_configured" in status["blockers"]["activation"]
    assert "provider_quota_guard_not_enabled" in status["blockers"]["activation"]
    assert "code_execution_backend_local" in status["advisories"]
    assert "octave_runtime_not_available" in status["advisories"]


def test_provider_readiness_can_report_activation_ready_without_secrets():
    status = collect_provider_readiness(
        generated_at="2026-06-16T00:00:00+00:00",
        external_providers_enabled=True,
        llm_base_url_configured=True,
        llm_api_key_configured=True,
        image_provider_backend="openai",
        image_provider_api_configured=True,
        hosted_execution_backend="cloudflare-sandbox",
        hosted_execution_configured=True,
        matlab_backend="matlab-engine",
        matlab_license_configured=True,
        provider_quota_guard_enabled=True,
        code_execution_backend="docker",
        docker_status={"configured": True, "available": True, "reason": "ok", "backend": "docker"},
        octave_available=True,
    )

    payload = json.dumps(status, ensure_ascii=False, sort_keys=True)
    assert status["local_foundation_ready"] is True
    assert status["activation_ready"] is True
    assert status["blockers"]["activation"] == []
    assert status["checks"]["external_image_provider"]["backend"] == "openai"
    assert status["checks"]["hosted_execution_provider"]["backend"] == "cloudflare-sandbox"
    assert status["checks"]["matlab_backend"]["backend"] == "matlab-engine"
    assert "hunter2" not in payload
    assert "sk-test" not in payload


def test_provider_readiness_sanitizes_secret_like_backend_values():
    status = collect_provider_readiness(
        generated_at="2026-06-16T00:00:00+00:00",
        external_providers_enabled=True,
        image_provider_backend="openai:hunter2@example.test",
        image_provider_api_configured=True,
        hosted_execution_backend="https://sandbox.example.test/hunter2",
        hosted_execution_configured=True,
        matlab_backend="matlab-engine:hunter2",
        matlab_license_configured=True,
        provider_quota_guard_enabled=True,
        docker_status={"configured": False, "available": False, "reason": "not_configured"},
        octave_available=False,
    )

    payload = json.dumps(status, ensure_ascii=False, sort_keys=True)
    assert "external_image_provider_unsupported" in status["blockers"]["activation"]
    assert "hosted_execution_provider_unsupported" in status["blockers"]["activation"]
    assert "matlab_backend_unsupported" in status["blockers"]["activation"]
    assert status["checks"]["external_image_provider"]["backend"] == "custom"
    assert "hunter2" not in payload
    assert "example.test" not in payload


def test_provider_readiness_blocks_invalid_local_foundation():
    status = collect_provider_readiness(
        generated_at="2026-06-16T00:00:00+00:00",
        code_execution_backend="docker",
        docker_status={"configured": True, "available": False, "reason": "docker_permission_denied"},
        artifact_registry_available=False,
        provider_failure_observability=False,
        octave_available=True,
    )

    assert status["local_foundation_ready"] is False
    assert "artifact_registry_unavailable" in status["blockers"]["local_foundation"]
    assert "provider_observability_unavailable" in status["blockers"]["local_foundation"]
    assert "docker_execution_unavailable" in status["blockers"]["local_foundation"]
    assert "docker_execution_docker_permission_denied" in status["advisories"]


def test_format_provider_readiness_markdown_is_no_secret():
    status = collect_provider_readiness(
        generated_at="2026-06-16T00:00:00+00:00",
        image_provider_backend="openai:hunter2@example.test",
        hosted_execution_backend="https://sandbox.example.test/hunter2",
        matlab_backend="matlab-engine:hunter2",
        docker_status={"configured": False, "available": False, "reason": "not_configured"},
        octave_available=False,
    )

    markdown = format_provider_readiness_markdown(status)

    assert "# FluxMind Provider Readiness" in markdown
    assert "Local foundation ready:" in markdown
    assert "Activation ready:" in markdown
    assert "hunter2" not in markdown
    assert "example.test" not in markdown
