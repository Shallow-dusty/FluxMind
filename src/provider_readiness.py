"""No-secret readiness for external provider activation."""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from typing import Any

from src import config
from src.providers import docker_execution_status, octave_runtime_details


PROVIDER_READINESS_SCHEMA_VERSION = 1
DISABLED_BACKENDS = {"", "none", "disabled", "local", "local-disabled", "mock", "local-mock"}
SAFE_BACKEND_CHARS = set("abcdefghijklmnopqrstuvwxyz0123456789_.-")
SUPPORTED_IMAGE_PROVIDERS = {
    "external",
    "openai",
    "replicate",
    "stability",
}
SUPPORTED_HOSTED_EXECUTION_PROVIDERS = {
    "cloudflare-sandbox",
    "e2b",
    "external",
    "modal",
}
SUPPORTED_MATLAB_BACKENDS = {
    "external",
    "matlab-batch",
    "matlab-engine",
    "matlab-online",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_backend_name(value: str | None) -> str:
    backend = (value or "none").strip().lower() or "none"
    if len(backend) > 64:
        return "custom"
    if any(char not in SAFE_BACKEND_CHARS for char in backend):
        return "custom"
    return backend


def _target_status(
    *,
    value: str | None,
    configured: bool,
    supported: set[str],
    missing_reason: str,
    unsupported_reason: str,
    incomplete_reason: str,
) -> dict[str, Any]:
    backend = _safe_backend_name(value)
    backend_configured = backend not in DISABLED_BACKENDS
    supported_backend = backend_configured and backend in supported
    available = supported_backend and configured
    if not backend_configured:
        reason = missing_reason
    elif not supported_backend:
        reason = unsupported_reason
    elif not configured:
        reason = incomplete_reason
    else:
        reason = "configured_not_connected"
    return {
        "backend": backend,
        "configured": backend_configured and configured,
        "backend_configured": backend_configured,
        "supported": supported_backend,
        "available": available,
        "reason": reason,
    }


def collect_provider_readiness(
    *,
    generated_at: str | None = None,
    external_providers_enabled: bool | None = None,
    llm_base_url_configured: bool | None = None,
    llm_api_key_configured: bool | None = None,
    image_provider_backend: str | None = None,
    image_provider_api_configured: bool | None = None,
    hosted_execution_backend: str | None = None,
    hosted_execution_configured: bool | None = None,
    matlab_backend: str | None = None,
    matlab_license_configured: bool | None = None,
    provider_quota_guard_enabled: bool | None = None,
    code_execution_backend: str | None = None,
    docker_status: dict[str, Any] | None = None,
    octave_available: bool | None = None,
    artifact_registry_available: bool = True,
    provider_failure_observability: bool = True,
) -> dict[str, Any]:
    """Collect no-secret readiness for activating real external providers."""
    providers_enabled = (
        config.EXTERNAL_PROVIDERS_ENABLED
        if external_providers_enabled is None
        else bool(external_providers_enabled)
    )
    base_url_configured = (
        bool(config.LLM_BASE_URL and "example.com" not in config.LLM_BASE_URL)
        if llm_base_url_configured is None
        else bool(llm_base_url_configured)
    )
    api_key_configured = (
        bool(config.LLM_API_KEY.strip())
        if llm_api_key_configured is None
        else bool(llm_api_key_configured)
    )
    image_api_configured = (
        config.IMAGE_PROVIDER_API_CONFIGURED
        if image_provider_api_configured is None
        else bool(image_provider_api_configured)
    )
    hosted_configured = (
        config.HOSTED_EXECUTION_CONFIGURED
        if hosted_execution_configured is None
        else bool(hosted_execution_configured)
    )
    matlab_license = (
        config.MATLAB_LICENSE_CONFIGURED
        if matlab_license_configured is None
        else bool(matlab_license_configured)
    )
    quota_guard = (
        config.PROVIDER_QUOTA_GUARD_ENABLED
        if provider_quota_guard_enabled is None
        else bool(provider_quota_guard_enabled)
    )
    execution_backend = (
        config.CODE_EXECUTION_BACKEND
        if code_execution_backend is None
        else (code_execution_backend or "local").strip().lower()
    )
    docker = docker_status
    if docker is None:
        docker = docker_execution_status(
            configured_backend=execution_backend,
            image=config.DOCKER_EXECUTION_IMAGE,
        )
    octave = (
        octave_available
        if octave_available is not None
        else octave_runtime_details(
            executable="octave",
            resolved_executable=shutil.which("octave"),
        )["octave_available"]
        == "true"
    )

    image_provider = _target_status(
        value=image_provider_backend if image_provider_backend is not None else config.IMAGE_PROVIDER_BACKEND,
        configured=image_api_configured,
        supported=SUPPORTED_IMAGE_PROVIDERS,
        missing_reason="external_image_provider_not_configured",
        unsupported_reason="unsupported_image_provider",
        incomplete_reason="image_provider_credentials_not_configured",
    )
    hosted_execution = _target_status(
        value=hosted_execution_backend if hosted_execution_backend is not None else config.HOSTED_EXECUTION_BACKEND,
        configured=hosted_configured,
        supported=SUPPORTED_HOSTED_EXECUTION_PROVIDERS,
        missing_reason="hosted_execution_provider_not_configured",
        unsupported_reason="unsupported_hosted_execution_provider",
        incomplete_reason="hosted_execution_credentials_not_configured",
    )
    matlab = _target_status(
        value=matlab_backend if matlab_backend is not None else config.MATLAB_BACKEND,
        configured=matlab_license,
        supported=SUPPORTED_MATLAB_BACKENDS,
        missing_reason="matlab_backend_not_configured",
        unsupported_reason="unsupported_matlab_backend",
        incomplete_reason="matlab_license_not_configured",
    )

    local_blockers: list[str] = []
    if not artifact_registry_available:
        local_blockers.append("artifact_registry_unavailable")
    if not provider_failure_observability:
        local_blockers.append("provider_observability_unavailable")
    if execution_backend not in {"local", "docker"}:
        local_blockers.append("unsupported_local_execution_backend")
    if execution_backend == "docker" and not docker.get("available", False):
        local_blockers.append("docker_execution_unavailable")

    activation_blockers: list[str] = []
    if not providers_enabled:
        activation_blockers.append("external_providers_disabled")
    for key, status in (
        ("external_image_provider", image_provider),
        ("hosted_execution_provider", hosted_execution),
        ("matlab_backend", matlab),
    ):
        if not status["backend_configured"]:
            activation_blockers.append(f"{key}_not_configured")
        elif not status["supported"]:
            activation_blockers.append(f"{key}_unsupported")
        elif not status["available"]:
            activation_blockers.append(status["reason"])
    if not quota_guard:
        activation_blockers.append("provider_quota_guard_not_enabled")

    advisories: list[str] = []
    if not base_url_configured:
        advisories.append("llm_base_url_not_configured")
    if not api_key_configured:
        advisories.append("llm_api_key_not_configured")
    if execution_backend == "local":
        advisories.append("code_execution_backend_local")
    if not octave:
        advisories.append("octave_runtime_not_available")
    if docker.get("configured") and not docker.get("available"):
        advisories.append(f"docker_execution_{docker.get('reason', 'unavailable')}")

    local_foundation_ready = not local_blockers
    activation_ready = local_foundation_ready and not activation_blockers

    return {
        "mode": "provider_readiness",
        "schema_version": PROVIDER_READINESS_SCHEMA_VERSION,
        "generated_at": generated_at or _utc_now(),
        "local_foundation_ready": local_foundation_ready,
        "activation_ready": activation_ready,
        "external_providers_enabled": providers_enabled,
        "content_exported": False,
        "secrets_exported": False,
        "connectivity_checked": False,
        "summary": {
            "llm_base_url_configured": base_url_configured,
            "llm_api_key_configured": api_key_configured,
            "local_image_mock_available": True,
            "local_python_execution_available": True,
            "local_octave_available": bool(octave),
            "docker_execution_available": bool(docker.get("available", False)),
            "artifact_registry_available": bool(artifact_registry_available),
            "provider_failure_observability": bool(provider_failure_observability),
            "external_image_provider_configured": bool(image_provider["configured"]),
            "hosted_execution_provider_configured": bool(hosted_execution["configured"]),
            "matlab_backend_configured": bool(matlab["configured"]),
            "provider_quota_guard_enabled": quota_guard,
        },
        "checks": {
            "text_llm": {
                "base_url_configured": base_url_configured,
                "api_key_configured": api_key_configured,
                "secret_exported": False,
            },
            "local_image_mock": {
                "available": True,
                "provider": "mock_svg_local",
            },
            "local_python_execution": {
                "available": True,
                "backend": execution_backend,
            },
            "docker_execution": {
                "configured": bool(docker.get("configured", False)),
                "available": bool(docker.get("available", False)),
                "reason": docker.get("reason", ""),
                "backend": docker.get("backend", execution_backend),
            },
            "local_octave": {
                "available": bool(octave),
                "provider": "gnu-octave-local",
            },
            "external_image_provider": image_provider,
            "hosted_execution_provider": hosted_execution,
            "matlab_backend": matlab,
            "provider_quota_guard": {
                "enabled": quota_guard,
                "reason": "enabled" if quota_guard else "provider_quota_guard_not_enabled",
            },
        },
        "blockers": {
            "local_foundation": local_blockers,
            "activation": activation_blockers,
        },
        "advisories": advisories,
        "notes": [
            "Local mock image, Python execution, Octave-compatible execution, and Docker readiness are foundations only.",
            "Activation still requires explicit external provider, hosted sandbox, MATLAB, and quota/cost guard configuration.",
        ],
    }


def _format_bool(value: Any) -> str:
    return "true" if bool(value) else "false"


def format_provider_readiness_markdown(status: dict[str, Any]) -> str:
    """Render provider readiness as no-secret Markdown."""
    summary = status.get("summary", {}) or {}
    checks = status.get("checks", {}) or {}
    blockers = status.get("blockers", {}) or {}
    lines = [
        "# FluxMind Provider Readiness",
        "",
        f"- Mode: {status.get('mode', '')}",
        f"- Schema version: {status.get('schema_version', '')}",
        f"- Generated at: {status.get('generated_at', '')}",
        f"- Local foundation ready: {_format_bool(status.get('local_foundation_ready', False))}",
        f"- Activation ready: {_format_bool(status.get('activation_ready', False))}",
        f"- External providers enabled: {_format_bool(status.get('external_providers_enabled', False))}",
        f"- Content exported: {_format_bool(status.get('content_exported', False))}",
        f"- Secrets exported: {_format_bool(status.get('secrets_exported', False))}",
        f"- Connectivity checked: {_format_bool(status.get('connectivity_checked', False))}",
        "",
        "## Local Foundation",
        "",
        f"- LLM base URL configured: {_format_bool(summary.get('llm_base_url_configured', False))}",
        f"- LLM API key configured: {_format_bool(summary.get('llm_api_key_configured', False))}",
        f"- Local image mock available: {_format_bool(summary.get('local_image_mock_available', False))}",
        f"- Local Python execution available: {_format_bool(summary.get('local_python_execution_available', False))}",
        f"- Local Octave available: {_format_bool(summary.get('local_octave_available', False))}",
        f"- Docker execution available: {_format_bool(summary.get('docker_execution_available', False))}",
        f"- Artifact registry available: {_format_bool(summary.get('artifact_registry_available', False))}",
        f"- Provider observability available: {_format_bool(summary.get('provider_failure_observability', False))}",
        "",
        "## Activation Targets",
        "",
        f"- External image provider: {checks.get('external_image_provider', {}).get('backend', '')} ({checks.get('external_image_provider', {}).get('reason', '')})",
        f"- Hosted execution provider: {checks.get('hosted_execution_provider', {}).get('backend', '')} ({checks.get('hosted_execution_provider', {}).get('reason', '')})",
        f"- MATLAB backend: {checks.get('matlab_backend', {}).get('backend', '')} ({checks.get('matlab_backend', {}).get('reason', '')})",
        f"- Provider quota guard: {_format_bool(checks.get('provider_quota_guard', {}).get('enabled', False))}",
        "",
        "## Blockers",
        "",
        f"- Local foundation: {', '.join(blockers.get('local_foundation', [])) or 'none'}",
        f"- Activation: {', '.join(blockers.get('activation', [])) or 'none'}",
        f"- Advisories: {', '.join(status.get('advisories', [])) or 'none'}",
    ]
    return "\n".join(lines)
