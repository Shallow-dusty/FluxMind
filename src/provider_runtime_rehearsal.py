"""No-secret local provider runtime rehearsal for FluxMind."""

from __future__ import annotations

import tempfile
import shutil
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import unquote, urlparse

from src.capabilities import CodeExecutionRequest, ImageGenerationRequest
from src.execution_policy import POLICY_VIOLATION_EXIT_CODE
from src.provider_guard import provider_quota_guard_decision
from src.provider_readiness import collect_provider_readiness
from src.providers import (
    LocalArtifactStore,
    LocalOctaveExecutionProvider,
    LocalPythonExecutionProvider,
    MockImageGenerationProvider,
    docker_execution_status,
)


PROVIDER_RUNTIME_REHEARSAL_SCHEMA_VERSION = 1
PROVIDER_RUNTIME_REHEARSAL_STATE_DIR = ".fluxmind-provider-runtime-rehearsal"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _format_bool(value: Any) -> str:
    return "true" if bool(value) else "false"


@contextmanager
def _rehearsal_root(root: Path | None) -> Iterator[Path]:
    if root is not None:
        root.mkdir(parents=True, exist_ok=True)
        state_root = root / PROVIDER_RUNTIME_REHEARSAL_STATE_DIR
        state_root.mkdir(parents=True, exist_ok=True)
        yield state_root
        return
    with tempfile.TemporaryDirectory(prefix="fluxmind-provider-rehearsal-") as temp_root:
        yield Path(temp_root)


def _file_uri_exists(uri: str) -> bool:
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        return False
    try:
        return Path(unquote(parsed.path)).is_file()
    except OSError:
        return False


def _positive_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _reset_rehearsal_artifacts(artifact_root: Path) -> None:
    if artifact_root.exists():
        shutil.rmtree(artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)


def _policy_violation_summary(
    result: Any,
    *,
    expected_codes: tuple[str, ...],
) -> dict[str, Any]:
    stderr = str(getattr(result, "stderr", ""))
    matched = [code for code in expected_codes if code in stderr]
    metadata = getattr(result, "runtime_metadata", {}) or {}
    ok = (
        not bool(getattr(result, "success", False))
        and int(getattr(result, "exit_code", 0) or 0) == POLICY_VIOLATION_EXIT_CODE
        and metadata.get("execution_policy_enforced") == "true"
        and metadata.get("policy_violation") == "true"
        and len(matched) == len(expected_codes)
    )
    return {
        "ok": ok,
        "exit_code": int(getattr(result, "exit_code", 0) or 0),
        "policy_enforced": metadata.get("execution_policy_enforced") == "true",
        "policy_violation": metadata.get("policy_violation") == "true",
        "violation_count": _positive_int(metadata.get("execution_policy_violations")),
        "expected_code_count": len(expected_codes),
        "matched_code_count": len(matched),
        "stdout_exported": False,
        "stderr_exported": False,
        "content_exported": False,
        "paths_exported": False,
        "artifact_count": len(getattr(result, "artifacts", []) or []),
    }


def collect_provider_runtime_rehearsal(
    *,
    root: Path | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Run a local no-key provider contract rehearsal.

    The rehearsal exercises the deterministic SVG image provider, local Python
    execution with artifact capture, and the Octave-compatible unavailable-or-
    executable branch. It also reports current provider-readiness local
    foundation status. It does not connect to external image, hosted execution,
    MATLAB, or LLM providers.
    """
    with _rehearsal_root(root) as rehearsal_root:
        artifact_root = rehearsal_root / "artifacts"
        _reset_rehearsal_artifacts(artifact_root)
        store = LocalArtifactStore(artifact_root)

        image_artifact = MockImageGenerationProvider(store).generate(
            ImageGenerationRequest(
                prompt="Provider rehearsal SMC observer diagram",
                diagram_template="sliding-mode-observer",
            )
        )
        image_ok = (
            image_artifact.kind == "image"
            and image_artifact.mime_type == "image/svg+xml"
            and _positive_int(image_artifact.metadata.get("byte_count")) > 0
            and bool(image_artifact.metadata.get("checksum_sha256"))
            and _file_uri_exists(image_artifact.uri)
        )

        python_result = LocalPythonExecutionProvider(store).run(
            CodeExecutionRequest(
                language="python",
                entrypoint="main.py",
                files={
                    "main.py": (
                        "from pathlib import Path\n"
                        "Path('summary.txt').write_text('provider-runtime-rehearsal', encoding='utf-8')\n"
                        "print('provider-runtime-rehearsal-ok')\n"
                    )
                },
                timeout_s=10,
                memory_mb=256,
            )
        )
        python_ok = (
            python_result.success
            and "provider-runtime-rehearsal-ok" in python_result.stdout
            and len(python_result.artifacts) >= 1
            and python_result.runtime_metadata.get("runtime_available") == "true"
            and python_result.runtime_metadata.get("execution_policy_enforced") == "true"
            and python_result.runtime_metadata.get("policy_violation") == "false"
        )
        python_abuse_result = LocalPythonExecutionProvider(store).run(
            CodeExecutionRequest(
                language="python",
                entrypoint="main.py",
                files={"main.py": "import subprocess\nsubprocess.run(['echo', 'blocked'])\n"},
                timeout_s=10,
                memory_mb=256,
            )
        )

        octave_result = LocalOctaveExecutionProvider(store).run(
            CodeExecutionRequest(
                language="octave",
                entrypoint="main.m",
                files={
                    "main.m": (
                        "disp('provider-runtime-rehearsal-ok');\n"
                        "fid = fopen('summary.txt', 'w');\n"
                        "fprintf(fid, 'provider-runtime-rehearsal');\n"
                        "fclose(fid);\n"
                    )
                },
                timeout_s=10,
                memory_mb=256,
            )
        )
        octave_available = octave_result.runtime_metadata.get("runtime_available") == "true"
        octave_ok = (
            (
                octave_available
                and octave_result.success
                and "provider-runtime-rehearsal-ok" in octave_result.stdout
            )
            or (
                not octave_available
                and octave_result.exit_code == 127
                and "GNU Octave executable not found" in octave_result.stderr
            )
        )
        octave_abuse_result = LocalOctaveExecutionProvider(store).run(
            CodeExecutionRequest(
                language="octave",
                entrypoint="main.m",
                files={"main.m": "system('echo blocked');\n"},
                timeout_s=10,
                memory_mb=256,
            )
        )
        python_abuse = _policy_violation_summary(
            python_abuse_result,
            expected_codes=("python_import_not_allowed", "python_call_not_allowed"),
        )
        octave_abuse = _policy_violation_summary(
            octave_abuse_result,
            expected_codes=("octave_shell_call",),
        )
        abuse_policy_ok = bool(python_abuse["ok"] and octave_abuse["ok"])

        docker_status = docker_execution_status(
            configured_backend="docker",
            image="python:3.12-slim",
        )
        quota_allow = provider_quota_guard_decision(
            operation="provider_rehearsal_allowed",
            provider="local-mock-svg-v1",
            estimated_prompt_tokens=100,
            requested_completion_tokens=50,
            provider_quota_guard_enabled=True,
            max_prompt_tokens=1000,
            max_completion_tokens=500,
            max_cost_usd="0",
        )
        quota_block = provider_quota_guard_decision(
            operation="provider_rehearsal_blocked",
            provider="local-mock-svg-v1",
            estimated_prompt_tokens=1001,
            requested_completion_tokens=50,
            provider_quota_guard_enabled=True,
            max_prompt_tokens=1000,
            max_completion_tokens=500,
            max_cost_usd="0",
        )
        quota_guard_ok = (
            quota_allow.get("allowed") is True
            and quota_block.get("allowed") is False
            and quota_block.get("reason") == "provider_prompt_token_limit_exceeded"
            and quota_allow.get("content_exported") is False
            and quota_block.get("secrets_exported") is False
        )
        readiness = collect_provider_readiness(
            generated_at=generated_at,
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
            docker_status=docker_status,
            octave_available=octave_available,
            artifact_registry_available=True,
            provider_failure_observability=True,
        )

    readiness_ok = bool(readiness.get("local_foundation_ready"))
    ok = image_ok and python_ok and octave_ok and abuse_policy_ok and quota_guard_ok and readiness_ok
    return {
        "mode": "provider_runtime_rehearsal",
        "schema_version": PROVIDER_RUNTIME_REHEARSAL_SCHEMA_VERSION,
        "generated_at": generated_at or _utc_now(),
        "ok": ok,
        "local_only": True,
        "external_activation_ready": False,
        "content_exported": False,
        "secrets_exported": False,
        "paths_exported": False,
        "connectivity_checked": False,
        "image_provider": {
            "ok": image_ok,
            "provider": "local-mock-svg-v1",
            "artifact_kind": image_artifact.kind,
            "mime_type": image_artifact.mime_type,
            "artifact_byte_count": _positive_int(image_artifact.metadata.get("byte_count")),
            "checksum_present": bool(image_artifact.metadata.get("checksum_sha256")),
        },
        "python_execution": {
            "ok": python_ok,
            "success": python_result.success,
            "exit_code": python_result.exit_code,
            "artifact_count": len(python_result.artifacts),
            "runtime_available": python_result.runtime_metadata.get("runtime_available") == "true",
            "policy_enforced": python_result.runtime_metadata.get("execution_policy_enforced") == "true",
            "policy_violation": python_result.runtime_metadata.get("policy_violation") == "true",
        },
        "octave_execution": {
            "ok": octave_ok,
            "success": octave_result.success,
            "exit_code": octave_result.exit_code,
            "artifact_count": len(octave_result.artifacts),
            "runtime_available": octave_available,
            "reason": "available" if octave_available else "runtime_unavailable",
        },
        "execution_abuse_policy": {
            "ok": abuse_policy_ok,
            "python": python_abuse,
            "octave": octave_abuse,
            "content_exported": False,
            "secrets_exported": False,
            "paths_exported": False,
        },
        "docker_execution": {
            "configured": bool(docker_status.get("configured", False)),
            "available": bool(docker_status.get("available", False)),
            "reason": str(docker_status.get("reason", "")),
        },
        "provider_quota_guard": {
            "ok": quota_guard_ok,
            "enabled": True,
            "allowed_reason": quota_allow.get("reason", ""),
            "blocked_reason": quota_block.get("reason", ""),
            "allowed": quota_allow.get("allowed", False),
            "blocked": not bool(quota_block.get("allowed", True)),
            "max_prompt_tokens_per_request": quota_allow.get("max_prompt_tokens_per_request", 0),
            "max_completion_tokens_per_request": quota_allow.get(
                "max_completion_tokens_per_request",
                0,
            ),
            "content_exported": False,
            "secrets_exported": False,
        },
        "readiness": {
            "ok": readiness_ok,
            "local_foundation_ready": readiness.get("local_foundation_ready", False),
            "activation_ready": readiness.get("activation_ready", False),
            "activation_blockers": readiness.get("blockers", {}).get("activation", []),
            "local_foundation_blockers": readiness.get("blockers", {}).get("local_foundation", []),
        },
        "notes": [
            "Rehearsal uses local mock image generation and local execution providers only.",
            "External provider activation remains controlled by provider_readiness.",
        ],
    }


def format_provider_runtime_rehearsal_markdown(status: dict[str, Any]) -> str:
    """Render provider runtime rehearsal as no-secret Markdown."""
    image = status.get("image_provider", {}) or {}
    python = status.get("python_execution", {}) or {}
    octave = status.get("octave_execution", {}) or {}
    abuse = status.get("execution_abuse_policy", {}) or {}
    abuse_python = abuse.get("python", {}) or {}
    abuse_octave = abuse.get("octave", {}) or {}
    docker = status.get("docker_execution", {}) or {}
    quota = status.get("provider_quota_guard", {}) or {}
    readiness = status.get("readiness", {}) or {}
    lines = [
        "# FluxMind Provider Runtime Rehearsal",
        "",
        f"- Mode: {status.get('mode', '')}",
        f"- Schema version: {status.get('schema_version', '')}",
        f"- Generated at: {status.get('generated_at', '')}",
        f"- OK: {_format_bool(status.get('ok', False))}",
        f"- Local only: {_format_bool(status.get('local_only', False))}",
        f"- External activation ready: {_format_bool(status.get('external_activation_ready', False))}",
        f"- Content exported: {_format_bool(status.get('content_exported', False))}",
        f"- Secrets exported: {_format_bool(status.get('secrets_exported', False))}",
        f"- Paths exported: {_format_bool(status.get('paths_exported', False))}",
        f"- Connectivity checked: {_format_bool(status.get('connectivity_checked', False))}",
        "",
        "## Image Provider",
        "",
        f"- OK: {_format_bool(image.get('ok', False))}",
        f"- Provider: {image.get('provider', '')}",
        f"- Artifact kind: {image.get('artifact_kind', '')}",
        f"- MIME type: {image.get('mime_type', '')}",
        f"- Artifact byte count: {image.get('artifact_byte_count', 0)}",
        f"- Checksum present: {_format_bool(image.get('checksum_present', False))}",
        "",
        "## Python Execution",
        "",
        f"- OK: {_format_bool(python.get('ok', False))}",
        f"- Success: {_format_bool(python.get('success', False))}",
        f"- Exit code: {python.get('exit_code', 0)}",
        f"- Artifact count: {python.get('artifact_count', 0)}",
        f"- Runtime available: {_format_bool(python.get('runtime_available', False))}",
        f"- Policy enforced: {_format_bool(python.get('policy_enforced', False))}",
        f"- Policy violation: {_format_bool(python.get('policy_violation', False))}",
        "",
        "## Octave Execution",
        "",
        f"- OK: {_format_bool(octave.get('ok', False))}",
        f"- Success: {_format_bool(octave.get('success', False))}",
        f"- Exit code: {octave.get('exit_code', 0)}",
        f"- Artifact count: {octave.get('artifact_count', 0)}",
        f"- Runtime available: {_format_bool(octave.get('runtime_available', False))}",
        f"- Reason: {octave.get('reason', '')}",
        "",
        "## Execution Abuse Policy",
        "",
        f"- OK: {_format_bool(abuse.get('ok', False))}",
        f"- Python policy violation: {_format_bool(abuse_python.get('policy_violation', False))}",
        f"- Python matched code count: {abuse_python.get('matched_code_count', 0)}",
        f"- Octave policy violation: {_format_bool(abuse_octave.get('policy_violation', False))}",
        f"- Octave matched code count: {abuse_octave.get('matched_code_count', 0)}",
        f"- Content exported: {_format_bool(abuse.get('content_exported', False))}",
        f"- Secrets exported: {_format_bool(abuse.get('secrets_exported', False))}",
        f"- Paths exported: {_format_bool(abuse.get('paths_exported', False))}",
        "",
        "## Docker Execution",
        "",
        f"- Configured: {_format_bool(docker.get('configured', False))}",
        f"- Available: {_format_bool(docker.get('available', False))}",
        f"- Reason: {docker.get('reason', '')}",
        "",
        "## Provider Quota Guard",
        "",
        f"- OK: {_format_bool(quota.get('ok', False))}",
        f"- Enabled: {_format_bool(quota.get('enabled', False))}",
        f"- Allowed decision: {_format_bool(quota.get('allowed', False))}",
        f"- Allowed reason: {quota.get('allowed_reason', '')}",
        f"- Blocked decision: {_format_bool(quota.get('blocked', False))}",
        f"- Blocked reason: {quota.get('blocked_reason', '')}",
        f"- Max prompt tokens/request: {quota.get('max_prompt_tokens_per_request', 0)}",
        f"- Max completion tokens/request: {quota.get('max_completion_tokens_per_request', 0)}",
        f"- Content exported: {_format_bool(quota.get('content_exported', False))}",
        f"- Secrets exported: {_format_bool(quota.get('secrets_exported', False))}",
        "",
        "## Readiness",
        "",
        f"- OK: {_format_bool(readiness.get('ok', False))}",
        f"- Local foundation ready: {_format_bool(readiness.get('local_foundation_ready', False))}",
        f"- Activation ready: {_format_bool(readiness.get('activation_ready', False))}",
        f"- Activation blockers: {', '.join(readiness.get('activation_blockers', [])) or 'none'}",
        f"- Local foundation blockers: {', '.join(readiness.get('local_foundation_blockers', [])) or 'none'}",
    ]
    return "\n".join(lines)
