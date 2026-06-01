"""No-key provider implementations for future platform capabilities.

These providers are deliberately local and replaceable. They let the app
develop artifact and execution flows without consuming real image-provider keys,
hosted sandbox accounts, or MATLAB licenses.
"""

from __future__ import annotations

import html
import hashlib
import json
import mimetypes
import os
import shutil
import threading
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from src.capabilities import (
    CodeExecutionProvider,
    CodeExecutionRequest,
    CodeExecutionResult,
    GeneratedArtifact,
    ImageGenerationProvider,
    ImageGenerationRequest,
)
from src.config import ARTIFACTS_DIR

MAX_EXECUTION_FILES = 32
MAX_EXECUTION_FILE_BYTES = 256 * 1024
MAX_EXECUTION_TOTAL_BYTES = 1024 * 1024


def docker_execution_status(
    *,
    configured_backend: str,
    image: str,
    timeout_s: float = 3.0,
) -> dict[str, str | bool]:
    """Return no-secret readiness for the future Docker execution backend."""
    docker_path = shutil.which("docker")
    status: dict[str, str | bool] = {
        "backend": configured_backend or "local",
        "configured": configured_backend == "docker",
        "available": False,
        "docker_executable": Path(docker_path).name if docker_path else "",
        "image": image,
        "reason": "not_configured",
    }
    if configured_backend != "docker":
        return status
    if docker_path is None:
        status["reason"] = "docker_not_found"
        return status
    try:
        proc = subprocess.run(
            [docker_path, "version", "--format", "{{.Server.Version}}"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        status["reason"] = "docker_timeout"
        return status
    except OSError as exc:
        status["reason"] = exc.__class__.__name__
        return status
    if proc.returncode != 0:
        stderr = (proc.stderr or "").lower()
        if "permission denied" in stderr:
            status["reason"] = "docker_permission_denied"
        else:
            status["reason"] = "docker_unavailable"
        return status
    status["available"] = True
    status["reason"] = "ok"
    status["docker_server_version"] = (proc.stdout or "").strip()
    return status


def execution_runtime_metadata(
    request: CodeExecutionRequest,
    *,
    memory_limit_enforced: bool,
    cpu_limit_enforced: bool,
    provider_runtime: str,
    runtime_available: bool = True,
    runtime_details: dict[str, str] | None = None,
) -> dict[str, str]:
    """Return no-secret execution environment metadata for persisted results."""
    input_total_bytes = sum(len(content.encode("utf-8")) for content in request.files.values())
    metadata = {
        "language": request.language,
        "entrypoint": request.entrypoint,
        "input_file_count": str(len(request.files)),
        "input_total_bytes": str(input_total_bytes),
        "provider_runtime": provider_runtime,
        "runtime_available": "true" if runtime_available else "false",
        "filesystem_isolation": "temporary_workdir",
        "network_policy_enforced": "false",
        "timeout_s": str(request.timeout_s),
        "cpu_time_s": str(max(1, int(request.timeout_s))),
        "memory_mb": str(request.memory_mb),
        "memory_limit_enforced": "true" if memory_limit_enforced else "false",
        "cpu_limit_enforced": "true" if cpu_limit_enforced else "false",
        "max_files": str(MAX_EXECUTION_FILES),
        "max_file_bytes": str(MAX_EXECUTION_FILE_BYTES),
        "max_total_file_bytes": str(MAX_EXECUTION_TOTAL_BYTES),
    }
    if runtime_details:
        metadata.update(runtime_details)
    return metadata


def python_runtime_details() -> dict[str, str]:
    """Return stable no-secret details for the local Python runtime."""
    return {
        "python_executable": Path(sys.executable).name,
        "python_version": sys.version.split()[0],
        "python_implementation": sys.implementation.name,
    }


def octave_runtime_details(
    *,
    executable: str,
    resolved_executable: str | None,
) -> dict[str, str]:
    """Return no-secret details for the local Octave-compatible runtime."""
    details = {
        "octave_executable": Path(resolved_executable or executable).name,
        "octave_available": "true" if resolved_executable else "false",
    }
    if resolved_executable:
        details["octave_executable_resolved"] = Path(resolved_executable).name
    return details


def python_execution_metadata(
    request: CodeExecutionRequest,
    *,
    memory_limit_enforced: bool,
    cpu_limit_enforced: bool,
) -> dict[str, str]:
    return execution_runtime_metadata(
        request,
        memory_limit_enforced=memory_limit_enforced,
        cpu_limit_enforced=cpu_limit_enforced,
        provider_runtime="python-local",
        runtime_details=python_runtime_details(),
    )


def octave_execution_metadata(
    request: CodeExecutionRequest,
    *,
    memory_limit_enforced: bool,
    cpu_limit_enforced: bool,
    executable: str,
    resolved_executable: str | None,
) -> dict[str, str]:
    return execution_runtime_metadata(
        request,
        memory_limit_enforced=memory_limit_enforced,
        cpu_limit_enforced=cpu_limit_enforced,
        provider_runtime="gnu-octave-local",
        runtime_available=resolved_executable is not None,
        runtime_details=octave_runtime_details(
            executable=executable,
            resolved_executable=resolved_executable,
        ),
    )


def execution_limit_preexec(memory_mb: int, timeout_s: int):
    """Return a Unix preexec hook for child process resource limits."""
    if os.name != "posix":
        return None, False, False

    try:
        import resource
    except ImportError:
        return None, False, False

    memory_limit_enforced = memory_mb > 0 and hasattr(resource, "RLIMIT_AS")
    cpu_limit_s = max(1, int(timeout_s)) if timeout_s > 0 else 0
    cpu_limit_enforced = cpu_limit_s > 0 and hasattr(resource, "RLIMIT_CPU")
    if not memory_limit_enforced and not cpu_limit_enforced:
        return None, False, False

    def apply_execution_limits() -> None:
        if memory_limit_enforced:
            memory_limit = int(memory_mb) * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
        if cpu_limit_enforced:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit_s, cpu_limit_s + 1))

    return apply_execution_limits, memory_limit_enforced, cpu_limit_enforced


def _resolve_workdir_path(workdir: Path, name: str) -> Path:
    """Resolve a user-provided relative path without allowing workdir escape."""
    if not name or Path(name).is_absolute():
        raise ValueError(f"Invalid execution file path: {name}")
    target = (workdir / name).resolve()
    try:
        target.relative_to(workdir.resolve())
    except ValueError as exc:
        raise ValueError(f"Execution file path escapes workdir: {name}") from exc
    return target


def _is_collectable_output(path: Path, workdir: Path, input_files: set[Path]) -> bool:
    """Return whether a generated file can be safely copied as an artifact."""
    if path.is_symlink() or not path.is_file():
        return False
    resolved = path.resolve()
    if resolved in input_files:
        return False
    try:
        resolved.relative_to(workdir.resolve())
    except ValueError:
        return False
    return True


def _materialize_execution_files(
    workdir: Path,
    request: CodeExecutionRequest,
    *,
    runtime_metadata: dict[str, str],
) -> tuple[Path, set[Path] | None, CodeExecutionResult | None]:
    """Write request files into a temp workdir or return a structured failure."""
    input_files: set[Path] = set()
    try:
        if len(request.files) > MAX_EXECUTION_FILES:
            raise ValueError(f"Too many execution input files: {len(request.files)} > {MAX_EXECUTION_FILES}")
        total_bytes = 0
        for name, content in request.files.items():
            target = _resolve_workdir_path(workdir, name)
            content_bytes = content.encode("utf-8")
            byte_count = len(content_bytes)
            if byte_count > MAX_EXECUTION_FILE_BYTES:
                raise ValueError(
                    f"Execution input file is too large: {name} "
                    f"({byte_count} > {MAX_EXECUTION_FILE_BYTES} bytes)"
                )
            total_bytes += byte_count
            if total_bytes > MAX_EXECUTION_TOTAL_BYTES:
                raise ValueError(
                    f"Execution input files are too large in total: "
                    f"{total_bytes} > {MAX_EXECUTION_TOTAL_BYTES} bytes"
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content_bytes)
            input_files.add(target.resolve())
        entrypoint = _resolve_workdir_path(workdir, request.entrypoint)
    except ValueError as exc:
        return (
            workdir / request.entrypoint,
            None,
            CodeExecutionResult(
                exit_code=2,
                stdout="",
                stderr=str(exc),
                runtime_metadata=runtime_metadata,
            ),
        )
    return entrypoint, input_files, None


class LocalArtifactStore:
    """Persist generated artifacts under the project artifact directory."""

    def __init__(self, root: Path | None = None):
        self.root = root or ARTIFACTS_DIR

    def write_text(self, relative_path: str, content: str, mime_type: str) -> GeneratedArtifact:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return self._artifact_for(path, mime_type)

    def write_bytes(self, relative_path: str, content: bytes, mime_type: str) -> GeneratedArtifact:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return self._artifact_for(path, mime_type)

    def copy_file(self, relative_path: str, source: Path, mime_type: str) -> GeneratedArtifact:
        return self.write_bytes(relative_path, source.read_bytes(), mime_type)

    @staticmethod
    def _artifact_for(path: Path, mime_type: str) -> GeneratedArtifact:
        if mime_type.startswith("image/"):
            kind = "image"
        elif mime_type.startswith("text/"):
            kind = "text"
        else:
            kind = "file"
        content = path.read_bytes()
        return GeneratedArtifact(
            kind=kind,
            uri=path.resolve().as_uri(),
            mime_type=mime_type,
            title=path.name,
            metadata={
                "provider": "local",
                "checksum_sha256": hashlib.sha256(content).hexdigest(),
                "byte_count": str(len(content)),
            },
        )


class MockImageGenerationProvider(ImageGenerationProvider):
    """Create deterministic SVG placeholder diagrams without external keys."""

    def __init__(self, store: LocalArtifactStore | None = None):
        self.store = store or LocalArtifactStore()

    def generate(self, request: ImageGenerationRequest) -> GeneratedArtifact:
        safe_prompt = html.escape(request.prompt[:240])
        safe_style = html.escape(request.style)
        svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1024" height="1024" viewBox="0 0 1024 1024">
  <rect width="1024" height="1024" fill="#f8fafc"/>
  <rect x="96" y="152" width="832" height="720" rx="24" fill="#ffffff" stroke="#1f2937" stroke-width="4"/>
  <text x="512" y="244" text-anchor="middle" font-family="Arial, sans-serif" font-size="42" fill="#111827">FluxMind Diagram Stub</text>
  <text x="512" y="314" text-anchor="middle" font-family="Arial, sans-serif" font-size="24" fill="#4b5563">style: {safe_style}</text>
  <rect x="192" y="424" width="220" height="112" rx="12" fill="#dbeafe" stroke="#2563eb" stroke-width="3"/>
  <rect x="604" y="424" width="220" height="112" rx="12" fill="#dcfce7" stroke="#16a34a" stroke-width="3"/>
  <path d="M420 480 H596" stroke="#111827" stroke-width="6" marker-end="url(#arrow)"/>
  <text x="302" y="492" text-anchor="middle" font-family="Arial, sans-serif" font-size="24" fill="#1e3a8a">Input</text>
  <text x="714" y="492" text-anchor="middle" font-family="Arial, sans-serif" font-size="24" fill="#14532d">Observer</text>
  <text x="512" y="668" text-anchor="middle" font-family="Arial, sans-serif" font-size="22" fill="#111827">{safe_prompt}</text>
  <defs>
    <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
      <path d="M0,0 L0,6 L9,3 z" fill="#111827"/>
    </marker>
  </defs>
</svg>
"""
        digest = hashlib.sha256(f"{request.style}\n{request.prompt}".encode()).hexdigest()[:12]
        name = f"diagram-{digest}.svg"
        artifact = self.store.write_text(f"mock-images/{name}", svg, "image/svg+xml")
        return GeneratedArtifact(
            kind=artifact.kind,
            uri=artifact.uri,
            mime_type=artifact.mime_type,
            title=artifact.title,
            metadata={
                **artifact.metadata,
                "model": "local-mock-svg-v1",
                "prompt": request.prompt,
                "style": request.style,
                "size": request.size,
                "reference_uris": json.dumps(request.reference_uris, ensure_ascii=False),
                "cost_estimate_usd": "0",
            },
        )


class LocalPythonExecutionProvider(CodeExecutionProvider):
    """Run Python snippets in a temporary working directory.

    This is a development provider, not a production sandbox. Production use
    still requires a dedicated isolated service with explicit resource limits.
    """

    max_artifact_bytes = 2 * 1024 * 1024

    def __init__(self, store: LocalArtifactStore | None = None):
        self.store = store or LocalArtifactStore()

    def run(
        self,
        request: CodeExecutionRequest,
        *,
        cancel_event: threading.Event | None = None,
    ) -> CodeExecutionResult:
        if request.language != "python":
            return CodeExecutionResult(
                exit_code=2,
                stdout="",
                stderr=f"Unsupported local language: {request.language}",
                runtime_metadata=python_execution_metadata(
                    request,
                    memory_limit_enforced=False,
                    cpu_limit_enforced=False,
                ),
            )

        with tempfile.TemporaryDirectory(prefix="fluxmind-exec-") as tmp:
            workdir = Path(tmp)
            entrypoint, input_files, failure = _materialize_execution_files(
                workdir,
                request,
                runtime_metadata=python_execution_metadata(
                    request,
                    memory_limit_enforced=False,
                    cpu_limit_enforced=False,
                ),
            )
            if failure is not None:
                return failure
            if not entrypoint.exists():
                return CodeExecutionResult(
                    exit_code=2,
                    stdout="",
                    stderr=f"Entrypoint not found: {request.entrypoint}",
                    runtime_metadata=python_execution_metadata(
                        request,
                        memory_limit_enforced=False,
                        cpu_limit_enforced=False,
                    ),
                )

            preexec_fn, memory_limit_enforced, cpu_limit_enforced = execution_limit_preexec(
                request.memory_mb,
                request.timeout_s,
            )
            proc = subprocess.Popen(
                [sys.executable, str(entrypoint)],
                cwd=workdir,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=preexec_fn,
            )
            started = time.monotonic()
            while proc.poll() is None:
                if cancel_event and cancel_event.is_set():
                    proc.terminate()
                    try:
                        stdout, stderr = proc.communicate(timeout=2)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        stdout, stderr = proc.communicate()
                    return CodeExecutionResult(
                        exit_code=130,
                        stdout=stdout or "",
                        stderr=(stderr or "") + "Execution cancelled.",
                        runtime_metadata=python_execution_metadata(
                            request,
                            memory_limit_enforced=memory_limit_enforced,
                            cpu_limit_enforced=cpu_limit_enforced,
                        ),
                    )
                if time.monotonic() - started > request.timeout_s:
                    proc.terminate()
                    try:
                        stdout, stderr = proc.communicate(timeout=2)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        stdout, stderr = proc.communicate()
                    return CodeExecutionResult(
                        exit_code=124,
                        stdout=stdout or "",
                        stderr=(stderr or "") + f"Execution timed out after {request.timeout_s}s",
                        runtime_metadata=python_execution_metadata(
                            request,
                            memory_limit_enforced=memory_limit_enforced,
                            cpu_limit_enforced=cpu_limit_enforced,
                        ),
                    )
                time.sleep(0.05)

            stdout, stderr = proc.communicate()
            return CodeExecutionResult(
                exit_code=proc.returncode,
                stdout=stdout,
                stderr=stderr,
                artifacts=self._collect_artifacts(workdir, input_files, request),
                runtime_metadata=python_execution_metadata(
                    request,
                    memory_limit_enforced=memory_limit_enforced,
                    cpu_limit_enforced=cpu_limit_enforced,
                ),
            )

    def _collect_artifacts(
        self,
        workdir: Path,
        input_files: set[Path],
        request: CodeExecutionRequest,
    ) -> list[GeneratedArtifact]:
        artifacts: list[GeneratedArtifact] = []
        digest = hashlib.sha256(
            f"{request.language}\n{request.entrypoint}\n{sorted(request.files)}".encode()
        ).hexdigest()[:12]
        for path in sorted(item for item in workdir.rglob("*") if item.is_file()):
            if not _is_collectable_output(path, workdir, input_files):
                continue
            if path.stat().st_size > self.max_artifact_bytes:
                continue
            mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            relative = path.relative_to(workdir).as_posix()
            safe_relative = relative.replace("/", "-")
            artifact = self.store.copy_file(
                f"code-runs/{digest}/{safe_relative}",
                path,
                mime_type,
            )
            kind = "plot" if mime_type.startswith("image/") else artifact.kind
            artifacts.append(
                GeneratedArtifact(
                    kind=kind,
                    uri=artifact.uri,
                    mime_type=artifact.mime_type,
                    title=relative,
                    metadata={
                        **artifact.metadata,
                        "source": relative,
                        "language": request.language,
                        "entrypoint": request.entrypoint,
                        "cost_estimate_usd": "0",
                    },
                )
            )
        return artifacts


class LocalOctaveExecutionProvider(CodeExecutionProvider):
    """Run GNU Octave-compatible scripts when a local octave binary exists.

    This is still a development provider. It does not activate MATLAB or a
    hosted sandbox; it only provides the no-key execution contract and artifact
    capture surface.
    """

    max_artifact_bytes = 2 * 1024 * 1024

    def __init__(
        self,
        store: LocalArtifactStore | None = None,
        *,
        executable: str = "octave",
    ):
        self.store = store or LocalArtifactStore()
        self.executable = executable

    def run(
        self,
        request: CodeExecutionRequest,
        *,
        cancel_event: threading.Event | None = None,
    ) -> CodeExecutionResult:
        if request.language != "octave":
            return CodeExecutionResult(
                exit_code=2,
                stdout="",
                stderr=f"Unsupported local language: {request.language}",
                runtime_metadata=octave_execution_metadata(
                    request,
                    memory_limit_enforced=False,
                    cpu_limit_enforced=False,
                    executable=self.executable,
                    resolved_executable=None,
                ),
            )

        octave_bin = shutil.which(self.executable)
        if octave_bin is None:
            return CodeExecutionResult(
                exit_code=127,
                stdout="",
                stderr=(
                    "GNU Octave executable not found. Install octave or attach "
                    "a hosted execution provider."
                ),
                runtime_metadata=octave_execution_metadata(
                    request,
                    memory_limit_enforced=False,
                    cpu_limit_enforced=False,
                    executable=self.executable,
                    resolved_executable=None,
                ),
            )

        with tempfile.TemporaryDirectory(prefix="fluxmind-octave-") as tmp:
            workdir = Path(tmp)
            entrypoint, input_files, failure = _materialize_execution_files(
                workdir,
                request,
                runtime_metadata=octave_execution_metadata(
                    request,
                    memory_limit_enforced=False,
                    cpu_limit_enforced=False,
                    executable=self.executable,
                    resolved_executable=octave_bin,
                ),
            )
            if failure is not None:
                return failure
            if not entrypoint.exists():
                return CodeExecutionResult(
                    exit_code=2,
                    stdout="",
                    stderr=f"Entrypoint not found: {request.entrypoint}",
                    runtime_metadata=octave_execution_metadata(
                        request,
                        memory_limit_enforced=False,
                        cpu_limit_enforced=False,
                        executable=self.executable,
                        resolved_executable=octave_bin,
                    ),
                )

            preexec_fn, memory_limit_enforced, cpu_limit_enforced = execution_limit_preexec(
                request.memory_mb,
                request.timeout_s,
            )
            proc = subprocess.Popen(
                [octave_bin, "--quiet", "--no-gui", str(entrypoint)],
                cwd=workdir,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=preexec_fn,
            )
            started = time.monotonic()
            while proc.poll() is None:
                if cancel_event and cancel_event.is_set():
                    proc.terminate()
                    try:
                        stdout, stderr = proc.communicate(timeout=2)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        stdout, stderr = proc.communicate()
                    return CodeExecutionResult(
                        exit_code=130,
                        stdout=stdout or "",
                        stderr=(stderr or "") + "Execution cancelled.",
                        runtime_metadata=octave_execution_metadata(
                            request,
                            memory_limit_enforced=memory_limit_enforced,
                            cpu_limit_enforced=cpu_limit_enforced,
                            executable=self.executable,
                            resolved_executable=octave_bin,
                        ),
                    )
                if time.monotonic() - started > request.timeout_s:
                    proc.terminate()
                    try:
                        stdout, stderr = proc.communicate(timeout=2)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        stdout, stderr = proc.communicate()
                    return CodeExecutionResult(
                        exit_code=124,
                        stdout=stdout or "",
                        stderr=(stderr or "") + f"Execution timed out after {request.timeout_s}s",
                        runtime_metadata=octave_execution_metadata(
                            request,
                            memory_limit_enforced=memory_limit_enforced,
                            cpu_limit_enforced=cpu_limit_enforced,
                            executable=self.executable,
                            resolved_executable=octave_bin,
                        ),
                    )
                time.sleep(0.05)

            stdout, stderr = proc.communicate()
            return CodeExecutionResult(
                exit_code=proc.returncode,
                stdout=stdout,
                stderr=stderr,
                artifacts=self._collect_artifacts(workdir, input_files, request),
                runtime_metadata=octave_execution_metadata(
                    request,
                    memory_limit_enforced=memory_limit_enforced,
                    cpu_limit_enforced=cpu_limit_enforced,
                    executable=self.executable,
                    resolved_executable=octave_bin,
                ),
            )

    def _collect_artifacts(
        self,
        workdir: Path,
        input_files: set[Path],
        request: CodeExecutionRequest,
    ) -> list[GeneratedArtifact]:
        artifacts: list[GeneratedArtifact] = []
        digest = hashlib.sha256(
            f"{request.language}\n{request.entrypoint}\n{sorted(request.files)}".encode()
        ).hexdigest()[:12]
        for path in sorted(item for item in workdir.rglob("*") if item.is_file()):
            if not _is_collectable_output(path, workdir, input_files):
                continue
            if path.stat().st_size > self.max_artifact_bytes:
                continue
            mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            relative = path.relative_to(workdir).as_posix()
            safe_relative = relative.replace("/", "-")
            artifact = self.store.copy_file(
                f"octave-runs/{digest}/{safe_relative}",
                path,
                mime_type,
            )
            kind = "plot" if mime_type.startswith("image/") else artifact.kind
            artifacts.append(
                GeneratedArtifact(
                    kind=kind,
                    uri=artifact.uri,
                    mime_type=artifact.mime_type,
                    title=relative,
                    metadata={
                        **artifact.metadata,
                        "source": relative,
                        "language": request.language,
                        "entrypoint": request.entrypoint,
                        "runtime": "gnu-octave-local",
                        "cost_estimate_usd": "0",
                    },
                )
            )
        return artifacts
