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
from dataclasses import dataclass
from pathlib import Path

from src.capabilities import (
    CodeExecutionProvider,
    CodeExecutionRequest,
    CodeExecutionResult,
    GeneratedArtifact,
    ImageGenerationProvider,
    ImageGenerationRequest,
)
from src.config import (
    ARTIFACTS_DIR,
    CODE_EXECUTION_ALLOWED_IMPORTS,
    CODE_EXECUTION_MAX_ARTIFACT_BYTES,
    CODE_EXECUTION_MAX_ARTIFACT_CANDIDATES,
    CODE_EXECUTION_MAX_ARTIFACT_TOTAL_BYTES,
    CODE_EXECUTION_MAX_ARTIFACTS,
    CODE_EXECUTION_MAX_STDERR_BYTES,
    CODE_EXECUTION_MAX_STDOUT_BYTES,
    CODE_EXECUTION_POLICY,
    DOCKER_EXECUTION_IMAGE,
)
from src.execution_policy import (
    POLICY_VIOLATION_EXIT_CODE,
    ExecutionPolicyResult,
    evaluate_execution_policy,
)

MAX_EXECUTION_FILES = 32
MAX_EXECUTION_FILE_BYTES = 256 * 1024
MAX_EXECUTION_TOTAL_BYTES = 1024 * 1024


@dataclass(frozen=True)
class CapturedProcessOutput:
    stdout: str
    stderr: str
    returncode: int
    stdout_bytes: int
    stderr_bytes: int
    stdout_truncated: bool
    stderr_truncated: bool
    timed_out: bool = False
    cancelled: bool = False


@dataclass(frozen=True)
class ExecutionArtifactCollection:
    artifacts: list[GeneratedArtifact]
    metadata: dict[str, str]


@dataclass
class WorkdirFileScanState:
    scanned_entries: int = 0
    scanned_files: int = 0
    unreadable_dirs: int = 0
    truncated: bool = False


class BoundedStreamReader:
    """Read a subprocess stream to completion while keeping only bounded bytes."""

    def __init__(self, stream, *, limit_bytes: int):
        self.stream = stream
        self.limit_bytes = max(limit_bytes, 0)
        self.total_bytes = 0
        self._chunks: list[bytes] = []
        self._thread = threading.Thread(target=self._read, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def join(self) -> None:
        self._thread.join(timeout=2)

    @property
    def truncated(self) -> bool:
        return self.total_bytes > self.limit_bytes

    def text(self) -> str:
        return b"".join(self._chunks).decode("utf-8", errors="replace")

    def _read(self) -> None:
        while True:
            chunk = self.stream.read(4096)
            if not chunk:
                return
            if isinstance(chunk, str):
                chunk = chunk.encode("utf-8", errors="replace")
            self.total_bytes += len(chunk)
            kept_bytes = sum(len(item) for item in self._chunks)
            remaining = self.limit_bytes - kept_bytes
            if remaining > 0:
                self._chunks.append(chunk[:remaining])


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
    filesystem_isolation: str = "temporary_workdir",
    network_policy_enforced: bool = False,
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
        "filesystem_isolation": filesystem_isolation,
        "network_policy_enforced": "true" if network_policy_enforced else "false",
        "timeout_s": str(request.timeout_s),
        "cpu_time_s": str(max(1, int(request.timeout_s))),
        "memory_mb": str(request.memory_mb),
        "memory_limit_enforced": "true" if memory_limit_enforced else "false",
        "cpu_limit_enforced": "true" if cpu_limit_enforced else "false",
        "max_files": str(MAX_EXECUTION_FILES),
        "max_file_bytes": str(MAX_EXECUTION_FILE_BYTES),
        "max_total_file_bytes": str(MAX_EXECUTION_TOTAL_BYTES),
        "max_stdout_bytes": str(CODE_EXECUTION_MAX_STDOUT_BYTES),
        "max_stderr_bytes": str(CODE_EXECUTION_MAX_STDERR_BYTES),
        "max_artifacts": str(max(CODE_EXECUTION_MAX_ARTIFACTS, 0)),
        "max_artifact_bytes": str(max(CODE_EXECUTION_MAX_ARTIFACT_BYTES, 0)),
        "max_artifact_total_bytes": str(max(CODE_EXECUTION_MAX_ARTIFACT_TOTAL_BYTES, 0)),
        "max_artifact_candidates": str(max(CODE_EXECUTION_MAX_ARTIFACT_CANDIDATES, 0)),
    }
    if runtime_details:
        metadata.update(runtime_details)
    return metadata


def apply_output_capture_metadata(
    metadata: dict[str, str],
    output: CapturedProcessOutput,
) -> dict[str, str]:
    return {
        **metadata,
        "stdout_bytes": str(output.stdout_bytes),
        "stderr_bytes": str(output.stderr_bytes),
        "stdout_truncated": "true" if output.stdout_truncated else "false",
        "stderr_truncated": "true" if output.stderr_truncated else "false",
        "output_truncated": "true" if output.stdout_truncated or output.stderr_truncated else "false",
    }


def apply_policy_metadata(
    metadata: dict[str, str],
    policy: ExecutionPolicyResult,
) -> dict[str, str]:
    return {**metadata, **policy.metadata()}


def evaluate_request_policy(request: CodeExecutionRequest) -> ExecutionPolicyResult:
    return evaluate_execution_policy(
        request,
        profile=CODE_EXECUTION_POLICY,
        allowed_python_imports=CODE_EXECUTION_ALLOWED_IMPORTS,
    )


def execution_policy_failure_result(
    request: CodeExecutionRequest,
    *,
    runtime_metadata: dict[str, str],
    policy: ExecutionPolicyResult,
) -> CodeExecutionResult | None:
    if policy.allowed:
        return None
    return CodeExecutionResult(
        exit_code=POLICY_VIOLATION_EXIT_CODE,
        stdout="",
        stderr=f"Execution policy violation: {policy.message()}",
        runtime_metadata=runtime_metadata,
    )


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


def docker_execution_metadata(
    request: CodeExecutionRequest,
    *,
    image: str,
    docker_path: str | None,
    runtime_available: bool,
    container_name: str,
    container_command: list[str],
    container_user: str,
    docker_returncode: int | None = None,
) -> dict[str, str]:
    details = {
        "docker_image": image,
        "docker_executable": Path(docker_path).name if docker_path else "docker",
        "docker_network": "none",
        "docker_read_only_rootfs": "true",
        "docker_pids_limit": "64",
        "docker_cpus": "1",
        "docker_container_name": container_name,
        "container_workdir": "/work",
        "container_command": " ".join(container_command),
    }
    if container_user:
        details["container_user"] = container_user
    if docker_returncode is not None:
        details["docker_returncode"] = str(docker_returncode)
    return execution_runtime_metadata(
        request,
        memory_limit_enforced=True,
        cpu_limit_enforced=True,
        provider_runtime=f"docker-{request.language}",
        runtime_available=runtime_available,
        runtime_details=details,
        filesystem_isolation="docker_container_bind_mount",
        network_policy_enforced=True,
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


def terminate_process(proc: subprocess.Popen) -> None:
    try:
        proc.terminate()
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def truncate_output_text(value: str, limit_bytes: int) -> tuple[str, int, bool]:
    raw = value.encode("utf-8", errors="replace")
    if len(raw) <= max(limit_bytes, 0):
        return value, len(raw), False
    return raw[: max(limit_bytes, 0)].decode("utf-8", errors="replace"), len(raw), True


def capture_process_output(
    proc: subprocess.Popen,
    *,
    timeout_s: int,
    cancel_event: threading.Event | None = None,
    on_cancel=None,
    on_timeout=None,
) -> CapturedProcessOutput:
    """Capture subprocess output with bounded stdout/stderr memory."""
    stdout_stream = getattr(proc, "stdout", None)
    stderr_stream = getattr(proc, "stderr", None)
    if stdout_stream is None or stderr_stream is None:
        return capture_process_output_fallback(
            proc,
            timeout_s=timeout_s,
            cancel_event=cancel_event,
            on_cancel=on_cancel,
            on_timeout=on_timeout,
        )

    stdout_reader = BoundedStreamReader(
        stdout_stream,
        limit_bytes=CODE_EXECUTION_MAX_STDOUT_BYTES,
    )
    stderr_reader = BoundedStreamReader(
        stderr_stream,
        limit_bytes=CODE_EXECUTION_MAX_STDERR_BYTES,
    )
    stdout_reader.start()
    stderr_reader.start()

    started = time.monotonic()
    timed_out = False
    cancelled = False
    while proc.poll() is None:
        if cancel_event and cancel_event.is_set():
            cancelled = True
            if on_cancel:
                on_cancel()
            terminate_process(proc)
            break
        if time.monotonic() - started > timeout_s:
            timed_out = True
            if on_timeout:
                on_timeout()
            terminate_process(proc)
            break
        time.sleep(0.05)

    stdout_reader.join()
    stderr_reader.join()
    return CapturedProcessOutput(
        stdout=stdout_reader.text(),
        stderr=stderr_reader.text(),
        returncode=proc.returncode if proc.returncode is not None else -1,
        stdout_bytes=stdout_reader.total_bytes,
        stderr_bytes=stderr_reader.total_bytes,
        stdout_truncated=stdout_reader.truncated,
        stderr_truncated=stderr_reader.truncated,
        timed_out=timed_out,
        cancelled=cancelled,
    )


def capture_process_output_fallback(
    proc,
    *,
    timeout_s: int,
    cancel_event: threading.Event | None = None,
    on_cancel=None,
    on_timeout=None,
) -> CapturedProcessOutput:
    """Fallback for tests/fakes that do not expose stdout/stderr streams."""
    started = time.monotonic()
    timed_out = False
    cancelled = False
    while proc.poll() is None:
        if cancel_event and cancel_event.is_set():
            cancelled = True
            if on_cancel:
                on_cancel()
            terminate_process(proc)
            break
        if time.monotonic() - started > timeout_s:
            timed_out = True
            if on_timeout:
                on_timeout()
            terminate_process(proc)
            break
        time.sleep(0.05)
    stdout, stderr = proc.communicate()
    stdout_text = stdout.decode("utf-8", errors="replace") if isinstance(stdout, bytes) else (stdout or "")
    stderr_text = stderr.decode("utf-8", errors="replace") if isinstance(stderr, bytes) else (stderr or "")
    stdout_text, stdout_bytes, stdout_truncated = truncate_output_text(
        stdout_text,
        CODE_EXECUTION_MAX_STDOUT_BYTES,
    )
    stderr_text, stderr_bytes, stderr_truncated = truncate_output_text(
        stderr_text,
        CODE_EXECUTION_MAX_STDERR_BYTES,
    )
    return CapturedProcessOutput(
        stdout=stdout_text,
        stderr=stderr_text,
        returncode=proc.returncode if proc.returncode is not None else -1,
        stdout_bytes=stdout_bytes,
        stderr_bytes=stderr_bytes,
        stdout_truncated=stdout_truncated,
        stderr_truncated=stderr_truncated,
        timed_out=timed_out,
        cancelled=cancelled,
    )


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


def _execution_artifact_digest(request: CodeExecutionRequest) -> str:
    return hashlib.sha256(
        f"{request.language}\n{request.entrypoint}\n{sorted(request.files)}".encode()
    ).hexdigest()[:12]


def _iter_workdir_files(
    workdir: Path,
    *,
    max_entries: int,
    scan_state: WorkdirFileScanState,
):
    """Yield workdir file paths while bounding traversal of generated output."""
    pending = [workdir]
    while pending:
        directory = pending.pop()
        entries: list[os.DirEntry] = []
        try:
            with os.scandir(directory) as iterator:
                for entry in iterator:
                    if scan_state.scanned_entries >= max_entries:
                        scan_state.truncated = True
                        break
                    scan_state.scanned_entries += 1
                    entries.append(entry)
        except OSError:
            scan_state.unreadable_dirs += 1
            continue

        child_dirs: list[Path] = []
        for entry in sorted(entries, key=lambda item: item.name):
            path = Path(entry.path)
            try:
                if entry.is_dir(follow_symlinks=False):
                    child_dirs.append(path)
                elif entry.is_file(follow_symlinks=False):
                    scan_state.scanned_files += 1
                    yield path
            except OSError:
                continue
        pending.extend(reversed(child_dirs))


def _collect_execution_artifacts(
    *,
    store: LocalArtifactStore,
    workdir: Path,
    input_files: set[Path],
    request: CodeExecutionRequest,
    artifact_prefix: str,
    metadata: dict[str, str],
    max_artifacts: int | None = None,
    max_artifact_bytes: int | None = None,
    max_artifact_total_bytes: int | None = None,
    max_artifact_candidates: int | None = None,
) -> ExecutionArtifactCollection:
    artifacts: list[GeneratedArtifact] = []
    digest = _execution_artifact_digest(request)
    exported_bytes = 0
    scan_state = WorkdirFileScanState()
    candidate_count = 0
    skipped_too_large = 0
    skipped_count_limit = 0
    skipped_total_bytes_limit = 0
    skipped_unreadable = 0
    max_artifacts = max(CODE_EXECUTION_MAX_ARTIFACTS if max_artifacts is None else max_artifacts, 0)
    max_artifact_bytes = max(
        CODE_EXECUTION_MAX_ARTIFACT_BYTES if max_artifact_bytes is None else max_artifact_bytes,
        0,
    )
    max_artifact_total_bytes = max(
        CODE_EXECUTION_MAX_ARTIFACT_TOTAL_BYTES
        if max_artifact_total_bytes is None
        else max_artifact_total_bytes,
        0,
    )
    max_artifact_candidates = max(
        CODE_EXECUTION_MAX_ARTIFACT_CANDIDATES
        if max_artifact_candidates is None
        else max_artifact_candidates,
        0,
    )

    for path in _iter_workdir_files(
        workdir,
        max_entries=max_artifact_candidates,
        scan_state=scan_state,
    ):
        if not _is_collectable_output(path, workdir, input_files):
            continue
        try:
            byte_count = path.stat().st_size
        except OSError:
            skipped_unreadable += 1
            continue
        candidate_count += 1
        if byte_count > max_artifact_bytes:
            skipped_too_large += 1
            continue
        if len(artifacts) >= max_artifacts:
            skipped_count_limit += 1
            continue
        if exported_bytes + byte_count > max_artifact_total_bytes:
            skipped_total_bytes_limit += 1
            continue
        mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        relative = path.relative_to(workdir).as_posix()
        safe_relative = relative.replace("/", "-")
        artifact = store.copy_file(
            f"{artifact_prefix}/{digest}/{safe_relative}",
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
                    **metadata,
                },
            )
        )
        exported_bytes += byte_count

    skipped_count = (
        skipped_too_large
        + skipped_count_limit
        + skipped_total_bytes_limit
        + skipped_unreadable
    )
    skipped_count += scan_state.unreadable_dirs
    collection_truncated = bool(scan_state.truncated or skipped_count)
    collection_metadata = {
        "max_artifacts": str(max_artifacts),
        "max_artifact_bytes": str(max_artifact_bytes),
        "max_artifact_total_bytes": str(max_artifact_total_bytes),
        "max_artifact_candidates": str(max_artifact_candidates),
        "artifact_scanned_entries": str(scan_state.scanned_entries),
        "artifact_scanned_files": str(scan_state.scanned_files),
        "artifact_candidate_count": str(candidate_count),
        "artifact_exported_count": str(len(artifacts)),
        "artifact_exported_bytes": str(exported_bytes),
        "artifact_skipped_count": str(skipped_count),
        "artifact_skipped_too_large_count": str(skipped_too_large),
        "artifact_skipped_count_limit": str(skipped_count_limit),
        "artifact_skipped_total_bytes_limit": str(skipped_total_bytes_limit),
        "artifact_skipped_unreadable_count": str(skipped_unreadable),
        "artifact_skipped_unreadable_dirs": str(scan_state.unreadable_dirs),
        "artifact_scan_truncated": "true" if scan_state.truncated else "false",
        "artifact_collection_truncated": "true" if collection_truncated else "false",
    }
    return ExecutionArtifactCollection(artifacts=artifacts, metadata=collection_metadata)


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
        return self.write_bytes(relative_path, content.encode("utf-8"), mime_type)

    def write_bytes(self, relative_path: str, content: bytes, mime_type: str) -> GeneratedArtifact:
        path = self._resolve_target(relative_path)
        self._atomic_write_bytes(path, content)
        return self._artifact_for(path, mime_type)

    def copy_file(self, relative_path: str, source: Path, mime_type: str) -> GeneratedArtifact:
        if source.is_symlink() or not source.is_file():
            raise ValueError("Artifact source is not a regular file.")
        return self.write_bytes(relative_path, source.read_bytes(), mime_type)

    def _resolve_target(self, relative_path: str) -> Path:
        relative = Path(relative_path)
        if not relative_path or relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Invalid artifact path: {relative_path}")
        root = self.root.resolve()
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.parent.resolve().relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Artifact path escapes artifact root: {relative_path}") from exc
        return path

    @staticmethod
    def _atomic_write_bytes(path: Path, content: bytes) -> None:
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temp_file:
                temp_file.write(content)
                temp_path = Path(temp_file.name)
            temp_path.replace(path)
        except Exception:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
            raise

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
    """Create deterministic SVG engineering diagrams without external keys."""

    def __init__(self, store: LocalArtifactStore | None = None):
        self.store = store or LocalArtifactStore()

    def generate(self, request: ImageGenerationRequest) -> GeneratedArtifact:
        safe_prompt = html.escape(request.prompt[:240])
        safe_style = html.escape(request.style)
        template = request.diagram_template if request.diagram_template in {
            "generic",
            "sliding-mode-observer",
            "pmsm-control-loop",
            "paper-figure-redraft",
        } else "generic"
        title, body = self._template_svg_body(template)
        svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1024" height="1024" viewBox="0 0 1024 1024">
  <rect width="1024" height="1024" fill="#f8fafc"/>
  <rect x="96" y="128" width="832" height="744" rx="20" fill="#ffffff" stroke="#1f2937" stroke-width="4"/>
  <text x="512" y="210" text-anchor="middle" font-family="Arial, sans-serif" font-size="40" fill="#111827">{title}</text>
  <text x="512" y="314" text-anchor="middle" font-family="Arial, sans-serif" font-size="24" fill="#4b5563">style: {safe_style}</text>
  {body}
  <text x="512" y="812" text-anchor="middle" font-family="Arial, sans-serif" font-size="21" fill="#111827">{safe_prompt}</text>
  <defs>
    <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
      <path d="M0,0 L0,6 L9,3 z" fill="#111827"/>
    </marker>
  </defs>
</svg>
"""
        digest = hashlib.sha256(f"{template}\n{request.style}\n{request.prompt}".encode()).hexdigest()[:12]
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
                "diagram_template": template,
                "reference_uris": json.dumps(request.reference_uris, ensure_ascii=False),
                "cost_estimate_usd": "0",
            },
        )

    @staticmethod
    def _template_svg_body(template: str) -> tuple[str, str]:
        if template == "sliding-mode-observer":
            return (
                "Sliding-Mode Observer",
                """
  <rect x="152" y="420" width="170" height="96" rx="10" fill="#dbeafe" stroke="#2563eb" stroke-width="3"/>
  <text x="237" y="477" text-anchor="middle" font-family="Arial, sans-serif" font-size="22" fill="#1e3a8a">Plant</text>
  <rect x="424" y="420" width="176" height="96" rx="10" fill="#e0f2fe" stroke="#0284c7" stroke-width="3"/>
  <text x="512" y="456" text-anchor="middle" font-family="Arial, sans-serif" font-size="21" fill="#075985">Sliding</text>
  <text x="512" y="484" text-anchor="middle" font-family="Arial, sans-serif" font-size="21" fill="#075985">Surface</text>
  <rect x="704" y="420" width="168" height="96" rx="10" fill="#dcfce7" stroke="#16a34a" stroke-width="3"/>
  <text x="788" y="477" text-anchor="middle" font-family="Arial, sans-serif" font-size="22" fill="#14532d">Observer</text>
  <path d="M324 468 H416" stroke="#111827" stroke-width="5" marker-end="url(#arrow)"/>
  <path d="M604 468 H696" stroke="#111827" stroke-width="5" marker-end="url(#arrow)"/>
  <path d="M788 520 C788 650 237 650 237 522" fill="none" stroke="#7c3aed" stroke-width="5" marker-end="url(#arrow)"/>
  <text x="512" y="630" text-anchor="middle" font-family="Arial, sans-serif" font-size="20" fill="#5b21b6">estimated states feedback</text>
""",
            )
        if template == "pmsm-control-loop":
            return (
                "PMSM Control Loop",
                """
  <rect x="132" y="405" width="155" height="96" rx="10" fill="#fef3c7" stroke="#d97706" stroke-width="3"/>
  <text x="210" y="461" text-anchor="middle" font-family="Arial, sans-serif" font-size="21" fill="#92400e">Reference</text>
  <circle cx="360" cy="453" r="48" fill="#ffffff" stroke="#374151" stroke-width="3"/>
  <text x="360" y="462" text-anchor="middle" font-family="Arial, sans-serif" font-size="34" fill="#111827">Σ</text>
  <rect x="456" y="405" width="160" height="96" rx="10" fill="#dbeafe" stroke="#2563eb" stroke-width="3"/>
  <text x="536" y="461" text-anchor="middle" font-family="Arial, sans-serif" font-size="21" fill="#1e3a8a">SMC</text>
  <rect x="704" y="405" width="150" height="96" rx="10" fill="#dcfce7" stroke="#16a34a" stroke-width="3"/>
  <text x="779" y="461" text-anchor="middle" font-family="Arial, sans-serif" font-size="21" fill="#14532d">PMSM</text>
  <path d="M288 453 H306" stroke="#111827" stroke-width="5" marker-end="url(#arrow)"/>
  <path d="M408 453 H448" stroke="#111827" stroke-width="5" marker-end="url(#arrow)"/>
  <path d="M620 453 H696" stroke="#111827" stroke-width="5" marker-end="url(#arrow)"/>
  <path d="M780 506 C780 656 360 656 360 507" fill="none" stroke="#7c3aed" stroke-width="5" marker-end="url(#arrow)"/>
  <text x="552" y="637" text-anchor="middle" font-family="Arial, sans-serif" font-size="20" fill="#5b21b6">speed/current feedback</text>
""",
            )
        if template == "paper-figure-redraft":
            return (
                "Paper Figure Redraft",
                """
  <rect x="158" y="390" width="708" height="272" rx="12" fill="#f9fafb" stroke="#374151" stroke-width="3"/>
  <path d="M205 604 C300 478 384 560 468 450 C552 340 626 494 792 418" fill="none" stroke="#2563eb" stroke-width="6"/>
  <path d="M210 610 H812" stroke="#111827" stroke-width="3" marker-end="url(#arrow)"/>
  <path d="M210 610 V420" stroke="#111827" stroke-width="3" marker-end="url(#arrow)"/>
  <text x="818" y="646" text-anchor="end" font-family="Arial, sans-serif" font-size="20" fill="#111827">time</text>
  <text x="250" y="420" text-anchor="middle" font-family="Arial, sans-serif" font-size="20" fill="#111827">state</text>
  <rect x="626" y="552" width="178" height="60" rx="8" fill="#ffffff" stroke="#6b7280" stroke-width="2"/>
  <path d="M646 582 H704" stroke="#2563eb" stroke-width="5"/>
  <text x="728" y="589" font-family="Arial, sans-serif" font-size="18" fill="#111827">redraft</text>
""",
            )
        return (
            "FluxMind Engineering Diagram",
            """
  <rect x="192" y="424" width="220" height="112" rx="12" fill="#dbeafe" stroke="#2563eb" stroke-width="3"/>
  <rect x="604" y="424" width="220" height="112" rx="12" fill="#dcfce7" stroke="#16a34a" stroke-width="3"/>
  <path d="M420 480 H596" stroke="#111827" stroke-width="6" marker-end="url(#arrow)"/>
  <text x="302" y="492" text-anchor="middle" font-family="Arial, sans-serif" font-size="24" fill="#1e3a8a">Input</text>
  <text x="714" y="492" text-anchor="middle" font-family="Arial, sans-serif" font-size="24" fill="#14532d">Observer</text>
""",
        )


class LocalPythonExecutionProvider(CodeExecutionProvider):
    """Run Python snippets in a temporary working directory.

    This is a development provider, not a production sandbox. Production use
    still requires a dedicated isolated service with explicit resource limits.
    """

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

        policy = evaluate_request_policy(request)
        initial_metadata = apply_policy_metadata(
            python_execution_metadata(
                request,
                memory_limit_enforced=False,
                cpu_limit_enforced=False,
            ),
            policy,
        )
        policy_failure = execution_policy_failure_result(
            request,
            runtime_metadata=initial_metadata,
            policy=policy,
        )
        if policy_failure is not None:
            return policy_failure

        with tempfile.TemporaryDirectory(prefix="fluxmind-exec-") as tmp:
            workdir = Path(tmp)
            entrypoint, input_files, failure = _materialize_execution_files(
                workdir,
                request,
                runtime_metadata=initial_metadata,
            )
            if failure is not None:
                return failure
            if not entrypoint.exists():
                return CodeExecutionResult(
                    exit_code=2,
                    stdout="",
                    stderr=f"Entrypoint not found: {request.entrypoint}",
                    runtime_metadata=initial_metadata,
                )

            preexec_fn, memory_limit_enforced, cpu_limit_enforced = execution_limit_preexec(
                request.memory_mb,
                request.timeout_s,
            )
            proc = subprocess.Popen(
                [sys.executable, str(entrypoint)],
                cwd=workdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=preexec_fn,
            )
            captured = capture_process_output(
                proc,
                timeout_s=request.timeout_s,
                cancel_event=cancel_event,
            )
            metadata = apply_output_capture_metadata(
                apply_policy_metadata(
                    python_execution_metadata(
                        request,
                        memory_limit_enforced=memory_limit_enforced,
                        cpu_limit_enforced=cpu_limit_enforced,
                    ),
                    policy,
                ),
                captured,
            )
            if captured.cancelled:
                return CodeExecutionResult(
                    exit_code=130,
                    stdout=captured.stdout,
                    stderr=captured.stderr + "Execution cancelled.",
                    runtime_metadata=metadata,
                )
            if captured.timed_out:
                return CodeExecutionResult(
                    exit_code=124,
                    stdout=captured.stdout,
                    stderr=captured.stderr + f"Execution timed out after {request.timeout_s}s",
                    runtime_metadata=metadata,
                )
            artifact_collection = self._collect_artifacts(workdir, input_files, request)
            return CodeExecutionResult(
                exit_code=captured.returncode,
                stdout=captured.stdout,
                stderr=captured.stderr,
                artifacts=artifact_collection.artifacts,
                runtime_metadata={**metadata, **artifact_collection.metadata},
            )

    def _collect_artifacts(
        self,
        workdir: Path,
        input_files: set[Path],
        request: CodeExecutionRequest,
    ) -> ExecutionArtifactCollection:
        return _collect_execution_artifacts(
            store=self.store,
            workdir=workdir,
            input_files=input_files,
            request=request,
            artifact_prefix="code-runs",
            metadata={"cost_estimate_usd": "0"},
        )


class LocalOctaveExecutionProvider(CodeExecutionProvider):
    """Run GNU Octave-compatible scripts when a local octave binary exists.

    This is still a development provider. It does not activate MATLAB or a
    hosted sandbox; it only provides the no-key execution contract and artifact
    capture surface.
    """

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

        policy = evaluate_request_policy(request)
        missing_runtime_metadata = apply_policy_metadata(
            octave_execution_metadata(
                request,
                memory_limit_enforced=False,
                cpu_limit_enforced=False,
                executable=self.executable,
                resolved_executable=None,
            ),
            policy,
        )
        policy_failure = execution_policy_failure_result(
            request,
            runtime_metadata=missing_runtime_metadata,
            policy=policy,
        )
        if policy_failure is not None:
            return policy_failure

        octave_bin = shutil.which(self.executable)
        if octave_bin is None:
            return CodeExecutionResult(
                exit_code=127,
                stdout="",
                stderr=(
                    "GNU Octave executable not found. Install octave or attach "
                    "a hosted execution provider."
                ),
                runtime_metadata=missing_runtime_metadata,
            )

        with tempfile.TemporaryDirectory(prefix="fluxmind-octave-") as tmp:
            workdir = Path(tmp)
            initial_metadata = apply_policy_metadata(
                octave_execution_metadata(
                    request,
                    memory_limit_enforced=False,
                    cpu_limit_enforced=False,
                    executable=self.executable,
                    resolved_executable=octave_bin,
                ),
                policy,
            )
            entrypoint, input_files, failure = _materialize_execution_files(
                workdir,
                request,
                runtime_metadata=initial_metadata,
            )
            if failure is not None:
                return failure
            if not entrypoint.exists():
                return CodeExecutionResult(
                    exit_code=2,
                    stdout="",
                    stderr=f"Entrypoint not found: {request.entrypoint}",
                    runtime_metadata=initial_metadata,
                )

            preexec_fn, memory_limit_enforced, cpu_limit_enforced = execution_limit_preexec(
                request.memory_mb,
                request.timeout_s,
            )
            proc = subprocess.Popen(
                [octave_bin, "--quiet", "--no-gui", str(entrypoint)],
                cwd=workdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=preexec_fn,
            )
            captured = capture_process_output(
                proc,
                timeout_s=request.timeout_s,
                cancel_event=cancel_event,
            )
            metadata = apply_output_capture_metadata(
                apply_policy_metadata(
                    octave_execution_metadata(
                        request,
                        memory_limit_enforced=memory_limit_enforced,
                        cpu_limit_enforced=cpu_limit_enforced,
                        executable=self.executable,
                        resolved_executable=octave_bin,
                    ),
                    policy,
                ),
                captured,
            )
            if captured.cancelled:
                return CodeExecutionResult(
                    exit_code=130,
                    stdout=captured.stdout,
                    stderr=captured.stderr + "Execution cancelled.",
                    runtime_metadata=metadata,
                )
            if captured.timed_out:
                return CodeExecutionResult(
                    exit_code=124,
                    stdout=captured.stdout,
                    stderr=captured.stderr + f"Execution timed out after {request.timeout_s}s",
                    runtime_metadata=metadata,
                )
            artifact_collection = self._collect_artifacts(workdir, input_files, request)
            return CodeExecutionResult(
                exit_code=captured.returncode,
                stdout=captured.stdout,
                stderr=captured.stderr,
                artifacts=artifact_collection.artifacts,
                runtime_metadata={**metadata, **artifact_collection.metadata},
            )

    def _collect_artifacts(
        self,
        workdir: Path,
        input_files: set[Path],
        request: CodeExecutionRequest,
    ) -> ExecutionArtifactCollection:
        return _collect_execution_artifacts(
            store=self.store,
            workdir=workdir,
            input_files=input_files,
            request=request,
            artifact_prefix="octave-runs",
            metadata={"runtime": "gnu-octave-local", "cost_estimate_usd": "0"},
        )


class DockerExecutionProvider(CodeExecutionProvider):
    """Run Python/Octave-compatible snippets inside a local Docker container."""

    def __init__(
        self,
        store: LocalArtifactStore | None = None,
        *,
        image: str = DOCKER_EXECUTION_IMAGE,
    ):
        self.store = store or LocalArtifactStore()
        self.image = image

    def run(
        self,
        request: CodeExecutionRequest,
        *,
        cancel_event: threading.Event | None = None,
    ) -> CodeExecutionResult:
        container_command = self._container_command(request)
        container_name = f"fluxmind-exec-{hashlib.sha256(str(time.time_ns()).encode()).hexdigest()[:16]}"
        docker_path = shutil.which("docker")
        container_user = self._container_user()
        policy = evaluate_request_policy(request)
        base_metadata = apply_policy_metadata(
            docker_execution_metadata(
                request,
                image=self.image,
                docker_path=docker_path,
                runtime_available=docker_path is not None,
                container_name=container_name,
                container_command=container_command,
                container_user=container_user,
            ),
            policy,
        )
        policy_failure = execution_policy_failure_result(
            request,
            runtime_metadata=base_metadata,
            policy=policy,
        )
        if policy_failure is not None:
            return policy_failure
        if docker_path is None:
            return CodeExecutionResult(
                exit_code=127,
                stdout="",
                stderr="Docker executable not found. Configure Docker before using CODE_EXECUTION_BACKEND=docker.",
                runtime_metadata=base_metadata,
            )
        if container_command == []:
            return CodeExecutionResult(
                exit_code=2,
                stdout="",
                stderr=f"Unsupported Docker execution language: {request.language}",
                runtime_metadata=base_metadata,
            )

        with tempfile.TemporaryDirectory(prefix="fluxmind-docker-") as tmp:
            workdir = Path(tmp)
            entrypoint, input_files, failure = _materialize_execution_files(
                workdir,
                request,
                runtime_metadata=base_metadata,
            )
            if failure is not None:
                return failure
            if not entrypoint.exists():
                return CodeExecutionResult(
                    exit_code=2,
                    stdout="",
                    stderr=f"Entrypoint not found: {request.entrypoint}",
                    runtime_metadata=base_metadata,
                )

            command = self._docker_command(
                docker_path=docker_path,
                workdir=workdir,
                container_name=container_name,
                container_user=container_user,
                container_command=container_command,
                memory_mb=request.memory_mb,
            )
            try:
                proc = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            except OSError as exc:
                metadata = dict(base_metadata)
                metadata["runtime_available"] = "false"
                metadata["docker_error"] = exc.__class__.__name__
                return CodeExecutionResult(
                    exit_code=127,
                    stdout="",
                    stderr=f"Docker execution backend unavailable: {exc}",
                    runtime_metadata=metadata,
                )

            captured = capture_process_output(
                proc,
                timeout_s=request.timeout_s,
                cancel_event=cancel_event,
                on_cancel=lambda: self._force_remove_container(docker_path, container_name),
                on_timeout=lambda: self._force_remove_container(docker_path, container_name),
            )
            if captured.cancelled:
                return CodeExecutionResult(
                    exit_code=130,
                    stdout=captured.stdout,
                    stderr=captured.stderr + "Execution cancelled.",
                    runtime_metadata=apply_output_capture_metadata(base_metadata, captured),
                )
            if captured.timed_out:
                return CodeExecutionResult(
                    exit_code=124,
                    stdout=captured.stdout,
                    stderr=captured.stderr + f"Execution timed out after {request.timeout_s}s",
                    runtime_metadata=apply_output_capture_metadata(base_metadata, captured),
                )
            metadata = docker_execution_metadata(
                request,
                image=self.image,
                docker_path=docker_path,
                runtime_available=captured.returncode not in {125, 126, 127},
                container_name=container_name,
                container_command=container_command,
                container_user=container_user,
                docker_returncode=captured.returncode,
            )
            metadata = apply_policy_metadata(metadata, policy)
            metadata = apply_output_capture_metadata(metadata, captured)
            exit_code = 127 if captured.returncode in {125, 126} else captured.returncode
            artifact_collection = self._collect_artifacts(workdir, input_files, request)
            return CodeExecutionResult(
                exit_code=exit_code,
                stdout=captured.stdout,
                stderr=captured.stderr,
                artifacts=artifact_collection.artifacts,
                runtime_metadata={**metadata, **artifact_collection.metadata},
            )

    @staticmethod
    def _container_user() -> str:
        if hasattr(os, "getuid") and hasattr(os, "getgid"):
            return f"{os.getuid()}:{os.getgid()}"
        return ""

    @staticmethod
    def _container_command(request: CodeExecutionRequest) -> list[str]:
        if request.language == "python":
            return ["python", request.entrypoint]
        if request.language in {"octave", "matlab"}:
            return ["octave", "--quiet", "--no-gui", request.entrypoint]
        return []

    def _docker_command(
        self,
        *,
        docker_path: str,
        workdir: Path,
        container_name: str,
        container_user: str,
        container_command: list[str],
        memory_mb: int,
    ) -> list[str]:
        command = [
            docker_path,
            "run",
            "--rm",
            "--name",
            container_name,
            "--network",
            "none",
            "--memory",
            f"{max(memory_mb, 64)}m",
            "--cpus",
            "1",
            "--pids-limit",
            "64",
            "--security-opt",
            "no-new-privileges",
            "--cap-drop",
            "ALL",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,size=64m",
            "-v",
            f"{workdir.resolve()}:/work:rw",
            "-w",
            "/work",
            "--env",
            "PYTHONUNBUFFERED=1",
        ]
        if container_user:
            command.extend(["--user", container_user])
        command.extend([self.image, *container_command])
        return command

    @staticmethod
    def _force_remove_container(docker_path: str, container_name: str) -> None:
        subprocess.run(
            [docker_path, "rm", "-f", container_name],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def _collect_artifacts(
        self,
        workdir: Path,
        input_files: set[Path],
        request: CodeExecutionRequest,
    ) -> ExecutionArtifactCollection:
        return _collect_execution_artifacts(
            store=self.store,
            workdir=workdir,
            input_files=input_files,
            request=request,
            artifact_prefix="docker-runs",
            metadata={
                "runtime": f"docker-{request.language}",
                "docker_image": self.image,
                "cost_estimate_usd": "0",
            },
        )
