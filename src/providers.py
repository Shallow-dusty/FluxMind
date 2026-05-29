"""No-key provider implementations for future platform capabilities.

These providers are deliberately local and replaceable. They let the app
develop artifact and execution flows without consuming real image-provider keys,
hosted sandbox accounts, or MATLAB licenses.
"""

from __future__ import annotations

import html
import hashlib
import mimetypes
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
        return GeneratedArtifact(
            kind=kind,
            uri=path.resolve().as_uri(),
            mime_type=mime_type,
            title=path.name,
            metadata={"provider": "local"},
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
        return self.store.write_text(f"mock-images/{name}", svg, "image/svg+xml")


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
            )

        with tempfile.TemporaryDirectory(prefix="fluxmind-exec-") as tmp:
            workdir = Path(tmp)
            input_files = set()
            for name, content in request.files.items():
                target = workdir / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
                input_files.add(target.resolve())

            entrypoint = workdir / request.entrypoint
            if not entrypoint.exists():
                return CodeExecutionResult(
                    exit_code=2,
                    stdout="",
                    stderr=f"Entrypoint not found: {request.entrypoint}",
                )

            proc = subprocess.Popen(
                [sys.executable, str(entrypoint)],
                cwd=workdir,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
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
                    )
                time.sleep(0.05)

            stdout, stderr = proc.communicate()
            return CodeExecutionResult(
                exit_code=proc.returncode,
                stdout=stdout,
                stderr=stderr,
                artifacts=self._collect_artifacts(workdir, input_files, request),
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
            if path.resolve() in input_files:
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
                    metadata={**artifact.metadata, "source": relative},
                )
            )
        return artifacts
