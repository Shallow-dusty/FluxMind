"""Provider-neutral capability contracts for FluxMind platform features.

These protocols keep image generation and code execution integrations behind
explicit interfaces instead of coupling them to Streamlit or the RAG chain.
Local no-key development providers live in `src.providers`; real external
providers can be added behind the same contracts later.
"""

from dataclasses import dataclass, field
from typing import Literal, Protocol


ArtifactKind = Literal["image", "plot", "file", "text"]
CodeLanguage = Literal["python", "octave", "matlab"]


@dataclass(frozen=True)
class GeneratedArtifact:
    """Metadata for an artifact produced by a model or execution backend."""

    kind: ArtifactKind
    uri: str
    mime_type: str
    title: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ImageGenerationRequest:
    """Provider-neutral image generation request."""

    prompt: str
    style: str = "engineering-diagram"
    size: str = "1024x1024"
    reference_uris: list[str] = field(default_factory=list)


class ImageGenerationProvider(Protocol):
    """Generate visual artifacts such as diagrams or paper-figure redrafts."""

    def generate(self, request: ImageGenerationRequest) -> GeneratedArtifact:
        ...


@dataclass(frozen=True)
class CodeExecutionRequest:
    """Provider-neutral isolated code execution request."""

    language: CodeLanguage
    entrypoint: str
    files: dict[str, str] = field(default_factory=dict)
    timeout_s: int = 30
    memory_mb: int = 512


@dataclass(frozen=True)
class CodeExecutionResult:
    """Captured result from an isolated execution backend."""

    exit_code: int
    stdout: str
    stderr: str
    artifacts: list[GeneratedArtifact] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.exit_code == 0


class CodeExecutionProvider(Protocol):
    """Run generated Python/Octave/MATLAB-compatible code in isolation."""

    def run(self, request: CodeExecutionRequest) -> CodeExecutionResult:
        ...
