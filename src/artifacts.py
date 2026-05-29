"""Local artifact listing and export helpers."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

from src.config import ARTIFACTS_DIR
from src.jobs import LocalJobStore


@dataclass(frozen=True)
class ArtifactRecord:
    """Exportable artifact metadata derived from persisted jobs."""

    artifact_id: str
    job_id: str
    job_kind: str
    kind: str
    uri: str
    mime_type: str
    title: str | None = None
    metadata: dict | None = None


def artifact_id_for_uri(uri: str) -> str:
    return hashlib.sha256(uri.encode()).hexdigest()[:16]


def local_artifact_path(uri: str) -> Path:
    """Resolve a file artifact URI and require it to stay under ARTIFACTS_DIR."""
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        raise ValueError("Only local file artifacts can be exported.")
    path = Path(unquote(parsed.path)).resolve()
    root = ARTIFACTS_DIR.resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("Artifact path escapes the local artifact directory.") from exc
    if not path.is_file():
        raise FileNotFoundError("Artifact file does not exist.")
    return path


class LocalArtifactRegistry:
    """Read artifact metadata from persisted local jobs."""

    def __init__(self, job_store: LocalJobStore | None = None):
        self.job_store = job_store or LocalJobStore()

    def list_artifacts(self, *, limit: int = 100) -> list[ArtifactRecord]:
        records: list[ArtifactRecord] = []
        for job in self.job_store.list_latest(limit=limit):
            for artifact in job.artifacts:
                uri = artifact.get("uri", "")
                if not uri:
                    continue
                records.append(
                    ArtifactRecord(
                        artifact_id=artifact_id_for_uri(uri),
                        job_id=job.job_id,
                        job_kind=job.kind,
                        kind=artifact.get("kind", "file"),
                        uri=uri,
                        mime_type=artifact.get("mime_type", "application/octet-stream"),
                        title=artifact.get("title"),
                        metadata=artifact.get("metadata") or {},
                    )
                )
        return records[:limit]

    def get_artifact(self, artifact_id: str) -> ArtifactRecord | None:
        for artifact in self.list_artifacts(limit=500):
            if artifact.artifact_id == artifact_id:
                return artifact
        return None

    def export_path(self, artifact_id: str) -> tuple[ArtifactRecord, Path]:
        artifact = self.get_artifact(artifact_id)
        if artifact is None:
            raise FileNotFoundError("Artifact not found.")
        return artifact, local_artifact_path(artifact.uri)
