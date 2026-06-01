"""Local artifact listing and export helpers."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass
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

    def __init__(self, job_store: LocalJobStore | None = None, db_path: Path | None = None):
        self.job_store = job_store or LocalJobStore()
        self.db_path = db_path or (ARTIFACTS_DIR / "artifacts.sqlite3")

    def list_artifacts(
        self,
        *,
        limit: int = 100,
        kind: str | None = None,
        job_kind: str | None = None,
        q: str | None = None,
    ) -> list[ArtifactRecord]:
        records = self._records_from_jobs(limit=max(limit, 1000))
        self._sync_sqlite(records)
        return self._filter_records(records, kind=kind, job_kind=job_kind, q=q)[:limit]

    @staticmethod
    def _filter_records(
        records: list[ArtifactRecord],
        *,
        kind: str | None = None,
        job_kind: str | None = None,
        q: str | None = None,
    ) -> list[ArtifactRecord]:
        kind = kind.strip() if kind else None
        job_kind = job_kind.strip() if job_kind else None
        query = (q or "").strip().casefold()
        filtered: list[ArtifactRecord] = []
        for record in records:
            if kind and record.kind != kind:
                continue
            if job_kind and record.job_kind != job_kind:
                continue
            if query:
                searchable = " ".join(
                    str(value or "")
                    for value in (
                        record.artifact_id,
                        record.job_id,
                        record.job_kind,
                        record.kind,
                        record.uri,
                        record.mime_type,
                        record.title,
                        json.dumps(record.metadata or {}, ensure_ascii=False, sort_keys=True),
                    )
                ).casefold()
                if query not in searchable:
                    continue
            filtered.append(record)
        return filtered

    def _records_from_jobs(self, *, limit: int = 100) -> list[ArtifactRecord]:
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
        self.list_artifacts(limit=500)
        record = self._get_sqlite(artifact_id)
        if record is not None:
            return record
        return None

    def export_path(self, artifact_id: str) -> tuple[ArtifactRecord, Path]:
        artifact = self.get_artifact(artifact_id)
        if artifact is None:
            raise FileNotFoundError("Artifact not found.")
        return artifact, local_artifact_path(artifact.uri)

    def storage_status(self) -> dict:
        """Return local artifact metadata storage state without exposing contents."""
        self.list_artifacts(limit=1000)
        row_count = 0
        if self.db_path.exists():
            with self._connect() as conn:
                row = conn.execute("SELECT COUNT(*) AS count FROM artifacts").fetchone()
                row_count = int(row["count"]) if row else 0
        return {
            "sqlite_exists": self.db_path.exists(),
            "sqlite_bytes": self.db_path.stat().st_size if self.db_path.exists() else 0,
            "sqlite_rows": row_count,
        }

    def integrity_status(self, *, limit: int = 1000) -> dict:
        """Verify local artifact files against persisted no-secret metadata."""
        records = self.list_artifacts(limit=limit)
        status = {
            "checked": 0,
            "ok": 0,
            "missing": 0,
            "unchecked": 0,
            "byte_count_mismatch": 0,
            "checksum_mismatch": 0,
            "issue_artifact_ids": [],
        }
        for record in records:
            metadata = record.metadata or {}
            expected_bytes = str(metadata.get("byte_count") or "")
            expected_checksum = str(metadata.get("checksum_sha256") or "")
            if not expected_bytes or not expected_checksum:
                self._record_integrity_issue(status, record.artifact_id, "unchecked")
                continue
            try:
                path = local_artifact_path(record.uri)
            except (FileNotFoundError, ValueError):
                self._record_integrity_issue(status, record.artifact_id, "missing")
                continue

            content = path.read_bytes()
            status["checked"] += 1
            mismatch = False
            if str(len(content)) != expected_bytes:
                status["byte_count_mismatch"] += 1
                mismatch = True
            if hashlib.sha256(content).hexdigest() != expected_checksum:
                status["checksum_mismatch"] += 1
                mismatch = True
            if mismatch:
                if len(status["issue_artifact_ids"]) < 5:
                    status["issue_artifact_ids"].append(record.artifact_id)
            else:
                status["ok"] += 1
        return status

    @staticmethod
    def _record_integrity_issue(status: dict, artifact_id: str, key: str) -> None:
        status[key] += 1
        if len(status["issue_artifact_ids"]) < 5:
            status["issue_artifact_ids"].append(artifact_id)

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_sqlite(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    job_kind TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    uri TEXT NOT NULL,
                    mime_type TEXT NOT NULL,
                    title TEXT,
                    metadata TEXT NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_artifacts_job_id ON artifacts(job_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_artifacts_kind ON artifacts(kind)")

    def _sync_sqlite(self, records: list[ArtifactRecord]) -> None:
        if not records and not self.db_path.exists():
            return
        self._ensure_sqlite()
        with self._connect() as conn:
            seen = set()
            for record in records:
                seen.add(record.artifact_id)
                conn.execute(
                    """
                    INSERT INTO artifacts (
                        artifact_id, job_id, job_kind, kind, uri, mime_type,
                        title, metadata, payload
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(artifact_id) DO UPDATE SET
                        job_id=excluded.job_id,
                        job_kind=excluded.job_kind,
                        kind=excluded.kind,
                        uri=excluded.uri,
                        mime_type=excluded.mime_type,
                        title=excluded.title,
                        metadata=excluded.metadata,
                        payload=excluded.payload
                    """,
                    (
                        record.artifact_id,
                        record.job_id,
                        record.job_kind,
                        record.kind,
                        record.uri,
                        record.mime_type,
                        record.title,
                        json.dumps(record.metadata or {}, ensure_ascii=False),
                        json.dumps(asdict(record), ensure_ascii=False),
                    ),
                )
            if seen:
                placeholders = ",".join("?" for _ in seen)
                conn.execute(
                    f"DELETE FROM artifacts WHERE artifact_id NOT IN ({placeholders})",
                    tuple(sorted(seen)),
                )
            else:
                conn.execute("DELETE FROM artifacts")

    def _get_sqlite(self, artifact_id: str) -> ArtifactRecord | None:
        if not self.db_path.exists():
            return None
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM artifacts WHERE artifact_id = ?",
                (artifact_id,),
            ).fetchone()
        if row is None:
            return None
        return ArtifactRecord(**json.loads(row["payload"]))


def format_artifact_references(
    artifacts: list[ArtifactRecord],
    *,
    limit: int = 5,
) -> str:
    """Format generated artifacts so RAG answers can cite stable artifact IDs."""
    if not artifacts:
        return "(No generated artifacts are currently available.)"

    parts: list[str] = []
    for artifact in artifacts[:limit]:
        metadata = artifact.metadata or {}
        prompt = str(metadata.get("prompt") or "").strip()
        style = str(metadata.get("style") or "").strip()
        diagram_template = str(metadata.get("diagram_template") or "").strip()
        source_refs = str(metadata.get("reference_uris") or "").strip()
        details = [
            f"kind={artifact.kind}",
            f"mime={artifact.mime_type}",
            f"job={artifact.job_id}",
        ]
        if artifact.title:
            details.append(f"title={artifact.title}")
        if style:
            details.append(f"style={style}")
        if diagram_template:
            details.append(f"template={diagram_template}")
        if source_refs:
            details.append(f"references={source_refs}")
        if prompt:
            details.append(f"prompt={prompt}")
        parts.append(f"[Artifact:{artifact.artifact_id}] " + "; ".join(details))
    return "\n".join(parts)
