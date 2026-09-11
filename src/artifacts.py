"""Local artifact listing and export helpers."""

from __future__ import annotations

import hashlib
import json
import mimetypes
import re
import sqlite3
import stat
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

from src.config import ARTIFACTS_DIR
from src.jobs import DEFAULT_OWNER_ID, DEFAULT_OWNER_LABEL, JobRecord, LocalJobStore, ownership_from_record


_MIME_EXTENSION_FALLBACKS = {
    "image/svg+xml": ".svg",
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "text/plain": ".txt",
    "application/json": ".json",
}
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
    owner_id: str = DEFAULT_OWNER_ID
    owner_label: str = DEFAULT_OWNER_LABEL
    ownership_source: str = "default"


def artifact_id_for_uri(uri: str) -> str:
    return hashlib.sha256(uri.encode()).hexdigest()[:16]


def _safe_download_suffix(mime_type: str, path: Path | None = None) -> str:
    suffix = _MIME_EXTENSION_FALLBACKS.get(mime_type.strip().lower())
    if suffix is None:
        suffix = mimetypes.guess_extension(mime_type.strip().lower() or "") or ""
    if not suffix and path is not None:
        suffix = path.suffix
    suffix = suffix.lower()
    if not suffix.startswith(".") or len(suffix) > 16:
        return ""
    if not all(char.isalnum() or char in {".", "_", "-"} for char in suffix):
        return ""
    return suffix


def _safe_filename_artifact_id(artifact_id: str) -> str:
    normalized = artifact_id.strip().lower()
    if normalized and len(normalized) <= 64 and all(
        char in "0123456789abcdef" for char in normalized
    ):
        return normalized
    return hashlib.sha256(artifact_id.encode()).hexdigest()[:16]


def _download_title(value: str | None) -> str:
    title = re.sub(r"[\x00-\x1f\x7f/\\:*?\"<>|]+", "-", (value or "").strip())
    title = re.sub(r"\s+", " ", title).strip(" .")
    return title[:120].rstrip(" .")


def safe_artifact_download_filename(artifact: ArtifactRecord, path: Path | None = None) -> str:
    """Return a useful, header-safe filename for a generated artifact."""
    suffix = _safe_download_suffix(artifact.mime_type, path)
    title = _download_title(artifact.title)
    if not title and path is not None:
        title = _download_title(path.name)
    if not title:
        title = f"artifact-{_safe_filename_artifact_id(artifact.artifact_id)}"
    if suffix:
        title_path = Path(title)
        if title_path.suffix.lower() != suffix:
            title = f"{title_path.stem or title}{suffix}"
    return title


def artifact_metadata(metadata: dict | None) -> dict[str, object]:
    result = dict(metadata or {})
    references = result.get("reference_uris")
    if isinstance(references, str):
        try:
            decoded = json.loads(references)
        except json.JSONDecodeError:
            decoded = references
        result["reference_uris"] = decoded
    return result


def artifact_to_dict(artifact: ArtifactRecord) -> dict[str, object]:
    """Return useful artifact metadata while keeping the internal file URI private."""
    return {
        "artifact_id": artifact.artifact_id,
        "job_id": artifact.job_id,
        "job_kind": artifact.job_kind,
        "kind": artifact.kind,
        "mime_type": artifact.mime_type,
        "title": artifact.title,
        "metadata": artifact_metadata(artifact.metadata),
        "owner_id": artifact.owner_id,
        "owner_label": artifact.owner_label,
        "ownership_source": artifact.ownership_source,
    }


def job_artifact_to_dict(job: JobRecord, artifact: dict) -> dict[str, object]:
    """Return an embedded job artifact in the same shape as registry artifacts."""
    uri = str(artifact.get("uri") or "")
    ownership = ownership_from_record(job)
    return artifact_to_dict(
        ArtifactRecord(
            artifact_id=artifact_id_for_uri(uri) if uri else "",
            job_id=job.job_id,
            job_kind=job.kind,
            kind=str(artifact.get("kind") or "file"),
            uri=uri,
            mime_type=str(artifact.get("mime_type") or "application/octet-stream"),
            title=artifact.get("title"),
            metadata=artifact.get("metadata") or {},
            owner_id=ownership["owner_id"],
            owner_label=ownership["owner_label"],
            ownership_source=ownership["ownership_source"],
        )
    )


def local_artifact_path(uri: str) -> Path:
    """Resolve a file artifact URI and require it to stay under ARTIFACTS_DIR."""
    parsed = urlparse(uri)
    if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
        raise ValueError("Only local file artifacts can be exported.")
    path = Path(unquote(parsed.path))
    if not path.is_absolute():
        raise ValueError("Artifact file URI must be absolute.")
    root = ARTIFACTS_DIR.resolve()
    try:
        path_stat = path.lstat()
    except FileNotFoundError:
        raise FileNotFoundError("Artifact file does not exist.")
    if stat.S_ISLNK(path_stat.st_mode):
        raise ValueError("Artifact symlinks cannot be exported.")
    if not stat.S_ISREG(path_stat.st_mode):
        raise FileNotFoundError("Artifact file does not exist.")
    try:
        resolved_path = path.resolve(strict=True)
        resolved_path.relative_to(root)
    except FileNotFoundError:
        raise FileNotFoundError("Artifact file does not exist.")
    except ValueError as exc:
        raise ValueError("Artifact path escapes the local artifact directory.") from exc
    return resolved_path


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
        owner_id: str | None = None,
        q: str | None = None,
    ) -> list[ArtifactRecord]:
        limit = max(limit, 0)
        records, synced_job_ids = self._records_from_jobs(limit=max(limit, 1000))
        self._sync_sqlite(records, synced_job_ids=synced_job_ids)
        return self._filter_records(records, kind=kind, job_kind=job_kind, owner_id=owner_id, q=q)[:limit]

    @staticmethod
    def _filter_records(
        records: list[ArtifactRecord],
        *,
        kind: str | None = None,
        job_kind: str | None = None,
        owner_id: str | None = None,
        q: str | None = None,
    ) -> list[ArtifactRecord]:
        kind = kind.strip() if kind else None
        job_kind = job_kind.strip() if job_kind else None
        owner_id = owner_id.strip() if owner_id else None
        query = (q or "").strip().casefold()
        filtered: list[ArtifactRecord] = []
        for record in records:
            if kind and record.kind != kind:
                continue
            if job_kind and record.job_kind != job_kind:
                continue
            if owner_id and record.owner_id != owner_id:
                continue
            if query:
                searchable = json.dumps(
                    artifact_to_dict(record),
                    ensure_ascii=False,
                    sort_keys=True,
                ).casefold()
                if query not in searchable:
                    continue
            filtered.append(record)
        return filtered

    def _records_from_jobs(self, *, limit: int = 100) -> tuple[list[ArtifactRecord], set[str]]:
        records: list[ArtifactRecord] = []
        synced_job_ids: set[str] = set()
        for job in self.job_store.list_latest(limit=limit):
            synced_job_ids.add(job.job_id)
            records.extend(self._records_from_job(job))
        return records, synced_job_ids

    @staticmethod
    def _records_from_job(job: JobRecord) -> list[ArtifactRecord]:
        records: list[ArtifactRecord] = []
        ownership = ownership_from_record(job)
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
                    owner_id=ownership["owner_id"],
                    owner_label=ownership["owner_label"],
                    ownership_source=ownership["ownership_source"],
                )
            )
        return records

    @staticmethod
    def _record_from_records(
        records: list[ArtifactRecord],
        artifact_id: str,
    ) -> ArtifactRecord | None:
        for record in records:
            if record.artifact_id == artifact_id:
                return record
        return None

    def get_artifact(self, artifact_id: str) -> ArtifactRecord | None:
        artifact_id = artifact_id.strip()
        if not artifact_id:
            return None
        record = self._get_sqlite(artifact_id)
        if record is not None:
            current_job = self.job_store.get(record.job_id)
            if current_job is not None:
                current_records = self._records_from_job(current_job)
                current_record = self._record_from_records(current_records, artifact_id)
                if current_record is not None:
                    self._sync_sqlite(current_records, synced_job_ids={current_record.job_id})
                    return current_record
                self._sync_sqlite([], synced_job_ids={record.job_id})

        for job in self.job_store.list_all_latest():
            job_records = self._records_from_job(job)
            record = self._record_from_records(job_records, artifact_id)
            if record is not None:
                self._sync_sqlite(job_records, synced_job_ids={record.job_id})
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
        """Verify local artifact files against persisted metadata."""
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
                    owner_id TEXT NOT NULL DEFAULT 'local-user',
                    owner_label TEXT NOT NULL DEFAULT 'Local user',
                    ownership_source TEXT NOT NULL DEFAULT 'default',
                    payload TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_artifacts_job_id ON artifacts(job_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_artifacts_kind ON artifacts(kind)")
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(artifacts)").fetchall()}
            for column, definition in {
                "owner_id": "TEXT NOT NULL DEFAULT 'local-user'",
                "owner_label": "TEXT NOT NULL DEFAULT 'Local user'",
                "ownership_source": "TEXT NOT NULL DEFAULT 'default'",
            }.items():
                if column not in columns:
                    conn.execute(f"ALTER TABLE artifacts ADD COLUMN {column} {definition}")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_artifacts_owner_id ON artifacts(owner_id)")

    def _sync_sqlite(
        self,
        records: list[ArtifactRecord],
        *,
        synced_job_ids: set[str],
    ) -> None:
        if not records and not synced_job_ids and not self.db_path.exists():
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
                        title, metadata, owner_id, owner_label, ownership_source,
                        payload
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(artifact_id) DO UPDATE SET
                        job_id=excluded.job_id,
                        job_kind=excluded.job_kind,
                        kind=excluded.kind,
                        uri=excluded.uri,
                        mime_type=excluded.mime_type,
                        title=excluded.title,
                        metadata=excluded.metadata,
                        owner_id=excluded.owner_id,
                        owner_label=excluded.owner_label,
                        ownership_source=excluded.ownership_source,
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
                        record.owner_id,
                        record.owner_label,
                        record.ownership_source,
                        json.dumps(asdict(record), ensure_ascii=False),
                    ),
                )
            if synced_job_ids:
                job_placeholders = ",".join("?" for _ in synced_job_ids)
                job_params = tuple(sorted(synced_job_ids))
                if seen:
                    artifact_placeholders = ",".join("?" for _ in seen)
                    conn.execute(
                        f"""
                        DELETE FROM artifacts
                        WHERE job_id IN ({job_placeholders})
                          AND artifact_id NOT IN ({artifact_placeholders})
                        """,
                        job_params + tuple(sorted(seen)),
                    )
                else:
                    conn.execute(
                        f"DELETE FROM artifacts WHERE job_id IN ({job_placeholders})",
                        job_params,
                    )
            else:
                conn.execute(
                    "DELETE FROM artifacts",
                )

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
        try:
            payload = json.loads(row["payload"])
            if not isinstance(payload, dict):
                return None
            payload.setdefault("owner_id", DEFAULT_OWNER_ID)
            payload.setdefault("owner_label", DEFAULT_OWNER_LABEL)
            payload.setdefault("ownership_source", "default")
            return ArtifactRecord(**payload)
        except (json.JSONDecodeError, TypeError, ValueError):
            return None


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
        artifact_data = artifact_to_dict(artifact)
        metadata = artifact_data["metadata"]
        references = metadata.get("reference_uris") or []
        details = [
            f"kind={artifact_data['kind']}",
            f"mime={artifact_data['mime_type']}",
            f"job_kind={artifact_data['job_kind']}",
        ]
        if artifact.title:
            details.append(f"title={artifact.title}")
        if metadata.get("provider") or metadata.get("model"):
            details.append(f"provider={metadata.get('provider') or metadata.get('model')}")
        if metadata.get("style"):
            details.append(f"style={metadata['style']}")
        if metadata.get("diagram_template"):
            details.append(f"template={metadata['diagram_template']}")
        if references:
            details.append(f"reference_count={len(references)}")
        if metadata.get("byte_count"):
            details.append(f"bytes={metadata['byte_count']}")
        parts.append(f"[Artifact:{artifact.artifact_id}] " + "; ".join(details))
    return "\n".join(parts)
