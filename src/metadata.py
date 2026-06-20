"""Local corpus metadata registry.

This is the first storage boundary for papers and indexing state. It is a
JSON-backed development store, not the final multi-user database.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from src.config import (
    CHUNK_METADATA_DB_FILE,
    CORPUS_METADATA_DB_FILE,
    CORPUS_METADATA_FILE,
    CORPUS_PROFILES_FILE,
    PAPERS_LIBRARY_DIR,
    PROJECT_ROOT,
)


PaperSourceKind = Literal["library", "upload", "paper"]
PaperIndexStatus = Literal["available", "active", "indexed", "failed"]
_ARXIV_RE = re.compile(r"arxiv(?:\.org/(?:abs|pdf)/|-)(\d{4}\.\d{4,5})(?:v\d+)?", re.IGNORECASE)
_SENSITIVE_PROFILE_FILENAME_RE = re.compile(
    r"(authorization|bearer|api[-_\s]?key|token|secret|sk-[a-z0-9])",
    re.IGNORECASE,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def project_relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON through a same-directory temp file, then atomically replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
            encoding="utf-8",
        ) as handle:
            temp_name = handle.name
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        Path(temp_name).replace(path)
    finally:
        if temp_name:
            temp_path = Path(temp_name)
            if temp_path.exists():
                temp_path.unlink()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _first_nonempty(*values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return None


def _derive_arxiv_id(path: Path, metadata: dict[str, Any]) -> str | None:
    for value in (
        metadata.get("arxiv_id"),
        metadata.get("source_url"),
        metadata.get("pdf_url"),
        path.name,
    ):
        if not value:
            continue
        match = _ARXIV_RE.search(str(value))
        if match:
            return match.group(1)
    return metadata.get("arxiv_id")


def _normalize_topic_tags(*values: Any) -> list[str]:
    tags: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value is None:
            continue
        candidates = value if isinstance(value, list) else re.split(r"[,;/|]+", str(value))
        for candidate in candidates:
            tag = str(candidate).strip()
            if not tag:
                continue
            key = tag.casefold()
            if key in seen:
                continue
            seen.add(key)
            tags.append(tag)
    return tags


@dataclass
class PaperRecord:
    """Serializable metadata for one selectable paper."""

    paper_id: str
    source_path: str
    filename: str
    source_kind: PaperSourceKind
    checksum_sha256: str
    title: str
    authors: str | None = None
    year: int | None = None
    topic: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    venue: str | None = None
    topic_tags: list[str] = field(default_factory=list)
    source_url: str | None = None
    pdf_url: str | None = None
    license: str | None = None
    active: bool = False
    indexed_status: PaperIndexStatus = "available"
    chunk_count: int | None = None
    parse_error: str | None = None
    index_error: str | None = None
    updated_at: str = ""


@dataclass(frozen=True)
class ChunkRecord:
    """Serializable metadata for one indexed text chunk."""

    chunk_id: str
    source_path: str
    source: str
    page: int | None
    chunk_index: int
    content_sha256: str
    char_count: int
    preview: str
    updated_at: str


@dataclass(frozen=True)
class CorpusProfile:
    """Named local corpus selection profile.

    This is a no-key bridge toward real workspace/corpus separation. It stores
    selectable source paths only, not paper contents or per-user ownership.
    """

    profile_id: str
    name: str
    source_paths: list[str]
    description: str | None = None
    paper_count: int = 0
    created_at: str = ""
    updated_at: str = ""


class CorpusMetadataStore:
    """JSON-backed metadata store for the current local corpus."""

    def __init__(self, path: Path | None = None, db_path: Path | None = None):
        self.path = path or CORPUS_METADATA_FILE
        self.db_path = db_path or (CORPUS_METADATA_DB_FILE if path is None else self.path.with_suffix(".sqlite3"))

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"version": 1, "papers": {}}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def save(self, payload: dict[str, Any]) -> None:
        atomic_write_json(self.path, payload)
        self._sync_sqlite(payload)

    def list_papers(self) -> list[PaperRecord]:
        payload = self.load()
        records = [PaperRecord(**item) for item in payload.get("papers", {}).values()]
        records.sort(key=lambda record: (not record.active, record.source_path.lower()))
        return records

    def storage_status(self) -> dict[str, Any]:
        """Return local corpus metadata storage state without exposing contents."""
        self._ensure_sqlite()
        sqlite_rows = 0
        if self.db_path.exists():
            with self._connect() as conn:
                row = conn.execute("SELECT COUNT(*) AS count FROM papers").fetchone()
                sqlite_rows = int(row["count"]) if row else 0
        return {
            "json_exists": self.path.exists(),
            "json_bytes": self.path.stat().st_size if self.path.exists() else 0,
            "sqlite_exists": self.db_path.exists(),
            "sqlite_bytes": self.db_path.stat().st_size if self.db_path.exists() else 0,
            "sqlite_rows": sqlite_rows,
        }

    def upsert_paper(
        self,
        path: Path,
        *,
        manifest_entry: dict[str, Any] | None = None,
        active: bool = False,
        indexed_status: PaperIndexStatus | None = None,
        chunk_count: int | None = None,
        parse_error: str | None = None,
        index_error: str | None = None,
    ) -> PaperRecord:
        payload = self.load()
        papers = payload.setdefault("papers", {})
        source_path = project_relative(path)
        current = papers.get(source_path, {})
        manifest_entry = manifest_entry or {}
        enrichment = current | manifest_entry
        source_kind: PaperSourceKind = "library" if PAPERS_LIBRARY_DIR in path.parents else "upload"
        if path.parent == PROJECT_ROOT / "papers":
            source_kind = "paper"
        current_status = current.get("indexed_status", "available")
        if indexed_status:
            status = indexed_status
        elif active:
            status = "active" if current_status == "available" else current_status
        else:
            status = "available" if current_status in {"active", "indexed"} else current_status
        record = PaperRecord(
            paper_id=current.get("paper_id", hashlib.sha256(source_path.encode()).hexdigest()[:16]),
            source_path=source_path,
            filename=path.name,
            source_kind=source_kind,
            checksum_sha256=file_sha256(path),
            title=manifest_entry.get("title") or current.get("title") or path.stem.replace("-", " "),
            authors=manifest_entry.get("authors") or current.get("authors"),
            year=manifest_entry.get("year") or current.get("year"),
            topic=manifest_entry.get("topic") or current.get("topic"),
            doi=_first_nonempty(manifest_entry.get("doi"), current.get("doi")),
            arxiv_id=_derive_arxiv_id(path, enrichment),
            venue=_first_nonempty(manifest_entry.get("venue"), current.get("venue")),
            topic_tags=_normalize_topic_tags(
                manifest_entry.get("topic_tags"),
                current.get("topic_tags"),
                manifest_entry.get("topic"),
                current.get("topic"),
            ),
            source_url=manifest_entry.get("source_url") or current.get("source_url"),
            pdf_url=manifest_entry.get("pdf_url") or current.get("pdf_url"),
            license=manifest_entry.get("license") or current.get("license"),
            active=active,
            indexed_status=status,
            chunk_count=chunk_count if chunk_count is not None else current.get("chunk_count"),
            parse_error=parse_error,
            index_error=index_error,
            updated_at=utc_now(),
        )
        papers[source_path] = asdict(record)
        self.save(payload)
        return record

    def refresh_from_files(
        self,
        paths: list[Path],
        *,
        active_paths: list[Path] | None = None,
        manifest: dict[str, dict] | None = None,
    ) -> list[PaperRecord]:
        active_set = {path.resolve() for path in active_paths or []}
        manifest = manifest or {}
        records = [
            self.upsert_paper(
                path,
                manifest_entry=manifest.get(path.name, {}),
                active=path.resolve() in active_set,
            )
            for path in paths
        ]
        current_source_paths = {project_relative(path) for path in paths}
        payload = self.load()
        papers = payload.setdefault("papers", {})
        stale_source_paths = set(papers) - current_source_paths
        if stale_source_paths:
            for source_path in stale_source_paths:
                papers.pop(source_path, None)
            self.save(payload)
        return sorted(records, key=lambda record: (not record.active, record.source_path.lower()))

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_sqlite(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS papers (
                    source_path TEXT PRIMARY KEY,
                    paper_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    source_kind TEXT NOT NULL,
                    checksum_sha256 TEXT NOT NULL,
                    title TEXT NOT NULL,
                    authors TEXT,
                    year INTEGER,
                    topic TEXT,
                    doi TEXT,
                    arxiv_id TEXT,
                    venue TEXT,
                    topic_tags TEXT,
                    source_url TEXT,
                    pdf_url TEXT,
                    license TEXT,
                    active INTEGER NOT NULL,
                    indexed_status TEXT NOT NULL,
                    chunk_count INTEGER,
                    parse_error TEXT,
                    index_error TEXT,
                    updated_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_papers_active ON papers(active)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_papers_indexed_status ON papers(indexed_status)")
            existing = {row["name"] for row in conn.execute("PRAGMA table_info(papers)").fetchall()}
            for column_name, column_type in {
                "doi": "TEXT",
                "arxiv_id": "TEXT",
                "venue": "TEXT",
                "topic_tags": "TEXT",
            }.items():
                if column_name not in existing:
                    conn.execute(f"ALTER TABLE papers ADD COLUMN {column_name} {column_type}")

    def _sync_sqlite(self, payload: dict[str, Any]) -> None:
        self._ensure_sqlite()
        papers = payload.get("papers", {})
        with self._connect() as conn:
            seen = set()
            for source_path, item in papers.items():
                seen.add(source_path)
                conn.execute(
                    """
                    INSERT INTO papers (
                        source_path, paper_id, filename, source_kind, checksum_sha256,
                        title, authors, year, topic, doi, arxiv_id, venue, topic_tags,
                        source_url, pdf_url, license,
                        active, indexed_status, chunk_count, parse_error, index_error,
                        updated_at, payload
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(source_path) DO UPDATE SET
                        paper_id=excluded.paper_id,
                        filename=excluded.filename,
                        source_kind=excluded.source_kind,
                        checksum_sha256=excluded.checksum_sha256,
                        title=excluded.title,
                        authors=excluded.authors,
                        year=excluded.year,
                        topic=excluded.topic,
                        doi=excluded.doi,
                        arxiv_id=excluded.arxiv_id,
                        venue=excluded.venue,
                        topic_tags=excluded.topic_tags,
                        source_url=excluded.source_url,
                        pdf_url=excluded.pdf_url,
                        license=excluded.license,
                        active=excluded.active,
                        indexed_status=excluded.indexed_status,
                        chunk_count=excluded.chunk_count,
                        parse_error=excluded.parse_error,
                        index_error=excluded.index_error,
                        updated_at=excluded.updated_at,
                        payload=excluded.payload
                    """,
                    (
                        source_path,
                        item["paper_id"],
                        item["filename"],
                        item["source_kind"],
                        item["checksum_sha256"],
                        item["title"],
                        item.get("authors"),
                        item.get("year"),
                        item.get("topic"),
                        item.get("doi"),
                        item.get("arxiv_id"),
                        item.get("venue"),
                        json.dumps(item.get("topic_tags", []), ensure_ascii=False),
                        item.get("source_url"),
                        item.get("pdf_url"),
                        item.get("license"),
                        1 if item.get("active") else 0,
                        item["indexed_status"],
                        item.get("chunk_count"),
                        item.get("parse_error"),
                        item.get("index_error"),
                        item.get("updated_at", ""),
                        json.dumps(item, ensure_ascii=False),
                    ),
                )
            if seen:
                placeholders = ",".join("?" for _ in seen)
                conn.execute(
                    f"DELETE FROM papers WHERE source_path NOT IN ({placeholders})",
                    tuple(sorted(seen)),
                )
            else:
                conn.execute("DELETE FROM papers")


def normalize_profile_id(value: str) -> str:
    """Return a stable local profile ID from user-facing text."""
    normalized = re.sub(r"[^a-z0-9_-]+", "-", value.strip().casefold())
    normalized = re.sub(r"-+", "-", normalized).strip("-_")
    if not normalized:
        raise ValueError("Corpus profile ID cannot be empty")
    return normalized[:80]


def safe_corpus_profile_report_filename(profile_id: str) -> str:
    """Return a header-safe no-secret filename for corpus profile reports."""
    try:
        safe_id = normalize_profile_id(profile_id)
    except ValueError:
        safe_id = ""
    if not safe_id or _SENSITIVE_PROFILE_FILENAME_RE.search(safe_id):
        safe_id = hashlib.sha256(profile_id.encode()).hexdigest()[:16]
    return f"fluxmind-corpus-profile-{safe_id}.md"


class CorpusProfileStore:
    """JSON-backed store for reusable local corpus selections."""

    def __init__(self, path: Path | None = None):
        self.path = path or CORPUS_PROFILES_FILE

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"version": 1, "profiles": {}}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def save(self, payload: dict[str, Any]) -> None:
        atomic_write_json(self.path, payload)

    def list_profiles(self) -> list[CorpusProfile]:
        payload = self.load()
        profiles = [
            CorpusProfile(**item)
            for item in payload.get("profiles", {}).values()
        ]
        profiles.sort(key=lambda profile: (profile.name.casefold(), profile.profile_id))
        return profiles

    def get_profile(self, profile_id: str) -> CorpusProfile:
        payload = self.load()
        normalized_id = normalize_profile_id(profile_id)
        item = payload.get("profiles", {}).get(normalized_id)
        if not item:
            raise KeyError(normalized_id)
        return CorpusProfile(**item)

    def upsert_profile(
        self,
        *,
        name: str,
        source_paths: list[str],
        profile_id: str | None = None,
        description: str | None = None,
    ) -> CorpusProfile:
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("Corpus profile name cannot be empty")
        clean_paths: list[str] = []
        seen: set[str] = set()
        for source_path in source_paths:
            clean_path = source_path.strip()
            if not clean_path or clean_path in seen:
                continue
            seen.add(clean_path)
            clean_paths.append(clean_path)
        if not clean_paths:
            raise ValueError("Corpus profile requires at least one source path")

        payload = self.load()
        profiles = payload.setdefault("profiles", {})
        normalized_id = normalize_profile_id(profile_id or clean_name)
        now = utc_now()
        existing = profiles.get(normalized_id, {})
        profile = CorpusProfile(
            profile_id=normalized_id,
            name=clean_name,
            source_paths=clean_paths,
            description=description.strip() if description and description.strip() else None,
            paper_count=len(clean_paths),
            created_at=existing.get("created_at") or now,
            updated_at=now,
        )
        profiles[normalized_id] = asdict(profile)
        self.save(payload)
        return profile

    def storage_status(self) -> dict[str, Any]:
        return {
            "json_exists": self.path.exists(),
            "json_bytes": self.path.stat().st_size if self.path.exists() else 0,
            "profiles": len(self.list_profiles()),
        }


class ChunkMetadataStore:
    """SQLite-backed current-state index for local RAG chunks."""

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or CHUNK_METADATA_DB_FILE

    def replace_for_sources(self, chunks: list[Any], *, source_paths: list[str]) -> list[ChunkRecord]:
        """Replace chunk metadata for a set of source paths."""
        source_set = set(source_paths)
        records = self._records_from_chunks(chunks)
        with self._connect() as conn:
            self._ensure_sqlite(conn)
            if source_set:
                placeholders = ",".join("?" for _ in source_set)
                conn.execute(
                    f"DELETE FROM chunks WHERE source_path IN ({placeholders})",
                    tuple(sorted(source_set)),
                )
            for record in records:
                conn.execute(
                    """
                    INSERT INTO chunks (
                        chunk_id, source_path, source, page, chunk_index,
                        content_sha256, char_count, preview, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(chunk_id) DO UPDATE SET
                        source_path=excluded.source_path,
                        source=excluded.source,
                        page=excluded.page,
                        chunk_index=excluded.chunk_index,
                        content_sha256=excluded.content_sha256,
                        char_count=excluded.char_count,
                        preview=excluded.preview,
                        updated_at=excluded.updated_at
                    """,
                    (
                        record.chunk_id,
                        record.source_path,
                        record.source,
                        record.page,
                        record.chunk_index,
                        record.content_sha256,
                        record.char_count,
                        record.preview,
                        record.updated_at,
                    ),
                )
        return records

    def list_chunks(
        self,
        *,
        source_path: str | None = None,
        page: int | None = None,
        q: str | None = None,
        limit: int = 100,
    ) -> list[ChunkRecord]:
        self._ensure_sqlite()
        clauses: list[str] = []
        params: list[Any] = []
        if source_path:
            clauses.append("source_path = ?")
            params.append(source_path)
        if page is not None:
            clauses.append("page = ?")
            params.append(page)
        query = (q or "").strip()
        if query:
            clauses.append(
                "("
                "chunk_id LIKE ? OR "
                "source_path LIKE ? OR "
                "source LIKE ? OR "
                "content_sha256 LIKE ? OR "
                "preview LIKE ?"
                ")"
            )
            pattern = f"%{query}%"
            params.extend([pattern, pattern, pattern, pattern, pattern])
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        order_by = "chunk_index ASC" if source_path else "updated_at DESC, source_path ASC, chunk_index ASC"
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT *
                FROM chunks
                {where}
                ORDER BY {order_by}
                LIMIT ?
                """,
                (*params, limit),
            ).fetchall()
        return [ChunkRecord(**dict(row)) for row in rows]

    def storage_status(self) -> dict[str, Any]:
        self._ensure_sqlite()
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS count FROM chunks").fetchone()
            source_row = conn.execute("SELECT COUNT(DISTINCT source_path) AS count FROM chunks").fetchone()
        return {
            "sqlite_exists": self.db_path.exists(),
            "sqlite_bytes": self.db_path.stat().st_size if self.db_path.exists() else 0,
            "sqlite_rows": int(row["count"]) if row else 0,
            "source_paths": int(source_row["count"]) if source_row else 0,
        }

    def source_paths(self) -> list[str]:
        """Return source paths represented by current chunk metadata."""
        self._ensure_sqlite()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT source_path FROM chunks ORDER BY source_path"
            ).fetchall()
        return [str(row["source_path"]) for row in rows]

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_sqlite(self, conn: sqlite3.Connection | None = None) -> None:
        if conn is None:
            with self._connect() as owned_conn:
                self._ensure_sqlite(owned_conn)
            return
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                source_path TEXT NOT NULL,
                source TEXT NOT NULL,
                page INTEGER,
                chunk_index INTEGER NOT NULL,
                content_sha256 TEXT NOT NULL,
                char_count INTEGER NOT NULL,
                preview TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_source_path ON chunks(source_path)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_page ON chunks(source_path, page)")

    @staticmethod
    def _records_from_chunks(chunks: list[Any]) -> list[ChunkRecord]:
        now = utc_now()
        per_source_counts: dict[str, int] = {}
        records: list[ChunkRecord] = []
        for chunk in chunks:
            metadata = getattr(chunk, "metadata", {}) or {}
            page_content = getattr(chunk, "page_content", "") or ""
            source_path = str(metadata.get("source_path") or metadata.get("source") or "unknown")
            source = str(metadata.get("source") or Path(source_path).name)
            page = metadata.get("page")
            page_number = int(page) if isinstance(page, int) or str(page).isdigit() else None
            chunk_index = per_source_counts.get(source_path, 0)
            per_source_counts[source_path] = chunk_index + 1
            digest = hashlib.sha256(
                f"{source_path}\n{page_number}\n{chunk_index}\n{page_content}".encode()
            ).hexdigest()
            records.append(
                ChunkRecord(
                    chunk_id=digest[:24],
                    source_path=source_path,
                    source=source,
                    page=page_number,
                    chunk_index=chunk_index,
                    content_sha256=hashlib.sha256(page_content.encode()).hexdigest(),
                    char_count=len(page_content),
                    preview=" ".join(page_content.split())[:240],
                    updated_at=now,
                )
            )
        return records
