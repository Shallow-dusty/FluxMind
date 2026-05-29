"""Local corpus metadata registry.

This is the first storage boundary for papers and indexing state. It is a
JSON-backed development store, not the final multi-user database.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from src.config import CORPUS_METADATA_FILE, PAPERS_LIBRARY_DIR, PROJECT_ROOT


PaperSourceKind = Literal["library", "upload", "paper"]
PaperIndexStatus = Literal["available", "active", "indexed", "failed"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def project_relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    source_url: str | None = None
    pdf_url: str | None = None
    license: str | None = None
    active: bool = False
    indexed_status: PaperIndexStatus = "available"
    chunk_count: int | None = None
    parse_error: str | None = None
    index_error: str | None = None
    updated_at: str = ""


class CorpusMetadataStore:
    """JSON-backed metadata store for the current local corpus."""

    def __init__(self, path: Path | None = None):
        self.path = path or CORPUS_METADATA_FILE

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"version": 1, "papers": {}}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def save(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def list_papers(self) -> list[PaperRecord]:
        payload = self.load()
        records = [PaperRecord(**item) for item in payload.get("papers", {}).values()]
        records.sort(key=lambda record: (not record.active, record.source_path.lower()))
        return records

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
        return sorted(records, key=lambda record: (not record.active, record.source_path.lower()))
