"""PDF ingestion pipeline: load -> split -> embed -> store in FAISS."""

import hashlib
import json
import re
import shutil
import threading
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.config import (
    ACTIVE_PAPERS_FILE,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    FAISS_INDEX_DIR,
    MAX_UPLOAD_SIZE_MB,
    PAPER_LIBRARY_MANIFEST,
    PAPERS_DIR,
    PAPERS_LIBRARY_DIR,
    PAPERS_UPLOADS_DIR,
    PROJECT_ROOT,
    UPLOAD_SCAN_BLOCK_ACTIVE_CONTENT,
    UPLOAD_SCAN_ENABLED,
    UPLOAD_SCAN_MAX_PAGES,
    UPLOAD_SCAN_REJECT_ENCRYPTED,
)
from src.embeddings import get_embedding_model
from src.metadata import ChunkMetadataStore, CorpusMetadataStore, PaperRecord, file_sha256
from src.runtime import append_runtime_event, logger


_FS_RESERVED_RE = re.compile(r'[\x00-\x1f<>:"/\\|?*]+')
_DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)
_ARXIV_RE = re.compile(
    r"(?:arxiv[:\s-]+|arxiv\.org/(?:abs|pdf)/)(\d{4}\.\d{4,5})(?:v\d+)?",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_KEYWORD_LINE_RE = re.compile(
    r"^(?:key\s*words?|index\s+terms?)\s*[:\-\u2013\u2014]\s*(.+)$",
    re.IGNORECASE,
)
_TABLE_CAPTION_RE = re.compile(r"^Table\s+\d+[.:]?\s*", re.IGNORECASE)
_FIGURE_CAPTION_RE = re.compile(r"^(?:Fig(?:ure)?\.?)\s+\d+[.:]?\s*", re.IGNORECASE)
_ALGORITHM_CAPTION_RE = re.compile(r"^Algorithm\s+\d+[.:]?\s*", re.IGNORECASE)
_EQUATION_HINT_RE = re.compile(
    r"(=|≤|≥|≠|≈|∈|∀|∑|√|α|β|γ|η|λ|ψ|ω|θ|∆|Δ|˙|\bdt\b|\bsgn\b|\btan[−-]?1\b)",
    re.IGNORECASE,
)
_EQUATION_CONTEXT_RE = re.compile(
    r"\b(equation|voltage|flux|sliding|surface|state[- ]space|control law)\b",
    re.IGNORECASE,
)
_AUTHOR_STOP_PREFIXES = (
    "abstract",
    "keywords",
    "key words",
    "index terms",
    "introduction",
    "i. introduction",
    "1 introduction",
)
_AFFILIATION_MARKERS = (
    "@",
    "university",
    "college",
    "department",
    "school",
    "institute",
    "laboratory",
    "lab ",
    "centre",
    "center",
    "faculty",
    "academy",
    "corporation",
    "company",
    "email",
)
_PDF_MAGIC_RE = re.compile(rb"%PDF-\d\.\d")
_PDF_ACTIVE_CONTENT_MARKERS: dict[str, tuple[bytes, ...]] = {
    "javascript": (b"JavaScript", b"JS"),
    "open_action": (b"OpenAction",),
    "additional_action": (b"AA",),
    "launch_action": (b"Launch",),
    "embedded_file": (b"EmbeddedFile", b"EmbeddedFiles", b"Filespec"),
    "rich_media": (b"RichMedia",),
    "xfa_form": (b"XFA",),
}


class IngestionCancelled(RuntimeError):
    """Raised when an index rebuild is cancelled before committing new state."""


@dataclass(frozen=True)
class PdfStructureMarker:
    """Best-effort layout marker extracted from a PDF page."""

    kind: str
    source: str
    source_path: str
    page: int
    text: str
    rule: str


@dataclass(frozen=True)
class UploadScanResult:
    """Metadata-only pre-write PDF upload scan result."""

    allowed: bool
    status: str
    reason_codes: tuple[str, ...]
    byte_count: int
    page_count: int = 0
    encrypted: bool = False
    active_content_markers: tuple[str, ...] = ()
    scan_enabled: bool = True

    def to_metadata(self) -> dict[str, Any]:
        return {
            "scan_enabled": self.scan_enabled,
            "status": self.status,
            "reason_codes": list(self.reason_codes),
            "byte_count": self.byte_count,
            "page_count": self.page_count,
            "encrypted": self.encrypted,
            "active_content_markers": list(self.active_content_markers),
            "active_content_marker_count": len(self.active_content_markers),
            "max_pages": UPLOAD_SCAN_MAX_PAGES,
            "reject_encrypted": UPLOAD_SCAN_REJECT_ENCRYPTED,
            "block_active_content": UPLOAD_SCAN_BLOCK_ACTIVE_CONTENT,
        }


def _raise_if_cancelled(cancel_event: threading.Event | None) -> None:
    if cancel_event and cancel_event.is_set():
        raise IngestionCancelled("Index rebuild was cancelled.")


def _safe_pdf_name(filename: str) -> str:
    """Return a safe PDF filename. Preserves Unicode (CJK, Cyrillic, etc.);
    strips control chars and filesystem-reserved characters; collapses
    whitespace to single hyphens; caps stem at 120 chars."""
    name = Path(filename).name.strip()
    if not name.lower().endswith(".pdf"):
        raise ValueError("Only PDF files are supported.")
    stem = unicodedata.normalize("NFC", Path(name).stem)
    stem = _FS_RESERVED_RE.sub("-", stem)
    stem = re.sub(r"\s+", "-", stem)
    stem = re.sub(r"-+", "-", stem).strip(".-_ ")
    if not stem:
        stem = "paper"
    return f"{stem[:120]}.pdf"


def _resolve_unique_path(target: Path) -> Path:
    """If target exists, append -1, -2, ... before the suffix until unique."""
    if not target.exists():
        return target
    stem, suffix, parent = target.stem, target.suffix, target.parent
    for i in range(1, 1000):
        candidate = parent / f"{stem}-{i}{suffix}"
        if not candidate.exists():
            return candidate
    raise ValueError("Too many uploads with conflicting names.")


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _has_pdf_magic(content: bytes) -> bool:
    return bool(_PDF_MAGIC_RE.search(content[:1024]))


def _pdf_name_object_pattern(name: bytes) -> bytes:
    return rb"/" + re.escape(name) + rb"(?=[\s/<>\[\]\(\)]|$)"


def _active_content_markers(content: bytes) -> tuple[str, ...]:
    markers: list[str] = []
    for code, names in _PDF_ACTIVE_CONTENT_MARKERS.items():
        for name in names:
            if re.search(_pdf_name_object_pattern(name), content, flags=re.IGNORECASE):
                markers.append(code)
                break
    return tuple(markers)


def scan_uploaded_pdf(pdf_bytes: bytes) -> UploadScanResult:
    """Run a local no-secret PDF upload scan before writing upload bytes."""
    byte_count = len(pdf_bytes)
    if not UPLOAD_SCAN_ENABLED:
        return UploadScanResult(
            allowed=True,
            status="skipped",
            reason_codes=("scan_disabled",),
            byte_count=byte_count,
            scan_enabled=False,
        )

    reason_codes: list[str] = []
    active_markers: tuple[str, ...] = ()
    page_count = 0
    encrypted = False

    if not _has_pdf_magic(pdf_bytes):
        reason_codes.append("invalid_pdf_magic")
    else:
        if UPLOAD_SCAN_BLOCK_ACTIVE_CONTENT:
            active_markers = _active_content_markers(pdf_bytes)
            reason_codes.extend(f"active_content_{marker}" for marker in active_markers)
        if not reason_codes:
            try:
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            except Exception:
                reason_codes.append("pdf_parse_failed")
            else:
                try:
                    encrypted = bool(
                        getattr(doc, "is_encrypted", False)
                        or getattr(doc, "needs_pass", False)
                    )
                    if encrypted and UPLOAD_SCAN_REJECT_ENCRYPTED:
                        reason_codes.append("encrypted_pdf")
                    else:
                        page_count = len(doc)
                        if page_count < 1:
                            reason_codes.append("pdf_empty")
                        elif UPLOAD_SCAN_MAX_PAGES > 0 and page_count > UPLOAD_SCAN_MAX_PAGES:
                            reason_codes.append("pdf_page_limit_exceeded")
                finally:
                    doc.close()

    allowed = not reason_codes
    return UploadScanResult(
        allowed=allowed,
        status="allowed" if allowed else "blocked",
        reason_codes=tuple(reason_codes),
        byte_count=byte_count,
        page_count=page_count,
        encrypted=encrypted,
        active_content_markers=active_markers,
    )


def _record_upload_scan_event(result: UploadScanResult) -> None:
    if not result.scan_enabled:
        return
    try:
        append_runtime_event(
            kind="upload_scan",
            code="upload_scan_allowed" if result.allowed else "upload_scan_blocked",
            message="Metadata-only PDF upload scan event.",
            metadata=result.to_metadata(),
        )
    except OSError:
        logger.warning("upload_scan.event_log_failed status=%s", result.status)


def _upload_scan_error(result: UploadScanResult) -> ValueError:
    reasons = ", ".join(result.reason_codes) or "blocked"
    return ValueError(f"Upload scan blocked PDF: {reasons}.")


def _find_existing_pdf_by_checksum(checksum_sha256: str) -> Path | None:
    """Return an existing selectable PDF with this checksum, if any."""
    for path in discover_pdfs():
        try:
            if file_sha256(path) == checksum_sha256:
                return path
        except OSError:
            continue
    return None


def _clear_directory_contents(directory: Path, *, exclude: set[Path] | None = None) -> None:
    """Clear a runtime directory without requiring write access to its parent."""
    excluded = {path.resolve() for path in exclude or set()}
    for item in directory.iterdir():
        if item.resolve() in excluded:
            continue
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def _save_rebuilt_vector_store(store: FAISS, index_path: Path) -> None:
    """Save a rebuilt FAISS store without clearing the live index first."""
    index_path.mkdir(parents=True, exist_ok=True)
    temp_path = index_path / ".rebuild_tmp"
    if temp_path.exists():
        shutil.rmtree(temp_path)
    try:
        store.save_local(str(temp_path))
        _clear_directory_contents(index_path, exclude={temp_path})
        for item in temp_path.iterdir():
            item.replace(index_path / item.name)
        temp_path.rmdir()
    except Exception:
        if temp_path.exists():
            shutil.rmtree(temp_path)
        raise


def _relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def _safe_source_path(path: Path) -> str:
    try:
        return _relative(path)
    except ValueError:
        return path.resolve().as_posix()


def _clean_metadata_value(value: Any) -> str | None:
    if value is None:
        return None
    cleaned = re.sub(r"\s+", " ", str(value)).strip()
    if not cleaned or cleaned.lower() in {"none", "null", "untitled"}:
        return None
    return cleaned


def _metadata_topic_tags(*values: Any) -> list[str]:
    tags: list[str] = []
    seen: set[str] = set()
    for value in values:
        if isinstance(value, list):
            candidates = value
        else:
            cleaned = _clean_metadata_value(value)
            if not cleaned:
                continue
            candidates = re.split(r"[,;/|]+", cleaned)
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


def _candidate_title_from_first_page(text: str) -> str | None:
    for raw_line in text.splitlines()[:12]:
        line = _clean_metadata_value(raw_line)
        if not line:
            continue
        if len(line) < 8 or len(line) > 220:
            continue
        if _DOI_RE.search(line) or line.lower().startswith(("abstract", "keywords", "citation")):
            continue
        return line
    return None


def _first_page_lines(text: str, *, limit: int = 30) -> list[str]:
    lines: list[str] = []
    for raw_line in text.splitlines()[:limit]:
        line = _clean_metadata_value(raw_line)
        if line:
            lines.append(line)
    return lines


def _line_key(value: str) -> str:
    return re.sub(r"\W+", "", value).casefold()


def _looks_like_author_line(line: str) -> bool:
    lowered = line.casefold()
    if any(lowered.startswith(prefix) for prefix in _AUTHOR_STOP_PREFIXES):
        return False
    if any(marker in lowered for marker in _AFFILIATION_MARKERS):
        return False
    if _DOI_RE.search(line) or _ARXIV_RE.search(line):
        return False
    if len(line) < 4 or len(line) > 180:
        return False
    if sum(ch.isdigit() for ch in line) > 4:
        return False

    separators = bool(re.search(r"\b(and|et al\.?)\b|[,;]", line, re.IGNORECASE))
    name_like_tokens = re.findall(r"\b[A-Z][A-Za-z'.-]{1,}\b", line)
    initial_tokens = re.findall(r"\b[A-Z]\.\b", line)
    return separators or len(name_like_tokens) + len(initial_tokens) >= 2


def _candidate_authors_from_first_page(text: str, title: str | None = None) -> str | None:
    lines = _first_page_lines(text)
    if not lines:
        return None

    start_index = 1
    if title:
        title_key = _line_key(title)
        for index, line in enumerate(lines[:12]):
            if _line_key(line) == title_key:
                start_index = index + 1
                break

    authors: list[str] = []
    for line in lines[start_index:start_index + 8]:
        lowered = line.casefold()
        if any(lowered.startswith(prefix) for prefix in _AUTHOR_STOP_PREFIXES):
            break
        if not _looks_like_author_line(line):
            if authors:
                break
            continue
        authors.append(line)
        if len(authors) >= 3:
            break
    return "; ".join(authors) if authors else None


def _candidate_topic_tags_from_first_page(text: str) -> list[str]:
    for line in _first_page_lines(text):
        match = _KEYWORD_LINE_RE.match(line)
        if match:
            return _metadata_topic_tags(match.group(1))
    return []


def load_library_manifest() -> dict[str, dict]:
    """Load curated paper metadata keyed by filename."""
    if not PAPER_LIBRARY_MANIFEST.exists():
        return {}
    return json.loads(PAPER_LIBRARY_MANIFEST.read_text(encoding="utf-8"))


def extract_pdf_bibliographic_metadata(path: Path) -> dict[str, Any]:
    """Best-effort no-key metadata extraction for uploaded/unmanifested PDFs."""
    try:
        doc = fitz.open(str(path))
    except Exception:
        return {}
    try:
        raw_metadata = doc.metadata or {}
        first_page_text = doc[0].get_text()[:5000] if len(doc) else ""
    except Exception:
        first_page_text = ""
        raw_metadata = {}
    finally:
        doc.close()

    title = _clean_metadata_value(raw_metadata.get("title"))
    if not title or title.casefold() == path.stem.casefold():
        title = _candidate_title_from_first_page(first_page_text)
    authors = _clean_metadata_value(raw_metadata.get("author"))
    if not authors:
        authors = _candidate_authors_from_first_page(first_page_text, title)
    subject = _clean_metadata_value(raw_metadata.get("subject"))
    keywords = _clean_metadata_value(raw_metadata.get("keywords"))
    doi_match = _DOI_RE.search(first_page_text)
    arxiv_match = _ARXIV_RE.search(f"{path.name}\n{first_page_text}")
    date_text = " ".join(
        str(raw_metadata.get(key, ""))
        for key in ("creationDate", "modDate")
    )
    year_match = _YEAR_RE.search(date_text) or _YEAR_RE.search(first_page_text[:2000])

    metadata: dict[str, Any] = {}
    if title:
        metadata["title"] = title
    if authors:
        metadata["authors"] = authors
    if year_match:
        metadata["year"] = int(year_match.group(0))
    if doi_match:
        metadata["doi"] = doi_match.group(0).rstrip(".,;)")
    if arxiv_match:
        metadata["arxiv_id"] = arxiv_match.group(1)
    tags = _metadata_topic_tags(keywords, subject, _candidate_topic_tags_from_first_page(first_page_text))
    if tags:
        metadata["topic_tags"] = tags
        metadata["topic"] = tags[0]
    return metadata


def _structure_marker_for_line(line: str) -> tuple[str, str] | None:
    if _TABLE_CAPTION_RE.search(line):
        return "table", "table_caption"
    if _FIGURE_CAPTION_RE.search(line):
        return "figure", "figure_caption"
    if _ALGORITHM_CAPTION_RE.search(line):
        return "algorithm", "algorithm_caption"
    if _EQUATION_HINT_RE.search(line):
        if len(line) <= 220 or _EQUATION_CONTEXT_RE.search(line):
            return "equation", "math_symbol"
    return None


def extract_pdf_structure_markers(
    path: Path,
    *,
    kinds: set[str] | None = None,
    page: int | None = None,
    max_pages: int | None = None,
    max_markers: int = 100,
) -> list[PdfStructureMarker]:
    """Extract no-key PDF layout markers for acceptance checks and APIs."""
    markers: list[PdfStructureMarker] = []
    allowed = {kind.casefold() for kind in kinds} if kinds else None
    source_path = _safe_source_path(path)
    doc = fitz.open(str(path))
    try:
        page_indexes: list[int]
        if page is not None:
            if page < 1 or page > len(doc):
                return []
            page_indexes = [page - 1]
        else:
            upper = len(doc) if max_pages is None else min(len(doc), max_pages)
            page_indexes = list(range(upper))
        for page_index in page_indexes:
            for raw_line in doc[page_index].get_text().splitlines():
                line = _clean_metadata_value(raw_line)
                if not line or len(line) < 3:
                    continue
                classified = _structure_marker_for_line(line)
                if classified is None:
                    continue
                kind, rule = classified
                if allowed and kind.casefold() not in allowed:
                    continue
                markers.append(
                    PdfStructureMarker(
                        kind=kind,
                        source=path.name,
                        source_path=source_path,
                        page=page_index + 1,
                        text=line[:500],
                        rule=rule,
                    )
                )
                if len(markers) >= max_markers:
                    return markers
    finally:
        doc.close()
    return markers


def paper_metadata_entries(paths: list[Path], manifest: dict[str, dict] | None = None) -> dict[str, dict]:
    """Return manifest-over-extracted metadata keyed by selectable filename."""
    manifest = manifest or {}
    return {
        path.name: extract_pdf_bibliographic_metadata(path) | manifest.get(path.name, {})
        for path in paths
    }


def refresh_paper_metadata() -> list[PaperRecord]:
    """Refresh local paper metadata from selectable files and active selection."""
    store = CorpusMetadataStore()
    paths = discover_pdfs()
    records = store.refresh_from_files(
        paths,
        active_paths=load_active_paper_paths(),
        manifest=paper_metadata_entries(paths, load_library_manifest()),
    )
    if (FAISS_INDEX_DIR / "index.faiss").exists():
        manifest = paper_metadata_entries(paths, load_library_manifest())
        for record in records:
            if record.active:
                path = PROJECT_ROOT / record.source_path
                record = store.upsert_paper(
                    path,
                    manifest_entry=manifest.get(path.name, {}),
                    active=True,
                    indexed_status="indexed",
                )
    return store.list_papers()


def discover_pdfs() -> list[Path]:
    """Return all selectable PDFs from the bundled library and uploads."""
    paths: list[Path] = []
    for directory in (PAPERS_LIBRARY_DIR, PAPERS_UPLOADS_DIR, PAPERS_DIR):
        if not directory.exists():
            continue
        paths.extend(p for p in directory.glob("*.pdf") if p.is_file())
    return sorted(set(paths), key=lambda p: _relative(p).lower())


def load_active_paper_paths() -> list[Path]:
    """Load the current manual paper selection, defaulting to bundled papers."""
    all_papers = discover_pdfs()
    if ACTIVE_PAPERS_FILE.exists():
        selected = json.loads(ACTIVE_PAPERS_FILE.read_text(encoding="utf-8"))
        paths = [PROJECT_ROOT / item for item in selected]
        return [p for p in paths if p.exists() and p.suffix.lower() == ".pdf"]
    return [p for p in all_papers if PAPERS_LIBRARY_DIR in p.parents]


def save_active_paper_paths(paths: list[Path]) -> None:
    """Persist the manual paper selection beside the FAISS index."""
    FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    payload = [_relative(p) for p in paths if p.exists() and p.suffix.lower() == ".pdf"]
    ACTIVE_PAPERS_FILE.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    CorpusMetadataStore().refresh_from_files(
        discover_pdfs(),
        active_paths=paths,
        manifest=paper_metadata_entries(discover_pdfs(), load_library_manifest()),
    )


def resolve_selectable_source_paths(source_paths: list[str]) -> list[Path]:
    """Resolve project-relative source paths and require selectable PDFs."""
    selectable = {path.resolve().relative_to(PROJECT_ROOT).as_posix(): path for path in discover_pdfs()}
    paths: list[Path] = []
    seen: set[str] = set()
    for source_path in source_paths:
        key = source_path.strip()
        if not key:
            continue
        if key not in selectable:
            raise ValueError(f"PDF path is not in the selectable corpus: {source_path}")
        if key in seen:
            continue
        seen.add(key)
        paths.append(selectable[key])
    return paths


def set_active_paper_source_paths(source_paths: list[str]) -> list[PaperRecord]:
    """Persist the active corpus selection without rebuilding the FAISS index."""
    paths = resolve_selectable_source_paths(source_paths)
    save_active_paper_paths(paths)
    return refresh_paper_metadata()


def load_pdf(path: Path) -> list[Document]:
    """Extract text from a single PDF using PyMuPDF."""
    documents = []
    doc = fitz.open(str(path))
    try:
        for page_num, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": path.name,
                            "source_path": _relative(path),
                            "page": page_num + 1,
                        },
                    )
                )
    finally:
        doc.close()
    return documents


def load_all_pdfs(
    directory: Path | None = None,
    *,
    cancel_event: threading.Event | None = None,
) -> list[Document]:
    """Load all PDFs from a directory or from the current paper selection."""
    if directory is None:
        return load_pdfs(load_active_paper_paths(), cancel_event=cancel_event)

    directory.mkdir(parents=True, exist_ok=True)
    return load_pdfs(sorted(directory.glob("*.pdf")), cancel_event=cancel_event)


def load_pdfs(paths: list[Path], *, cancel_event: threading.Event | None = None) -> list[Document]:
    """Load PDF documents from explicit paths."""
    all_docs = []
    for pdf_path in paths:
        _raise_if_cancelled(cancel_event)
        all_docs.extend(load_pdf(pdf_path))
    _raise_if_cancelled(cancel_event)
    return all_docs


def split_documents(
    documents: list[Document],
    *,
    cancel_event: threading.Event | None = None,
) -> list[Document]:
    """Split documents into chunks for embedding."""
    _raise_if_cancelled(cancel_event)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    _raise_if_cancelled(cancel_event)
    return chunks


def build_vector_store(
    documents: list[Document] | None = None,
    *,
    rebuild: bool = False,
    cancel_event: threading.Event | None = None,
) -> FAISS:
    """Build or load FAISS vector store."""
    _raise_if_cancelled(cancel_event)
    embeddings = get_embedding_model()
    index_path = FAISS_INDEX_DIR

    # If index exists, load it
    if not rebuild and index_path.exists() and (index_path / "index.faiss").exists():
        store = FAISS.load_local(
            str(index_path), embeddings, allow_dangerous_deserialization=True
        )
        # If new documents provided, add them
        if documents:
            chunks = split_documents(documents, cancel_event=cancel_event)
            if chunks:
                _raise_if_cancelled(cancel_event)
                store.add_documents(chunks)
                _raise_if_cancelled(cancel_event)
                store.save_local(str(index_path))
        return store

    # Build from scratch
    if documents is None:
        documents = load_all_pdfs(cancel_event=cancel_event)

    if not documents:
        _raise_if_cancelled(cancel_event)
        # Create empty store with a placeholder
        store = FAISS.from_documents(
            [Document(page_content="FluxMind knowledge base initialized.", metadata={"source": "system"})],
            embeddings,
        )
        _raise_if_cancelled(cancel_event)
        if rebuild:
            _save_rebuilt_vector_store(store, index_path)
        else:
            store.save_local(str(index_path))
        return store

    chunks = split_documents(documents, cancel_event=cancel_event)
    _raise_if_cancelled(cancel_event)
    store = FAISS.from_documents(chunks, embeddings)
    _raise_if_cancelled(cancel_event)
    if rebuild:
        _save_rebuilt_vector_store(store, index_path)
    else:
        store.save_local(str(index_path))
    return store


def rebuild_vector_store_from_pdfs(
    paths: list[Path],
    *,
    cancel_event: threading.Event | None = None,
) -> tuple[FAISS, int]:
    """Rebuild the FAISS index from selected PDFs and persist the selection."""
    _raise_if_cancelled(cancel_event)
    docs = load_pdfs(paths, cancel_event=cancel_event)
    chunks = split_documents(docs, cancel_event=cancel_event) if docs else []
    _raise_if_cancelled(cancel_event)
    store = build_vector_store(docs, rebuild=True, cancel_event=cancel_event)
    _raise_if_cancelled(cancel_event)
    save_active_paper_paths(paths)
    chunk_count = len(chunks)
    chunk_counts_by_source: dict[str, int] = {}
    for chunk in chunks:
        source_path = chunk.metadata.get("source_path")
        if source_path:
            chunk_counts_by_source[source_path] = chunk_counts_by_source.get(source_path, 0) + 1
    ChunkMetadataStore().replace_for_sources(
        chunks,
        source_paths=[_relative(path) for path in paths],
    )
    _raise_if_cancelled(cancel_event)
    manifest = load_library_manifest()
    metadata_entries = paper_metadata_entries(paths, manifest)
    metadata_store = CorpusMetadataStore()
    for path in paths:
        source_path = _relative(path)
        metadata_store.upsert_paper(
            path,
            manifest_entry=metadata_entries.get(path.name, {}),
            active=True,
            indexed_status="indexed",
            chunk_count=chunk_counts_by_source.get(source_path, 0),
        )
    return store, chunk_count


def ingest_uploaded_pdf(pdf_bytes: bytes, filename: str) -> tuple[Path, int]:
    """Ingest one uploaded PDF into the vector store. Returns path and chunk count."""
    max_bytes = MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if len(pdf_bytes) > max_bytes:
        raise ValueError(f"PDF is larger than {MAX_UPLOAD_SIZE_MB} MB.")

    scan_result = scan_uploaded_pdf(pdf_bytes)
    _record_upload_scan_event(scan_result)
    if not scan_result.allowed:
        raise _upload_scan_error(scan_result)

    checksum_sha256 = _sha256_bytes(pdf_bytes)
    PAPERS_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = _find_existing_pdf_by_checksum(checksum_sha256)
    if pdf_path is None:
        pdf_path = _resolve_unique_path(PAPERS_UPLOADS_DIR / _safe_pdf_name(filename))
        pdf_path.write_bytes(pdf_bytes)
    else:
        metadata_store = CorpusMetadataStore()
        existing_record = next(
            (
                record
                for record in metadata_store.list_papers()
                if record.source_path == _relative(pdf_path)
            ),
            None,
        )
        if (
            existing_record
            and existing_record.indexed_status == "indexed"
            and (FAISS_INDEX_DIR / "index.faiss").exists()
        ):
            active = load_active_paper_paths()
            if pdf_path not in active:
                save_active_paper_paths(active + [pdf_path])
            metadata_store.upsert_paper(
                pdf_path,
                manifest_entry=extract_pdf_bibliographic_metadata(pdf_path),
                active=True,
                indexed_status="indexed",
                chunk_count=existing_record.chunk_count,
            )
            return pdf_path, existing_record.chunk_count or 0

    # Load and add to store
    docs = load_pdf(pdf_path)
    chunks = split_documents(docs)
    if not chunks:
        raise ValueError("PDF did not contain extractable text.")

    embeddings = get_embedding_model()
    FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)

    if (FAISS_INDEX_DIR / "index.faiss").exists():
        store = FAISS.load_local(
            str(FAISS_INDEX_DIR), embeddings, allow_dangerous_deserialization=True
        )
        store.add_documents(chunks)
    else:
        store = FAISS.from_documents(chunks, embeddings)

    store.save_local(str(FAISS_INDEX_DIR))

    active = load_active_paper_paths()
    if pdf_path not in active:
        save_active_paper_paths(active + [pdf_path])

    CorpusMetadataStore().upsert_paper(
        pdf_path,
        manifest_entry=extract_pdf_bibliographic_metadata(pdf_path),
        active=True,
        indexed_status="indexed",
        chunk_count=len(chunks),
    )
    ChunkMetadataStore().replace_for_sources(chunks, source_paths=[_relative(pdf_path)])

    return pdf_path, len(chunks)
