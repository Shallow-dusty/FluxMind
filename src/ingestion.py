"""PDF ingestion pipeline: load -> split -> embed -> store in FAISS."""

import json
import re
import shutil
import unicodedata
from pathlib import Path

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
)
from src.embeddings import get_embedding_model


_FS_RESERVED_RE = re.compile(r'[\x00-\x1f<>:"/\\|?*]+')


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


def _relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def load_library_manifest() -> dict[str, dict]:
    """Load curated paper metadata keyed by filename."""
    if not PAPER_LIBRARY_MANIFEST.exists():
        return {}
    return json.loads(PAPER_LIBRARY_MANIFEST.read_text(encoding="utf-8"))


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


def load_all_pdfs(directory: Path | None = None) -> list[Document]:
    """Load all PDFs from a directory or from the current paper selection."""
    if directory is None:
        return load_pdfs(load_active_paper_paths())

    directory.mkdir(parents=True, exist_ok=True)
    return load_pdfs(sorted(directory.glob("*.pdf")))


def load_pdfs(paths: list[Path]) -> list[Document]:
    """Load PDF documents from explicit paths."""
    all_docs = []
    for pdf_path in paths:
        all_docs.extend(load_pdf(pdf_path))
    return all_docs


def split_documents(documents: list[Document]) -> list[Document]:
    """Split documents into chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def build_vector_store(
    documents: list[Document] | None = None,
    *,
    rebuild: bool = False,
) -> FAISS:
    """Build or load FAISS vector store."""
    embeddings = get_embedding_model()
    index_path = FAISS_INDEX_DIR

    if rebuild and index_path.exists():
        shutil.rmtree(index_path)

    # If index exists, load it
    if index_path.exists() and (index_path / "index.faiss").exists():
        store = FAISS.load_local(
            str(index_path), embeddings, allow_dangerous_deserialization=True
        )
        # If new documents provided, add them
        if documents:
            chunks = split_documents(documents)
            if chunks:
                store.add_documents(chunks)
                store.save_local(str(index_path))
        return store

    # Build from scratch
    if documents is None:
        documents = load_all_pdfs()

    if not documents:
        # Create empty store with a placeholder
        store = FAISS.from_documents(
            [Document(page_content="FluxMind knowledge base initialized.", metadata={"source": "system"})],
            embeddings,
        )
        store.save_local(str(index_path))
        return store

    chunks = split_documents(documents)
    store = FAISS.from_documents(chunks, embeddings)
    store.save_local(str(index_path))
    return store


def rebuild_vector_store_from_pdfs(paths: list[Path]) -> tuple[FAISS, int]:
    """Rebuild the FAISS index from selected PDFs and persist the selection."""
    docs = load_pdfs(paths)
    store = build_vector_store(docs, rebuild=True)
    save_active_paper_paths(paths)
    return store, len(split_documents(docs)) if docs else 0


def ingest_uploaded_pdf(pdf_bytes: bytes, filename: str) -> tuple[Path, int]:
    """Ingest one uploaded PDF into the vector store. Returns path and chunk count."""
    max_bytes = MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if len(pdf_bytes) > max_bytes:
        raise ValueError(f"PDF is larger than {MAX_UPLOAD_SIZE_MB} MB.")

    # Save PDF to papers dir
    PAPERS_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = _resolve_unique_path(PAPERS_UPLOADS_DIR / _safe_pdf_name(filename))
    pdf_path.write_bytes(pdf_bytes)

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

    return pdf_path, len(chunks)
