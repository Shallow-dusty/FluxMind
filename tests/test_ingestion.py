import json
from pathlib import Path

import pytest
from langchain_core.documents import Document

from src import ingestion
from src.metadata import ChunkMetadataStore


def _minimal_pdf_bytes(text: str = "FluxMind upload scan fixture", *, pages: int = 1) -> bytes:
    import fitz

    document = fitz.open()
    for _ in range(pages):
        page = document.new_page()
        page.insert_text((72, 72), text)
    data = document.tobytes()
    document.close()
    return data


def test_safe_pdf_name_preserves_unicode_and_strips_reserved_chars():
    assert ingestion._safe_pdf_name(" 磁链 观测<>:\"|?*.PDF ") == "磁链-观测.pdf"


def test_safe_pdf_name_uses_basename_to_prevent_path_traversal():
    assert ingestion._safe_pdf_name("../../磁链 观测.pdf") == "磁链-观测.pdf"


def test_safe_pdf_name_rejects_non_pdf():
    with pytest.raises(ValueError, match="Only PDF"):
        ingestion._safe_pdf_name("notes.txt")


def test_scan_uploaded_pdf_allows_valid_pdf():
    result = ingestion.scan_uploaded_pdf(_minimal_pdf_bytes(pages=2))

    assert result.allowed is True
    assert result.status == "allowed"
    assert result.reason_codes == ()
    assert result.page_count == 2
    assert result.to_metadata()["active_content_marker_count"] == 0


def test_scan_uploaded_pdf_blocks_invalid_magic():
    result = ingestion.scan_uploaded_pdf(b"not a pdf")

    assert result.allowed is False
    assert result.status == "blocked"
    assert result.reason_codes == ("invalid_pdf_magic",)


def test_scan_uploaded_pdf_blocks_active_content_marker():
    result = ingestion.scan_uploaded_pdf(b"%PDF-1.7\n1 0 obj << /OpenAction 2 0 R /JavaScript 3 0 R >>\n%%EOF")

    assert result.allowed is False
    assert "active_content_javascript" in result.reason_codes
    assert "active_content_open_action" in result.reason_codes
    assert set(result.active_content_markers) == {"javascript", "open_action"}


def test_scan_uploaded_pdf_blocks_page_limit(monkeypatch):
    monkeypatch.setattr(ingestion, "UPLOAD_SCAN_MAX_PAGES", 1)

    result = ingestion.scan_uploaded_pdf(_minimal_pdf_bytes(pages=2))

    assert result.allowed is False
    assert result.reason_codes == ("pdf_page_limit_exceeded",)
    assert result.page_count == 2


def test_ingest_uploaded_pdf_rejects_failed_scan_before_write(tmp_path: Path, monkeypatch):
    root = tmp_path
    upload_dir = root / "papers" / "uploads"

    monkeypatch.setattr(ingestion, "PROJECT_ROOT", root)
    monkeypatch.setattr(ingestion, "PAPERS_DIR", root / "papers")
    monkeypatch.setattr(ingestion, "PAPERS_LIBRARY_DIR", root / "papers" / "library")
    monkeypatch.setattr(ingestion, "PAPERS_UPLOADS_DIR", upload_dir)

    with pytest.raises(ValueError, match="invalid_pdf_magic"):
        ingestion.ingest_uploaded_pdf(b"not a pdf", "paper.pdf")

    assert not upload_dir.exists()


def test_ingest_uploaded_pdf_records_metadata_only_scan_event(tmp_path: Path, monkeypatch):
    root = tmp_path
    metadata_dir = root / "metadata"
    upload_dir = root / "papers" / "uploads"

    monkeypatch.setattr(ingestion, "PROJECT_ROOT", root)
    monkeypatch.setattr(ingestion, "PAPERS_DIR", root / "papers")
    monkeypatch.setattr(ingestion, "PAPERS_LIBRARY_DIR", root / "papers" / "library")
    monkeypatch.setattr(ingestion, "PAPERS_UPLOADS_DIR", upload_dir)
    monkeypatch.setattr("src.runtime.RUNTIME_EVENTS_FILE", metadata_dir / "runtime_events.jsonl")

    with pytest.raises(ValueError, match="active_content_javascript"):
        ingestion.ingest_uploaded_pdf(b"%PDF-1.7\n1 0 obj << /JavaScript 2 0 R >>\n%%EOF", "secret-title.pdf")

    events = [
        json.loads(line)
        for line in (metadata_dir / "runtime_events.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert events[0]["kind"] == "upload_scan"
    assert events[0]["code"] == "upload_scan_blocked"
    assert events[0]["metadata"]["reason_codes"] == ["active_content_javascript"]
    assert "secret-title" not in json.dumps(events[0], ensure_ascii=False)


def test_resolve_unique_path_adds_suffix(tmp_path: Path):
    target = tmp_path / "paper.pdf"
    target.write_text("already here", encoding="utf-8")

    assert ingestion._resolve_unique_path(target) == tmp_path / "paper-1.pdf"


def test_clear_directory_contents_keeps_runtime_directory(tmp_path: Path):
    runtime_dir = tmp_path / "faiss_index"
    nested = runtime_dir / "nested"
    nested.mkdir(parents=True)
    (runtime_dir / "index.faiss").write_text("old index", encoding="utf-8")
    (nested / "stale").write_text("old nested data", encoding="utf-8")

    ingestion._clear_directory_contents(runtime_dir)

    assert runtime_dir.exists()
    assert list(runtime_dir.iterdir()) == []


def test_save_rebuilt_vector_store_replaces_after_success(tmp_path: Path):
    index_dir = tmp_path / "faiss_index"
    index_dir.mkdir()
    (index_dir / "index.faiss").write_text("old index", encoding="utf-8")
    (index_dir / "index.pkl").write_text("old metadata", encoding="utf-8")

    class FakeStore:
        def save_local(self, path: str) -> None:
            target = Path(path)
            target.mkdir(parents=True)
            (target / "index.faiss").write_text("new index", encoding="utf-8")
            (target / "index.pkl").write_text("new metadata", encoding="utf-8")

    ingestion._save_rebuilt_vector_store(FakeStore(), index_dir)

    assert (index_dir / "index.faiss").read_text(encoding="utf-8") == "new index"
    assert (index_dir / "index.pkl").read_text(encoding="utf-8") == "new metadata"
    assert not (index_dir / ".rebuild_tmp").exists()


def test_save_rebuilt_vector_store_preserves_live_index_on_failure(tmp_path: Path):
    index_dir = tmp_path / "faiss_index"
    index_dir.mkdir()
    (index_dir / "index.faiss").write_text("old index", encoding="utf-8")

    class FailingStore:
        def save_local(self, path: str) -> None:
            target = Path(path)
            target.mkdir(parents=True)
            (target / "partial").write_text("partial", encoding="utf-8")
            raise RuntimeError("save failed")

    with pytest.raises(RuntimeError, match="save failed"):
        ingestion._save_rebuilt_vector_store(FailingStore(), index_dir)

    assert (index_dir / "index.faiss").read_text(encoding="utf-8") == "old index"
    assert not (index_dir / ".rebuild_tmp").exists()


def test_refresh_paper_metadata_marks_active_indexed(tmp_path: Path, monkeypatch):
    root = tmp_path
    library = root / "papers" / "library"
    index_dir = root / "faiss_index"
    library.mkdir(parents=True)
    index_dir.mkdir()
    paper = library / "paper.pdf"
    paper.write_bytes(b"%PDF-1.4 test")
    (index_dir / "index.faiss").write_bytes(b"index")
    (index_dir / "active_papers.json").write_text(
        '["papers/library/paper.pdf"]',
        encoding="utf-8",
    )
    (library / "manifest.json").write_text(
        (
            '{"paper.pdf": {'
            '"title": "Paper", '
            '"doi": "10.1234/paper", '
            '"venue": "Test Venue", '
            '"topic_tags": ["SMC", "flux"]'
            "}}"
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(ingestion, "PROJECT_ROOT", root)
    monkeypatch.setattr(ingestion, "PAPERS_DIR", root / "papers")
    monkeypatch.setattr(ingestion, "PAPERS_LIBRARY_DIR", library)
    monkeypatch.setattr(ingestion, "PAPERS_UPLOADS_DIR", root / "papers" / "uploads")
    monkeypatch.setattr(ingestion, "PAPER_LIBRARY_MANIFEST", library / "manifest.json")
    monkeypatch.setattr(ingestion, "FAISS_INDEX_DIR", index_dir)
    monkeypatch.setattr(ingestion, "ACTIVE_PAPERS_FILE", index_dir / "active_papers.json")
    monkeypatch.setattr("src.metadata.PROJECT_ROOT", root)
    monkeypatch.setattr("src.metadata.PAPERS_LIBRARY_DIR", library)
    monkeypatch.setattr("src.metadata.CORPUS_METADATA_FILE", root / "metadata" / "corpus.json")

    records = ingestion.refresh_paper_metadata()

    assert len(records) == 1
    assert records[0].title == "Paper"
    assert records[0].doi == "10.1234/paper"
    assert records[0].venue == "Test Venue"
    assert records[0].topic_tags == ["SMC", "flux"]
    assert records[0].active is True
    assert records[0].indexed_status == "indexed"


def test_extract_pdf_bibliographic_metadata_from_uploaded_pdf(tmp_path: Path):
    import fitz

    pdf_path = tmp_path / "uploads" / "paper.pdf"
    pdf_path.parent.mkdir()
    document = fitz.open()
    page = document.new_page()
    page.insert_text(
        (72, 72),
            (
                "Automatic Flux Observer Metadata\n"
                "Abstract\n"
                "This 2024 paper includes DOI 10.1234/flux.observer and arXiv:2401.12345."
            ),
        )
    document.set_metadata(
        {
            "title": "Automatic Flux Observer Metadata",
            "author": "Ada Control; Max Observer",
            "subject": "flux observer; sensorless control",
            "keywords": "PMSM, observer",
            "creationDate": "D:20240101000000",
        }
    )
    document.save(pdf_path)
    document.close()

    metadata = ingestion.extract_pdf_bibliographic_metadata(pdf_path)

    assert metadata["title"] == "Automatic Flux Observer Metadata"
    assert metadata["authors"] == "Ada Control; Max Observer"
    assert metadata["year"] == 2024
    assert metadata["doi"] == "10.1234/flux.observer"
    assert metadata["arxiv_id"] == "2401.12345"
    assert metadata["topic_tags"] == ["PMSM", "observer", "flux observer", "sensorless control"]


def test_extract_pdf_bibliographic_metadata_from_first_page_authors_and_keywords(tmp_path: Path):
    import fitz

    pdf_path = tmp_path / "uploads" / "first-page-paper.pdf"
    pdf_path.parent.mkdir()
    document = fitz.open()
    page = document.new_page()
    page.insert_text(
        (72, 72),
        (
            "Robust Flux Observer for Sensorless PMSM Control\n"
            "Alice Wang, Bob Chen and Carol Li\n"
            "Department of Electrical Engineering, Example University\n"
            "Index Terms: sliding mode observer, flux linkage, PMSM drives\n"
            "Abstract\n"
            "This 2025 paper studies sensorless observers."
        ),
    )
    document.set_metadata({"title": "", "author": "", "keywords": "", "subject": ""})
    document.save(pdf_path)
    document.close()

    metadata = ingestion.extract_pdf_bibliographic_metadata(pdf_path)

    assert metadata["title"] == "Robust Flux Observer for Sensorless PMSM Control"
    assert metadata["authors"] == "Alice Wang, Bob Chen and Carol Li"
    assert metadata["year"] == 2025
    assert metadata["topic_tags"] == ["sliding mode observer", "flux linkage", "PMSM drives"]


def test_extract_pdf_structure_markers_finds_layout_markers(tmp_path: Path):
    import fitz

    pdf_path = tmp_path / "structure.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text(
        (72, 72),
        (
            "PMSM model excerpt\n"
            "ud = Rsid + Ld did/dt\n"
            "Table 1. PMSM parameter summary\n"
            "Figure 2. PMSM control block diagram\n"
            "Algorithm 1. Observer tuning loop\n"
        ),
    )
    document.save(pdf_path)
    document.close()

    markers = ingestion.extract_pdf_structure_markers(pdf_path)

    assert any(marker.kind == "equation" and "ud = Rsid" in marker.text for marker in markers)
    assert any(marker.kind == "table" and "Table 1" in marker.text for marker in markers)
    assert any(marker.kind == "figure" and "Figure 2" in marker.text for marker in markers)
    assert any(marker.kind == "algorithm" and "Algorithm 1" in marker.text for marker in markers)
    assert {marker.page for marker in markers} == {1}


def test_refresh_paper_metadata_uses_extracted_upload_metadata(tmp_path: Path, monkeypatch):
    import fitz

    root = tmp_path
    library = root / "papers" / "library"
    upload_dir = root / "papers" / "uploads"
    library.mkdir(parents=True)
    upload_dir.mkdir(parents=True)
    uploaded = upload_dir / "upload.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Uploaded Observer Paper\nDOI 10.5678/uploaded.paper")
    document.set_metadata({"author": "Upload Author", "keywords": "uploaded, observer"})
    document.save(uploaded)
    document.close()

    monkeypatch.setattr(ingestion, "PROJECT_ROOT", root)
    monkeypatch.setattr(ingestion, "PAPERS_DIR", root / "papers")
    monkeypatch.setattr(ingestion, "PAPERS_LIBRARY_DIR", library)
    monkeypatch.setattr(ingestion, "PAPERS_UPLOADS_DIR", upload_dir)
    monkeypatch.setattr(ingestion, "PAPER_LIBRARY_MANIFEST", library / "manifest.json")
    monkeypatch.setattr(ingestion, "FAISS_INDEX_DIR", root / "faiss_index")
    monkeypatch.setattr(ingestion, "ACTIVE_PAPERS_FILE", root / "faiss_index" / "active_papers.json")
    monkeypatch.setattr("src.metadata.PROJECT_ROOT", root)
    monkeypatch.setattr("src.metadata.PAPERS_LIBRARY_DIR", library)
    monkeypatch.setattr("src.metadata.CORPUS_METADATA_FILE", root / "metadata" / "corpus.json")

    records = ingestion.refresh_paper_metadata()

    assert records[0].source_path == "papers/uploads/upload.pdf"
    assert records[0].title == "Uploaded Observer Paper"
    assert records[0].authors == "Upload Author"
    assert records[0].doi == "10.5678/uploaded.paper"
    assert records[0].topic_tags == ["uploaded", "observer"]


def test_manifest_metadata_overrides_extracted_pdf_metadata(tmp_path: Path, monkeypatch):
    import fitz

    root = tmp_path
    library = root / "papers" / "library"
    library.mkdir(parents=True)
    paper = library / "paper.pdf"
    document = fitz.open()
    document.new_page().insert_text((72, 72), "Extracted Title")
    document.set_metadata({"title": "Extracted Title", "author": "Extracted Author"})
    document.save(paper)
    document.close()
    (library / "manifest.json").write_text(
        '{"paper.pdf": {"title": "Curated Title", "authors": "Curated Author"}}',
        encoding="utf-8",
    )

    monkeypatch.setattr(ingestion, "PROJECT_ROOT", root)
    monkeypatch.setattr(ingestion, "PAPERS_DIR", root / "papers")
    monkeypatch.setattr(ingestion, "PAPERS_LIBRARY_DIR", library)
    monkeypatch.setattr(ingestion, "PAPERS_UPLOADS_DIR", root / "papers" / "uploads")
    monkeypatch.setattr(ingestion, "PAPER_LIBRARY_MANIFEST", library / "manifest.json")
    monkeypatch.setattr(ingestion, "FAISS_INDEX_DIR", root / "faiss_index")
    monkeypatch.setattr(ingestion, "ACTIVE_PAPERS_FILE", root / "faiss_index" / "active_papers.json")
    monkeypatch.setattr("src.metadata.PROJECT_ROOT", root)
    monkeypatch.setattr("src.metadata.PAPERS_LIBRARY_DIR", library)
    monkeypatch.setattr("src.metadata.CORPUS_METADATA_FILE", root / "metadata" / "corpus.json")

    records = ingestion.refresh_paper_metadata()

    assert records[0].title == "Curated Title"
    assert records[0].authors == "Curated Author"


def test_set_active_paper_source_paths_persists_selection(tmp_path: Path, monkeypatch):
    root = tmp_path
    library = root / "papers" / "library"
    upload_dir = root / "papers" / "uploads"
    index_dir = root / "faiss_index"
    library.mkdir(parents=True)
    upload_dir.mkdir(parents=True)
    first = library / "first.pdf"
    second = upload_dir / "second.pdf"
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    monkeypatch.setattr(ingestion, "PROJECT_ROOT", root)
    monkeypatch.setattr(ingestion, "PAPERS_DIR", root / "papers")
    monkeypatch.setattr(ingestion, "PAPERS_LIBRARY_DIR", library)
    monkeypatch.setattr(ingestion, "PAPERS_UPLOADS_DIR", upload_dir)
    monkeypatch.setattr(ingestion, "PAPER_LIBRARY_MANIFEST", library / "manifest.json")
    monkeypatch.setattr(ingestion, "FAISS_INDEX_DIR", index_dir)
    monkeypatch.setattr(ingestion, "ACTIVE_PAPERS_FILE", index_dir / "active_papers.json")
    monkeypatch.setattr("src.metadata.PROJECT_ROOT", root)
    monkeypatch.setattr("src.metadata.PAPERS_LIBRARY_DIR", library)
    monkeypatch.setattr("src.metadata.CORPUS_METADATA_FILE", root / "metadata" / "corpus.json")

    records = ingestion.set_active_paper_source_paths(["papers/uploads/second.pdf"])

    assert (index_dir / "active_papers.json").read_text(encoding="utf-8").strip() == (
        '[\n  "papers/uploads/second.pdf"\n]'
    )
    assert [record.source_path for record in records if record.active] == [
        "papers/uploads/second.pdf"
    ]


def test_set_active_paper_source_paths_rejects_unselectable(tmp_path: Path, monkeypatch):
    root = tmp_path
    library = root / "papers" / "library"
    library.mkdir(parents=True)
    (library / "paper.pdf").write_bytes(b"paper")

    monkeypatch.setattr(ingestion, "PROJECT_ROOT", root)
    monkeypatch.setattr(ingestion, "PAPERS_DIR", root / "papers")
    monkeypatch.setattr(ingestion, "PAPERS_LIBRARY_DIR", library)
    monkeypatch.setattr(ingestion, "PAPERS_UPLOADS_DIR", root / "papers" / "uploads")

    with pytest.raises(ValueError, match="selectable corpus"):
        ingestion.set_active_paper_source_paths(["papers/library/missing.pdf"])


def test_ingest_uploaded_pdf_reuses_indexed_duplicate(tmp_path: Path, monkeypatch):
    root = tmp_path
    upload_dir = root / "papers" / "uploads"
    library = root / "papers" / "library"
    index_dir = root / "faiss_index"
    metadata_dir = root / "metadata"
    upload_dir.mkdir(parents=True)
    library.mkdir(parents=True)
    index_dir.mkdir()
    pdf_bytes = _minimal_pdf_bytes("duplicate upload")
    existing = upload_dir / "existing.pdf"
    existing.write_bytes(pdf_bytes)
    (index_dir / "index.faiss").write_bytes(b"index")

    monkeypatch.setattr(ingestion, "PROJECT_ROOT", root)
    monkeypatch.setattr(ingestion, "PAPERS_DIR", root / "papers")
    monkeypatch.setattr(ingestion, "PAPERS_LIBRARY_DIR", library)
    monkeypatch.setattr(ingestion, "PAPERS_UPLOADS_DIR", upload_dir)
    monkeypatch.setattr(ingestion, "PAPER_LIBRARY_MANIFEST", library / "manifest.json")
    monkeypatch.setattr(ingestion, "FAISS_INDEX_DIR", index_dir)
    monkeypatch.setattr(ingestion, "ACTIVE_PAPERS_FILE", index_dir / "active_papers.json")
    monkeypatch.setattr("src.metadata.PROJECT_ROOT", root)
    monkeypatch.setattr("src.metadata.PAPERS_LIBRARY_DIR", library)
    monkeypatch.setattr("src.metadata.CORPUS_METADATA_FILE", metadata_dir / "corpus.json")
    monkeypatch.setattr("src.metadata.CORPUS_METADATA_DB_FILE", metadata_dir / "corpus.sqlite3")
    monkeypatch.setattr("src.metadata.CHUNK_METADATA_DB_FILE", metadata_dir / "chunks.sqlite3")

    store = ingestion.CorpusMetadataStore()
    store.upsert_paper(
        existing,
        active=True,
        indexed_status="indexed",
        chunk_count=7,
    )

    def fail_if_reindexed(_path):
        raise AssertionError("duplicate upload should reuse the indexed existing PDF")

    monkeypatch.setattr(ingestion, "load_pdf", fail_if_reindexed)

    path, chunk_count = ingestion.ingest_uploaded_pdf(pdf_bytes, "same-content.pdf")

    assert path == existing
    assert chunk_count == 7
    assert sorted(path.name for path in upload_dir.glob("*.pdf")) == ["existing.pdf"]
    assert (index_dir / "active_papers.json").read_text(encoding="utf-8").strip() == (
        '[\n  "papers/uploads/existing.pdf"\n]'
    )


def test_rebuild_vector_store_records_chunk_metadata(tmp_path: Path, monkeypatch):
    root = tmp_path
    library = root / "papers" / "library"
    metadata_dir = root / "metadata"
    library.mkdir(parents=True)
    paper = library / "paper.pdf"
    paper.write_bytes(b"%PDF-1.4 test")
    chunk = Document(
        page_content="chunk text",
        metadata={"source": "paper.pdf", "source_path": "papers/library/paper.pdf", "page": 1},
    )

    monkeypatch.setattr(ingestion, "PROJECT_ROOT", root)
    monkeypatch.setattr(ingestion, "PAPERS_LIBRARY_DIR", library)
    monkeypatch.setattr(ingestion, "PAPER_LIBRARY_MANIFEST", library / "manifest.json")
    monkeypatch.setattr("src.metadata.PROJECT_ROOT", root)
    monkeypatch.setattr("src.metadata.PAPERS_LIBRARY_DIR", library)
    monkeypatch.setattr("src.metadata.CORPUS_METADATA_FILE", metadata_dir / "corpus.json")
    monkeypatch.setattr("src.metadata.CORPUS_METADATA_DB_FILE", metadata_dir / "corpus.sqlite3")
    monkeypatch.setattr("src.metadata.CHUNK_METADATA_DB_FILE", metadata_dir / "chunks.sqlite3")
    monkeypatch.setattr(ingestion, "load_pdfs", lambda _paths, **_kwargs: [chunk])
    monkeypatch.setattr(ingestion, "split_documents", lambda _docs, **_kwargs: [chunk])
    monkeypatch.setattr(ingestion, "build_vector_store", lambda _docs, *, rebuild, **_kwargs: object())
    monkeypatch.setattr(ingestion, "save_active_paper_paths", lambda _paths: None)

    _store, chunk_count = ingestion.rebuild_vector_store_from_pdfs([paper])

    chunks = ChunkMetadataStore(metadata_dir / "chunks.sqlite3").list_chunks(
        source_path="papers/library/paper.pdf"
    )
    assert chunk_count == 1
    assert len(chunks) == 1
    assert chunks[0].source_path == "papers/library/paper.pdf"
    assert chunks[0].page == 1
    assert chunks[0].char_count == len("chunk text")


def test_rebuild_vector_store_honors_cancellation_before_commit(tmp_path: Path, monkeypatch):
    import threading

    root = tmp_path
    library = root / "papers" / "library"
    library.mkdir(parents=True)
    paper = library / "paper.pdf"
    paper.write_bytes(b"%PDF-1.4 test")
    cancel_event = threading.Event()
    chunk = Document(
        page_content="chunk text",
        metadata={"source": "paper.pdf", "source_path": "papers/library/paper.pdf", "page": 1},
    )
    committed = {"active_saved": False}

    def split_and_cancel(_docs, **_kwargs):
        cancel_event.set()
        return [chunk]

    def fail_if_committed(_paths):
        committed["active_saved"] = True

    monkeypatch.setattr(ingestion, "PROJECT_ROOT", root)
    monkeypatch.setattr(ingestion, "PAPERS_LIBRARY_DIR", library)
    monkeypatch.setattr(ingestion, "load_pdfs", lambda _paths, **_kwargs: [chunk])
    monkeypatch.setattr(ingestion, "split_documents", split_and_cancel)
    monkeypatch.setattr(ingestion, "save_active_paper_paths", fail_if_committed)

    with pytest.raises(ingestion.IngestionCancelled):
        ingestion.rebuild_vector_store_from_pdfs([paper], cancel_event=cancel_event)

    assert committed["active_saved"] is False
