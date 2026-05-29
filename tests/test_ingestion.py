from pathlib import Path

import pytest

from src import ingestion


def test_safe_pdf_name_preserves_unicode_and_strips_reserved_chars():
    assert ingestion._safe_pdf_name(" 磁链 观测<>:\"|?*.PDF ") == "磁链-观测.pdf"


def test_safe_pdf_name_uses_basename_to_prevent_path_traversal():
    assert ingestion._safe_pdf_name("../../磁链 观测.pdf") == "磁链-观测.pdf"


def test_safe_pdf_name_rejects_non_pdf():
    with pytest.raises(ValueError, match="Only PDF"):
        ingestion._safe_pdf_name("notes.txt")


def test_resolve_unique_path_adds_suffix(tmp_path: Path):
    target = tmp_path / "paper.pdf"
    target.write_text("already here", encoding="utf-8")

    assert ingestion._resolve_unique_path(target) == tmp_path / "paper-1.pdf"


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
        '{"paper.pdf": {"title": "Paper"}}',
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
    assert records[0].active is True
    assert records[0].indexed_status == "indexed"
