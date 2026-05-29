from pathlib import Path

from src.metadata import CorpusMetadataStore


def test_corpus_metadata_store_records_paper_state(tmp_path: Path, monkeypatch):
    root = tmp_path
    library = root / "papers" / "library"
    library.mkdir(parents=True)
    paper = library / "paper.pdf"
    paper.write_bytes(b"%PDF-1.4 test")

    monkeypatch.setattr("src.metadata.PROJECT_ROOT", root)
    monkeypatch.setattr("src.metadata.PAPERS_LIBRARY_DIR", library)
    store = CorpusMetadataStore(root / "metadata" / "corpus.json")

    record = store.upsert_paper(
        paper,
        manifest_entry={"title": "Paper Title", "year": 2026, "topic": "SMC"},
        active=True,
        indexed_status="indexed",
        chunk_count=7,
    )
    loaded = store.list_papers()[0]

    assert record.source_path == "papers/library/paper.pdf"
    assert loaded.title == "Paper Title"
    assert loaded.source_kind == "library"
    assert loaded.active is True
    assert loaded.indexed_status == "indexed"
    assert loaded.chunk_count == 7
    assert len(loaded.checksum_sha256) == 64


def test_refresh_from_files_marks_active_papers(tmp_path: Path, monkeypatch):
    root = tmp_path
    library = root / "papers" / "library"
    library.mkdir(parents=True)
    active = library / "active.pdf"
    inactive = library / "inactive.pdf"
    active.write_bytes(b"active")
    inactive.write_bytes(b"inactive")

    monkeypatch.setattr("src.metadata.PROJECT_ROOT", root)
    monkeypatch.setattr("src.metadata.PAPERS_LIBRARY_DIR", library)
    store = CorpusMetadataStore(root / "metadata" / "corpus.json")

    records = store.refresh_from_files([inactive, active], active_paths=[active])

    assert [record.source_path for record in records] == [
        "papers/library/active.pdf",
        "papers/library/inactive.pdf",
    ]
    assert records[0].active is True
    assert records[1].active is False
