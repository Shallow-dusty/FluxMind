import hashlib
from pathlib import Path
import sqlite3

from langchain_core.documents import Document

from src.metadata import (
    ChunkMetadataStore,
    CorpusMetadataStore,
    CorpusProfileStore,
    atomic_write_json,
    safe_corpus_profile_report_filename,
)


def test_atomic_write_json_replaces_without_temp_file(tmp_path: Path):
    path = tmp_path / "metadata" / "corpus.json"

    atomic_write_json(path, {"version": 1, "papers": {"a": {"title": "A"}}})
    atomic_write_json(path, {"version": 1, "papers": {"b": {"title": "B"}}})

    assert '"b"' in path.read_text(encoding="utf-8")
    assert not list(path.parent.glob(".corpus.json.*.tmp"))


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
        manifest_entry={
            "title": "Paper Title",
            "year": 2026,
            "topic": "SMC",
            "doi": "10.1234/example",
            "venue": "Control Journal",
            "topic_tags": ["SMC", "observer"],
        },
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
    assert loaded.doi == "10.1234/example"
    assert loaded.venue == "Control Journal"
    assert loaded.topic_tags == ["SMC", "observer"]
    assert len(loaded.checksum_sha256) == 64
    assert store.storage_status()["sqlite_rows"] == 1
    with sqlite3.connect(store.db_path) as conn:
        row = conn.execute(
            "SELECT active, indexed_status, chunk_count, doi, venue, topic_tags FROM papers WHERE source_path = ?",
            ("papers/library/paper.pdf",),
        ).fetchone()
    assert row == (1, "indexed", 7, "10.1234/example", "Control Journal", '["SMC", "observer"]')


def test_corpus_metadata_store_derives_arxiv_id_and_topic_tags(tmp_path: Path, monkeypatch):
    root = tmp_path
    library = root / "papers" / "library"
    library.mkdir(parents=True)
    paper = library / "arxiv-2510-18420-smc-pmsm-review.pdf"
    paper.write_bytes(b"%PDF-1.4 test")

    monkeypatch.setattr("src.metadata.PROJECT_ROOT", root)
    monkeypatch.setattr("src.metadata.PAPERS_LIBRARY_DIR", library)
    store = CorpusMetadataStore(root / "metadata" / "corpus.json")

    record = store.upsert_paper(
        paper,
        manifest_entry={
            "title": "Review",
            "topic": "PMSM SMC review",
            "source_url": "https://arxiv.org/abs/2510.18420v1",
        },
    )

    assert record.arxiv_id == "2510.18420"
    assert record.topic_tags == ["PMSM SMC review"]


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


def test_refresh_from_files_marks_deactivated_paper_available(tmp_path: Path, monkeypatch):
    root = tmp_path
    library = root / "papers" / "library"
    library.mkdir(parents=True)
    paper = library / "paper.pdf"
    paper.write_bytes(b"paper")

    monkeypatch.setattr("src.metadata.PROJECT_ROOT", root)
    monkeypatch.setattr("src.metadata.PAPERS_LIBRARY_DIR", library)
    store = CorpusMetadataStore(root / "metadata" / "corpus.json")
    store.upsert_paper(paper, active=True, indexed_status="indexed")

    records = store.refresh_from_files([paper], active_paths=[])

    assert records[0].active is False
    assert records[0].indexed_status == "available"


def test_corpus_metadata_sqlite_mirror_removes_missing_records(tmp_path: Path, monkeypatch):
    root = tmp_path
    library = root / "papers" / "library"
    library.mkdir(parents=True)
    first = library / "first.pdf"
    second = library / "second.pdf"
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    monkeypatch.setattr("src.metadata.PROJECT_ROOT", root)
    monkeypatch.setattr("src.metadata.PAPERS_LIBRARY_DIR", library)
    store = CorpusMetadataStore(root / "metadata" / "corpus.json")

    store.refresh_from_files([first, second], active_paths=[first])
    store.refresh_from_files([first], active_paths=[first])

    with sqlite3.connect(store.db_path) as conn:
        rows = conn.execute("SELECT source_path FROM papers ORDER BY source_path").fetchall()
    assert rows == [("papers/library/first.pdf",)]


def test_corpus_metadata_sqlite_migrates_bibliographic_columns(tmp_path: Path, monkeypatch):
    root = tmp_path
    library = root / "papers" / "library"
    library.mkdir(parents=True)
    paper = library / "paper.pdf"
    paper.write_bytes(b"paper")
    metadata_dir = root / "metadata"
    metadata_dir.mkdir()
    db_path = metadata_dir / "corpus.sqlite3"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE papers (
                source_path TEXT PRIMARY KEY,
                paper_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                source_kind TEXT NOT NULL,
                checksum_sha256 TEXT NOT NULL,
                title TEXT NOT NULL,
                authors TEXT,
                year INTEGER,
                topic TEXT,
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

    monkeypatch.setattr("src.metadata.PROJECT_ROOT", root)
    monkeypatch.setattr("src.metadata.PAPERS_LIBRARY_DIR", library)
    store = CorpusMetadataStore(metadata_dir / "corpus.json", db_path=db_path)
    store.upsert_paper(paper, manifest_entry={"doi": "10.1234/migrated"})

    with sqlite3.connect(db_path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(papers)").fetchall()}
        row = conn.execute("SELECT doi FROM papers WHERE source_path = ?", ("papers/library/paper.pdf",)).fetchone()

    assert {"doi", "arxiv_id", "venue", "topic_tags"} <= columns
    assert row == ("10.1234/migrated",)


def test_corpus_profile_store_persists_named_selection(tmp_path: Path):
    store = CorpusProfileStore(tmp_path / "corpus_profiles.json")

    profile = store.upsert_profile(
        profile_id="SMC Core",
        name="SMC Core",
        description="Core local papers",
        source_paths=[
            "papers/library/first.pdf",
            "papers/library/first.pdf",
            "papers/library/second.pdf",
        ],
    )
    updated = store.upsert_profile(
        profile_id="smc-core",
        name="SMC Core Updated",
        source_paths=["papers/library/second.pdf"],
    )

    assert profile.profile_id == "smc-core"
    assert profile.source_paths == ["papers/library/first.pdf", "papers/library/second.pdf"]
    assert updated.created_at == profile.created_at
    assert updated.name == "SMC Core Updated"
    assert updated.paper_count == 1
    assert store.get_profile("SMC Core").source_paths == ["papers/library/second.pdf"]
    assert store.storage_status()["profiles"] == 1


def test_corpus_profile_report_filename_is_safe():
    assert (
        safe_corpus_profile_report_filename("SMC Core")
        == "fluxmind-corpus-profile-smc-core.md"
    )

    sensitive_id = 'abc"\r\nContent-Disposition: x-secret'
    filename = safe_corpus_profile_report_filename(sensitive_id)
    expected_id = hashlib.sha256(sensitive_id.encode()).hexdigest()[:16]

    assert filename == f"fluxmind-corpus-profile-{expected_id}.md"
    assert "secret" not in filename
    assert '"' not in filename
    assert "\r" not in filename
    assert "\n" not in filename


def test_chunk_metadata_store_replaces_source_chunks(tmp_path: Path):
    store = ChunkMetadataStore(tmp_path / "chunks.sqlite3")
    first = Document(
        page_content="alpha beta gamma",
        metadata={"source": "paper.pdf", "source_path": "papers/library/paper.pdf", "page": 3},
    )
    second = Document(
        page_content="delta",
        metadata={"source": "paper.pdf", "source_path": "papers/library/paper.pdf", "page": 4},
    )

    records = store.replace_for_sources([first, second], source_paths=["papers/library/paper.pdf"])
    replacement = Document(
        page_content="replacement",
        metadata={"source": "paper.pdf", "source_path": "papers/library/paper.pdf", "page": 5},
    )
    store.replace_for_sources([replacement], source_paths=["papers/library/paper.pdf"])

    assert len(records) == 2
    chunks = store.list_chunks(source_path="papers/library/paper.pdf")
    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].page == 5
    assert store.list_chunks(page=5)[0].chunk_id == chunks[0].chunk_id
    assert store.list_chunks(q="replacement")[0].chunk_id == chunks[0].chunk_id
    assert store.list_chunks(source_path="papers/library/paper.pdf", page=4) == []
    assert store.source_paths() == ["papers/library/paper.pdf"]
    assert store.storage_status()["sqlite_rows"] == 1
