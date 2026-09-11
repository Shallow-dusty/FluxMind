import os
from pathlib import Path

from src import admin
from src.admin import (
    AdminStatus,
    RuntimeDirectoryStatus,
    apply_retention_delete,
    collect_corpus_profile_status,
    collect_corpus_status,
    collect_retention_preview,
    format_admin_metrics,
    format_admin_status_report,
    format_corpus_profile_status_report,
)
from src.metadata import CorpusProfile


def _status() -> AdminStatus:
    return AdminStatus(
        runtime_dirs=[
            RuntimeDirectoryStatus(
                name="metadata",
                path="metadata",
                exists=True,
                writable=True,
                bytes=12,
            )
        ],
        jobs={
            "total": 3,
            "failed": 1,
            "by_status": {"succeeded": 2, "failed": 1},
        },
        corpus={
            "library_papers": 52,
            "active_papers": 5,
            "indexed_sources": 5,
            "rebuild_required": False,
        },
        artifacts={"total": 2, "bytes": 100},
        users={"total": 4, "active": 3},
        providers={
            "llm_model": "model",
            "image_backend": "openai",
            "execution_backend": "docker",
            "recent_failures": {"total_recent": 1},
        },
        activity={
            "queries": {"total_recent": 8},
            "retrieval": {"citation_failed_recent": 1},
        },
        storage={"total_runtime_bytes": 1000},
        config={},
    )


def test_admin_report_and_metrics_cover_operational_state():
    report = format_admin_status_report(_status())
    metrics = format_admin_metrics(_status())

    assert "# FluxMind Runtime Status" in report
    assert "Library papers: 52" in report
    assert "Recent provider failures: 1" in report
    assert "fluxmind_jobs_total 3" in metrics
    assert "fluxmind_citation_failures_recent 1" in metrics


def test_collect_corpus_status_detects_index_drift(tmp_path, monkeypatch):
    active = tmp_path / "papers" / "a.pdf"
    active.parent.mkdir(parents=True)
    active.write_bytes(b"pdf")
    index_dir = tmp_path / "faiss_index"
    index_dir.mkdir()
    (index_dir / "index.faiss").write_bytes(b"index")

    class FakeChunks:
        def source_paths(self):
            return ["papers/b.pdf"]

        def storage_status(self):
            return {"sqlite_rows": 10, "source_paths": 1}

    class FakeProfiles:
        def list_profiles(self):
            return []

    monkeypatch.setattr(admin, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(admin, "FAISS_INDEX_DIR", index_dir)
    monkeypatch.setattr(admin, "load_library_manifest", lambda: {"a.pdf": {}})
    monkeypatch.setattr(admin, "load_active_paper_paths", lambda: [active])
    monkeypatch.setattr(admin, "ChunkMetadataStore", FakeChunks)
    monkeypatch.setattr(admin, "CorpusProfileStore", FakeProfiles)

    status = collect_corpus_status()

    assert status["library_papers"] == 1
    assert status["active_papers"] == 1
    assert status["indexed_sources"] == 1
    assert status["index_exists"] is True
    assert status["rebuild_required"] is True


def test_corpus_profile_status_and_report_show_sources(tmp_path, monkeypatch):
    source = tmp_path / "papers" / "a.pdf"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"pdf")
    profile = CorpusProfile(
        profile_id="smc",
        name="SMC",
        source_paths=["papers/a.pdf"],
        paper_count=1,
    )

    class FakeProfiles:
        def get_profile(self, profile_id):
            assert profile_id == "smc"
            return profile

    class FakeChunks:
        def source_paths(self):
            return ["papers/a.pdf"]

    monkeypatch.setattr(admin, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(admin, "CorpusProfileStore", FakeProfiles)
    monkeypatch.setattr(admin, "ChunkMetadataStore", FakeChunks)
    monkeypatch.setattr(admin, "load_active_paper_paths", lambda: [source])

    status = collect_corpus_profile_status("smc")
    report = format_corpus_profile_status_report(status)

    assert status["active"] is True
    assert status["indexed"] is True
    assert status["rebuild_required"] is False
    assert "`papers/a.pdf`" in report


def _old_file(path: Path, *, now: float, age_days: int, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    old = now - age_days * 86400
    os.utime(path, (old, old))


def test_retention_preview_lists_only_old_runtime_files(tmp_path, monkeypatch):
    now = 2_000_000_000.0
    uploads = tmp_path / "papers" / "uploads"
    artifacts = tmp_path / "artifacts"
    _old_file(uploads / "old.pdf", now=now, age_days=10, content=b"old")
    _old_file(uploads / "new.pdf", now=now, age_days=1, content=b"new")
    _old_file(artifacts / "run" / "plot.png", now=now, age_days=10, content=b"plot")
    _old_file(artifacts / "artifacts.sqlite3", now=now, age_days=10, content=b"db")

    monkeypatch.setattr(admin, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(admin, "PAPERS_UPLOADS_DIR", uploads)
    monkeypatch.setattr(admin, "ARTIFACTS_DIR", artifacts)
    monkeypatch.setattr(admin, "RETENTION_DELETE_ENABLED", False)

    preview = collect_retention_preview(
        upload_days=7,
        artifact_days=7,
        now_ts=now,
    )

    assert preview["candidate_count"] == 2
    assert {item["path"] for item in preview["candidates"]} == {
        "papers/uploads/old.pdf",
        "artifacts/run/plot.png",
    }


def test_retention_delete_respects_switch(tmp_path, monkeypatch):
    now = 2_000_000_000.0
    uploads = tmp_path / "papers" / "uploads"
    artifacts = tmp_path / "artifacts"
    old_upload = uploads / "old.pdf"
    old_artifact = artifacts / "plot.png"
    _old_file(old_upload, now=now, age_days=10, content=b"old")
    _old_file(old_artifact, now=now, age_days=10, content=b"plot")

    monkeypatch.setattr(admin, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(admin, "PAPERS_UPLOADS_DIR", uploads)
    monkeypatch.setattr(admin, "ARTIFACTS_DIR", artifacts)
    monkeypatch.setattr(admin, "RETENTION_DELETE_ENABLED", False)
    disabled = apply_retention_delete(
        upload_days=7,
        artifact_days=7,
        now_ts=now,
    )
    assert disabled["deleted_count"] == 0
    assert old_upload.exists()

    monkeypatch.setattr(admin, "RETENTION_DELETE_ENABLED", True)
    enabled = apply_retention_delete(
        upload_days=7,
        artifact_days=7,
        now_ts=now,
    )
    assert enabled["deleted_count"] == 2
    assert not old_upload.exists()
    assert not old_artifact.exists()
