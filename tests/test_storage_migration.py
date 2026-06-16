import json

import src.storage_migration as storage_migration
from src.jobs import LocalJobStore
from src.storage_migration import (
    format_storage_migration_rehearsal_markdown,
    run_storage_migration_rehearsal,
)


def _seed_runtime(root):
    metadata = root / "metadata"
    metadata.mkdir(parents=True)
    (metadata / "corpus.json").write_text(
        json.dumps({"version": 1, "papers": []}),
        encoding="utf-8",
    )
    (metadata / "runtime_events.jsonl").write_text("", encoding="utf-8")
    jobs = root / "jobs"
    jobs.mkdir()
    LocalJobStore(jobs / "jobs.jsonl")._ensure_sqlite()
    artifacts = root / "artifacts"
    artifacts.mkdir()
    (artifacts / "plot.svg").write_text("<svg />", encoding="utf-8")
    nested = artifacts / "nested"
    nested.mkdir()
    (nested / "plot.json").write_text("{}", encoding="utf-8")
    uploads = root / "papers" / "uploads"
    uploads.mkdir(parents=True)
    (uploads / "paper.pdf").write_bytes(b"%PDF")
    index = root / "faiss_index"
    index.mkdir()
    (index / "active_papers.json").write_text("[]", encoding="utf-8")
    (root / ".env").write_text("SECRET_TOKEN=do-not-copy\n", encoding="utf-8")


def test_storage_migration_rehearsal_copies_and_verifies_runtime_without_secrets(tmp_path):
    source = tmp_path / "source"
    staging = tmp_path / "staging"
    source.mkdir()
    _seed_runtime(source)

    status = run_storage_migration_rehearsal(
        project_root=source,
        staging_root=staging,
        generated_at="2026-06-16T00:00:00+00:00",
    )

    payload = json.dumps(status, ensure_ascii=False, sort_keys=True)
    assert status["rehearsal_ok"] is True
    assert status["content_copied_to_staging"] is True
    assert status["content_exported_in_report"] is False
    assert status["secrets_copied"] is False
    assert status["secrets_exported"] is False
    assert status["summary"]["source_preflight_ok"] is True
    assert status["summary"]["restore_check_ok"] is True
    assert status["summary"]["staged_storage_schema_ok"] is True
    assert status["summary"]["copied_files"] >= 5
    assert (staging / "metadata" / "corpus.json").exists()
    assert (staging / "jobs" / "jobs.sqlite3").exists()
    assert not (staging / ".env").exists()
    assert "do-not-copy" not in payload


def test_storage_migration_rehearsal_generates_timestamp_when_not_supplied(tmp_path):
    source = tmp_path / "source"
    staging = tmp_path / "staging"
    source.mkdir()
    _seed_runtime(source)

    status = run_storage_migration_rehearsal(project_root=source, staging_root=staging)

    assert status["rehearsal_ok"] is True
    assert status["generated_at"]


def test_storage_migration_rehearsal_rejects_nonempty_staging_without_overwrite(tmp_path):
    source = tmp_path / "source"
    staging = tmp_path / "staging"
    source.mkdir()
    staging.mkdir()
    marker = staging / "marker.txt"
    marker.write_text("keep", encoding="utf-8")
    _seed_runtime(source)

    status = run_storage_migration_rehearsal(
        project_root=source,
        staging_root=staging,
        generated_at="2026-06-16T00:00:00+00:00",
    )

    assert status["rehearsal_ok"] is False
    assert "staging_root_not_empty" in status["blockers"]
    assert marker.read_text(encoding="utf-8") == "keep"


def test_storage_migration_rehearsal_can_overwrite_staging(tmp_path):
    source = tmp_path / "source"
    staging = tmp_path / "staging"
    source.mkdir()
    staging.mkdir()
    (staging / "marker.txt").write_text("remove", encoding="utf-8")
    _seed_runtime(source)

    status = run_storage_migration_rehearsal(
        project_root=source,
        staging_root=staging,
        overwrite_staging=True,
        generated_at="2026-06-16T00:00:00+00:00",
    )

    assert status["rehearsal_ok"] is True
    assert status["staging_root_overwritten"] is True
    assert not (staging / "marker.txt").exists()


def test_storage_migration_rehearsal_rejects_staging_inside_project(tmp_path):
    _seed_runtime(tmp_path)

    status = run_storage_migration_rehearsal(
        project_root=tmp_path,
        staging_root=tmp_path / "staging",
        generated_at="2026-06-16T00:00:00+00:00",
    )

    assert status["rehearsal_ok"] is False
    assert "staging_root_inside_project" in status["blockers"]


def test_storage_migration_rehearsal_skips_symlinks(tmp_path):
    source = tmp_path / "source"
    staging = tmp_path / "staging"
    source.mkdir()
    _seed_runtime(source)
    (source / "artifacts" / "leak").symlink_to(source / ".env")

    status = run_storage_migration_rehearsal(
        project_root=source,
        staging_root=staging,
        generated_at="2026-06-16T00:00:00+00:00",
    )

    assert status["rehearsal_ok"] is True
    assert status["summary"]["skipped_symlinks"] == 1
    assert not (staging / "artifacts" / "leak").exists()


def test_storage_migration_rehearsal_can_include_runtime_dependencies(tmp_path):
    source = tmp_path / "source"
    staging = tmp_path / "staging"
    source.mkdir()
    _seed_runtime(source)
    models = source / "models"
    models.mkdir()
    (models / "model.bin").write_bytes(b"model")

    status = run_storage_migration_rehearsal(
        project_root=source,
        staging_root=staging,
        include_runtime_dependencies=True,
        generated_at="2026-06-16T00:00:00+00:00",
    )

    groups = {group["name"]: group for group in status["copy"]["groups"]}
    assert status["rehearsal_ok"] is True
    assert groups["models"]["status"] == "copied"
    assert (staging / "models" / "model.bin").exists()


def test_storage_migration_copy_group_reports_runtime_dependency_skip(tmp_path):
    source = tmp_path / "models"
    source.mkdir()

    result = storage_migration._copy_runtime_group(
        name="models",
        source_path=source,
        target_path=tmp_path / "staging" / "models",
        restore_priority="runtime_dependency",
        include_runtime_dependencies=False,
    )

    assert result["status"] == "skipped_runtime_dependency"
    assert result["source_exists"] is True


def test_storage_migration_copy_group_rejects_file_source(tmp_path):
    source = tmp_path / "metadata"
    source.write_text("not a directory", encoding="utf-8")

    result = storage_migration._copy_runtime_group(
        name="metadata",
        source_path=source,
        target_path=tmp_path / "staging" / "metadata",
        restore_priority="required",
        include_runtime_dependencies=False,
    )

    assert result["status"] == "source_not_directory"
    assert result["errors"] == ["source_not_directory"]


def test_storage_migration_copy_group_reports_copy_failure(tmp_path, monkeypatch):
    source = tmp_path / "metadata"
    source.mkdir()
    (source / "corpus.json").write_text("{}", encoding="utf-8")

    def fail_copy(*_args, **_kwargs):
        raise OSError("copy failed")

    monkeypatch.setattr(storage_migration.shutil, "copy2", fail_copy)
    result = storage_migration._copy_runtime_group(
        name="metadata",
        source_path=source,
        target_path=tmp_path / "staging" / "metadata",
        restore_priority="required",
        include_runtime_dependencies=False,
    )

    assert result["status"] == "copy_failed"
    assert result["errors"] == ["copy_failed"]


def test_storage_migration_rehearsal_reports_source_preflight_failure(tmp_path, monkeypatch):
    source = tmp_path / "source"
    staging = tmp_path / "staging"
    source.mkdir()
    _seed_runtime(source)
    monkeypatch.setattr(
        storage_migration,
        "collect_platform_migration_preflight",
        lambda **_kwargs: {"preflight_ok": False, "activation_ready": False, "blockers": {}},
    )

    status = run_storage_migration_rehearsal(
        project_root=source,
        staging_root=staging,
        generated_at="2026-06-16T00:00:00+00:00",
    )

    assert status["rehearsal_ok"] is False
    assert "source_preflight_failed" in status["blockers"]


def test_storage_migration_rehearsal_reports_staged_restore_failure(tmp_path, monkeypatch):
    source = tmp_path / "source"
    staging = tmp_path / "staging"
    source.mkdir()
    _seed_runtime(source)
    monkeypatch.setattr(
        storage_migration,
        "collect_runtime_restore_check",
        lambda *_args, **_kwargs: {"ok": False, "manifest_errors": ["bad_manifest"]},
    )

    status = run_storage_migration_rehearsal(
        project_root=source,
        staging_root=staging,
        generated_at="2026-06-16T00:00:00+00:00",
    )

    assert status["rehearsal_ok"] is False
    assert "staged_restore_check_failed" in status["blockers"]


def test_storage_migration_rehearsal_reports_staged_schema_drift(tmp_path, monkeypatch):
    source = tmp_path / "source"
    staging = tmp_path / "staging"
    source.mkdir()
    _seed_runtime(source)
    monkeypatch.setattr(
        storage_migration,
        "storage_schema_status_for_root",
        lambda _root: {"ok": False, "problem_count": 1, "stores": []},
    )

    status = run_storage_migration_rehearsal(
        project_root=source,
        staging_root=staging,
        generated_at="2026-06-16T00:00:00+00:00",
    )

    assert status["rehearsal_ok"] is False
    assert "staged_storage_schema_drift" in status["blockers"]


def test_format_storage_migration_rehearsal_markdown_is_no_secret(tmp_path):
    source = tmp_path / "source"
    staging = tmp_path / "staging"
    source.mkdir()
    _seed_runtime(source)

    status = run_storage_migration_rehearsal(
        project_root=source,
        staging_root=staging,
        generated_at="2026-06-16T00:00:00+00:00",
    )
    markdown = format_storage_migration_rehearsal_markdown(status)

    assert "# FluxMind Runtime Migration Rehearsal" in markdown
    assert "Rehearsal OK: true" in markdown
    assert "Content exported in report: false" in markdown
    assert "Secrets copied: false" in markdown
    assert "do-not-copy" not in markdown
