import hashlib
import json
import subprocess
import sys
from pathlib import Path

from src.storage_manifest import (
    RuntimeFileSpec,
    RuntimeGroupSpec,
    collect_runtime_backup_manifest,
    format_runtime_backup_manifest_markdown,
)


def test_runtime_backup_manifest_lists_hashes_without_content(tmp_path: Path):
    root = tmp_path
    metadata_dir = root / "metadata"
    jobs_dir = root / "jobs"
    metadata_dir.mkdir()
    jobs_dir.mkdir()
    corpus = metadata_dir / "corpus.json"
    secret_job = jobs_dir / "jobs.jsonl"
    corpus.write_text('{"title":"paper","api_key":"not exported"}', encoding="utf-8")
    secret_job.write_text("provider secret text\n", encoding="utf-8")
    (root / ".env").write_text("LLM_API_KEY=secret\n", encoding="utf-8")

    manifest = collect_runtime_backup_manifest(
        project_root=root,
        generated_at="2026-06-02T00:00:00+00:00",
        groups=(
            RuntimeGroupSpec(
                name="metadata",
                path=metadata_dir,
                restore_priority="required",
                known_files=(RuntimeFileSpec("corpus_json", corpus),),
            ),
            RuntimeGroupSpec(
                name="jobs",
                path=jobs_dir,
                restore_priority="required",
                known_files=(RuntimeFileSpec("jobs_jsonl", secret_job),),
            ),
        ),
    )

    assert manifest["schema_version"] == 1
    assert manifest["content_exported"] is False
    assert manifest["secrets_exported"] is False
    assert manifest["delete_enabled"] is False
    assert manifest["env_file_present"] is True
    assert manifest["env_file_content_exported"] is False
    assert manifest["total_files"] == 2
    assert manifest["total_bytes"] == corpus.stat().st_size + secret_job.stat().st_size
    assert "not exported" not in str(manifest)
    assert "provider secret text" not in str(manifest)
    assert "LLM_API_KEY" not in str(manifest)

    groups = {group["name"]: group for group in manifest["groups"]}
    corpus_info = groups["metadata"]["known_files"][0]
    assert corpus_info["path"] == "metadata/corpus.json"
    assert corpus_info["sha256"] == hashlib.sha256(corpus.read_bytes()).hexdigest()
    assert groups["jobs"]["known_files"][0]["sha256"] == hashlib.sha256(secret_job.read_bytes()).hexdigest()


def test_runtime_backup_manifest_markdown_is_no_secret(tmp_path: Path):
    root = tmp_path
    metadata_dir = root / "metadata"
    metadata_dir.mkdir()
    corpus = metadata_dir / "corpus.json"
    corpus.write_text("classified content", encoding="utf-8")

    manifest = collect_runtime_backup_manifest(
        project_root=root,
        generated_at="2026-06-02T00:00:00+00:00",
        groups=(
            RuntimeGroupSpec(
                name="metadata",
                path=metadata_dir,
                restore_priority="required",
                known_files=(RuntimeFileSpec("corpus_json", corpus),),
            ),
        ),
    )

    report = format_runtime_backup_manifest_markdown(manifest)

    assert "# FluxMind Runtime Backup Manifest" in report
    assert "Content exported: false" in report
    assert "Secrets exported: false" in report
    assert "metadata/corpus.json" in report
    assert hashlib.sha256(corpus.read_bytes()).hexdigest() in report
    assert "classified content" not in report


def test_runtime_manifest_cli_emits_json():
    proc = subprocess.run(
        [sys.executable, "scripts/runtime_manifest.py"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )

    data = json.loads(proc.stdout)
    assert data["mode"] == "local_runtime_backup_manifest"
    assert data["content_exported"] is False
    assert "groups" in data
