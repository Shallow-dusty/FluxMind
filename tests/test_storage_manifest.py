import hashlib
import json
import subprocess
import sys
from pathlib import Path

from src.storage_manifest import (
    RuntimeFileSpec,
    RuntimeGroupSpec,
    collect_runtime_backup_manifest,
    collect_runtime_restore_check,
    default_runtime_groups,
    format_runtime_backup_manifest_markdown,
    format_runtime_restore_check_markdown,
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


def test_default_runtime_manifest_includes_local_registry_state():
    groups = {group.name: group for group in default_runtime_groups()}
    metadata_files = {file_spec.name for file_spec in groups["metadata"].known_files}

    assert "api_key_registry_sqlite" in metadata_files
    assert "product_registry_sqlite" in metadata_files


def test_runtime_restore_check_accepts_matching_manifest_without_content(tmp_path: Path):
    root = tmp_path
    metadata_dir = root / "metadata"
    metadata_dir.mkdir()
    corpus = metadata_dir / "corpus.json"
    corpus.write_text('{"api_key":"secret-not-exported"}', encoding="utf-8")

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

    check = collect_runtime_restore_check(
        manifest,
        project_root=root,
        generated_at="2026-06-02T00:01:00+00:00",
    )

    assert check["mode"] == "local_runtime_restore_dry_run"
    assert check["content_restored"] is False
    assert check["delete_enabled"] is False
    assert check["ok"] is True
    assert check["checked_groups"] == 1
    assert check["checked_files"] == 1
    assert check["missing_groups"] == 0
    assert check["mismatched_groups"] == 0
    assert check["missing_files"] == 0
    assert check["mismatched_files"] == 0
    assert check["groups"][0]["status"] == "ok"
    assert check["groups"][0]["known_files"][0]["status"] == "ok"
    assert "secret-not-exported" not in str(check)


def test_runtime_restore_check_reports_missing_and_mismatched_targets(tmp_path: Path):
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_metadata = source_root / "metadata"
    source_jobs = source_root / "jobs"
    target_metadata = target_root / "metadata"
    source_metadata.mkdir(parents=True)
    source_jobs.mkdir()
    target_metadata.mkdir(parents=True)
    source_corpus = source_metadata / "corpus.json"
    source_jobs_file = source_jobs / "jobs.jsonl"
    source_corpus.write_text("original corpus", encoding="utf-8")
    source_jobs_file.write_text("job\n", encoding="utf-8")
    (target_metadata / "corpus.json").write_text("changed corpus", encoding="utf-8")

    manifest = collect_runtime_backup_manifest(
        project_root=source_root,
        generated_at="2026-06-02T00:00:00+00:00",
        groups=(
            RuntimeGroupSpec(
                name="metadata",
                path=source_metadata,
                restore_priority="required",
                known_files=(RuntimeFileSpec("corpus_json", source_corpus),),
            ),
            RuntimeGroupSpec(
                name="jobs",
                path=source_jobs,
                restore_priority="required",
                known_files=(RuntimeFileSpec("jobs_jsonl", source_jobs_file),),
            ),
        ),
    )

    check = collect_runtime_restore_check(manifest, project_root=target_root)

    assert check["ok"] is False
    assert check["missing_groups"] == 1
    assert check["mismatched_groups"] == 1
    assert check["missing_files"] == 1
    assert check["mismatched_files"] == 1
    groups = {group["name"]: group for group in check["groups"]}
    assert groups["metadata"]["status"] == "byte_count_mismatch"
    assert groups["metadata"]["known_files"][0]["status"] == "sha256_mismatch"
    assert groups["jobs"]["status"] == "missing"
    assert groups["jobs"]["known_files"][0]["status"] == "missing"


def test_runtime_restore_check_supports_absolute_manifest_paths(tmp_path: Path):
    root = tmp_path / "absolute-root"
    root.mkdir()
    file_path = root / "runtime.json"
    file_path.write_text("runtime", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "generated_at": "2026-06-02T00:00:00+00:00",
        "mode": "local_runtime_backup_manifest",
        "content_exported": False,
        "secrets_exported": False,
        "delete_enabled": False,
        "hash_algorithm": "sha256",
        "groups": [
            {
                "name": "absolute",
                "path": str(root),
                "exists": True,
                "restore_priority": "required",
                "files": 1,
                "bytes": file_path.stat().st_size,
                "known_files": [
                    {
                        "name": "runtime_json",
                        "path": str(file_path),
                        "exists": True,
                        "is_file": True,
                        "bytes": file_path.stat().st_size,
                        "sha256": hashlib.sha256(file_path.read_bytes()).hexdigest(),
                    }
                ],
            }
        ],
    }

    check = collect_runtime_restore_check(manifest, project_root=tmp_path / "other-root")

    assert check["ok"] is True
    assert check["groups"][0]["path"] == str(root)
    assert check["groups"][0]["known_files"][0]["path"] == str(file_path)


def test_runtime_restore_check_rejects_wrong_manifest_contract(tmp_path: Path):
    root = tmp_path
    metadata_dir = root / "metadata"
    metadata_dir.mkdir()
    corpus = metadata_dir / "corpus.json"
    corpus.write_text("corpus", encoding="utf-8")
    manifest = collect_runtime_backup_manifest(
        project_root=root,
        groups=(
            RuntimeGroupSpec(
                name="metadata",
                path=metadata_dir,
                restore_priority="required",
                known_files=(RuntimeFileSpec("corpus_json", corpus),),
            ),
        ),
    )
    manifest["schema_version"] = 2
    manifest["hash_algorithm"] = "md5"
    manifest["secrets_exported"] = True

    check = collect_runtime_restore_check(manifest, project_root=root)

    assert check["ok"] is False
    assert check["missing_groups"] == 0
    assert check["missing_files"] == 0
    assert "schema_version must be 1" in check["manifest_errors"]
    assert "hash_algorithm must be sha256" in check["manifest_errors"]
    assert "secrets_exported must be false" in check["manifest_errors"]


def test_runtime_restore_check_reports_malformed_groups_without_crashing():
    check = collect_runtime_restore_check(
        {
            "schema_version": 1,
            "mode": "local_runtime_backup_manifest",
            "content_exported": False,
            "secrets_exported": False,
            "delete_enabled": False,
            "hash_algorithm": "sha256",
            "groups": "not-a-list",
        }
    )

    assert check["ok"] is False
    assert check["checked_groups"] == 0
    assert check["manifest_errors"] == ["groups must be a list"]


def test_runtime_restore_check_markdown_is_no_action_no_secret(tmp_path: Path):
    root = tmp_path
    metadata_dir = root / "metadata"
    metadata_dir.mkdir()
    corpus = metadata_dir / "corpus.json"
    corpus.write_text("classified content", encoding="utf-8")
    manifest = collect_runtime_backup_manifest(
        project_root=root,
        groups=(
            RuntimeGroupSpec(
                name="metadata",
                path=metadata_dir,
                restore_priority="required",
                known_files=(RuntimeFileSpec("corpus_json", corpus),),
            ),
        ),
    )

    report = format_runtime_restore_check_markdown(
        collect_runtime_restore_check(manifest, project_root=root)
    )

    assert "# FluxMind Runtime Restore Dry Run" in report
    assert "No files are copied" in report
    assert "Content restored: false" in report
    assert "Delete enabled: false" in report
    assert "classified content" not in report


def test_runtime_manifest_cli_restore_check_emits_json(tmp_path: Path):
    root = tmp_path
    metadata_dir = root / "metadata"
    metadata_dir.mkdir()
    corpus = metadata_dir / "corpus.json"
    corpus.write_text("corpus", encoding="utf-8")
    manifest = collect_runtime_backup_manifest(
        project_root=root,
        groups=(
            RuntimeGroupSpec(
                name="metadata",
                path=metadata_dir,
                restore_priority="required",
                known_files=(RuntimeFileSpec("corpus_json", corpus),),
            ),
        ),
    )
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/runtime_manifest.py",
            "--restore-check",
            str(manifest_path),
            "--target-root",
            str(root),
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )

    data = json.loads(proc.stdout)
    assert data["mode"] == "local_runtime_restore_dry_run"
    assert data["ok"] is True


def test_runtime_manifest_cli_restore_check_exits_nonzero_on_mismatch(tmp_path: Path):
    root = tmp_path
    metadata_dir = root / "metadata"
    metadata_dir.mkdir()
    corpus = metadata_dir / "corpus.json"
    corpus.write_text("corpus", encoding="utf-8")
    manifest = collect_runtime_backup_manifest(
        project_root=root,
        groups=(
            RuntimeGroupSpec(
                name="metadata",
                path=metadata_dir,
                restore_priority="required",
                known_files=(RuntimeFileSpec("corpus_json", corpus),),
            ),
        ),
    )
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    corpus.write_text("changed", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/runtime_manifest.py",
            "--restore-check",
            str(manifest_path),
            "--target-root",
            str(root),
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
    )

    assert proc.returncode == 1
    assert json.loads(proc.stdout)["ok"] is False
