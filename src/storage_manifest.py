"""No-secret runtime backup manifest helpers."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import (
    ACTIVE_PAPERS_FILE,
    API_KEY_REGISTRY_FILE,
    ARTIFACTS_DIR,
    CHUNK_METADATA_DB_FILE,
    CORPUS_METADATA_DB_FILE,
    CORPUS_METADATA_FILE,
    CORPUS_PROFILES_FILE,
    FAISS_INDEX_DIR,
    JOBS_DB_FILE,
    JOBS_DIR,
    JOBS_FILE,
    METADATA_DIR,
    PAPERS_UPLOADS_DIR,
    PROJECT_ROOT,
    RUNTIME_EVENTS_FILE,
)


@dataclass(frozen=True)
class RuntimeFileSpec:
    name: str
    path: Path


@dataclass(frozen=True)
class RuntimeGroupSpec:
    name: str
    path: Path
    restore_priority: str
    known_files: tuple[RuntimeFileSpec, ...] = ()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _relative_path(path: Path, *, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_totals(path: Path) -> tuple[int, int]:
    files = 0
    bytes_total = 0
    if not path.exists():
        return files, bytes_total
    for item in path.rglob("*"):
        if item.is_symlink() or not item.is_file():
            continue
        files += 1
        bytes_total += item.stat().st_size
    return files, bytes_total


def _file_manifest(file_spec: RuntimeFileSpec, *, project_root: Path) -> dict[str, Any]:
    path = file_spec.path
    exists = path.exists()
    is_file = path.is_file() and not path.is_symlink()
    return {
        "name": file_spec.name,
        "path": _relative_path(path, project_root=project_root),
        "exists": exists,
        "is_file": is_file,
        "bytes": path.stat().st_size if is_file else 0,
        "sha256": _sha256_file(path) if is_file else None,
    }


def _group_manifest(group: RuntimeGroupSpec, *, project_root: Path) -> dict[str, Any]:
    files, bytes_total = _directory_totals(group.path)
    return {
        "name": group.name,
        "path": _relative_path(group.path, project_root=project_root),
        "exists": group.path.exists(),
        "restore_priority": group.restore_priority,
        "files": files,
        "bytes": bytes_total,
        "known_files": [
            _file_manifest(file_spec, project_root=project_root)
            for file_spec in group.known_files
        ],
    }


def _manifest_path(value: str, *, project_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return project_root / path


def _manifest_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _restore_manifest_errors(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if manifest.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if manifest.get("mode") != "local_runtime_backup_manifest":
        errors.append("mode must be local_runtime_backup_manifest")
    if manifest.get("hash_algorithm") != "sha256":
        errors.append("hash_algorithm must be sha256")
    if manifest.get("content_exported") is not False:
        errors.append("content_exported must be false")
    if manifest.get("secrets_exported") is not False:
        errors.append("secrets_exported must be false")
    if manifest.get("delete_enabled") is not False:
        errors.append("delete_enabled must be false")
    if not isinstance(manifest.get("groups"), list):
        errors.append("groups must be a list")
    return errors


def default_runtime_groups() -> tuple[RuntimeGroupSpec, ...]:
    """Return runtime trees that source deploys intentionally exclude."""
    return (
        RuntimeGroupSpec(
            name="metadata",
            path=METADATA_DIR,
            restore_priority="required",
            known_files=(
                RuntimeFileSpec("corpus_json", CORPUS_METADATA_FILE),
                RuntimeFileSpec("corpus_profiles_json", CORPUS_PROFILES_FILE),
                RuntimeFileSpec("corpus_sqlite", CORPUS_METADATA_DB_FILE),
                RuntimeFileSpec("chunks_sqlite", CHUNK_METADATA_DB_FILE),
                RuntimeFileSpec("api_key_registry_sqlite", API_KEY_REGISTRY_FILE),
                RuntimeFileSpec("runtime_events_jsonl", RUNTIME_EVENTS_FILE),
            ),
        ),
        RuntimeGroupSpec(
            name="jobs",
            path=JOBS_DIR,
            restore_priority="required",
            known_files=(
                RuntimeFileSpec("jobs_jsonl", JOBS_FILE),
                RuntimeFileSpec("jobs_sqlite", JOBS_DB_FILE),
            ),
        ),
        RuntimeGroupSpec(
            name="artifacts",
            path=ARTIFACTS_DIR,
            restore_priority="required",
            known_files=(RuntimeFileSpec("artifacts_sqlite", ARTIFACTS_DIR / "artifacts.sqlite3"),),
        ),
        RuntimeGroupSpec(
            name="uploads",
            path=PAPERS_UPLOADS_DIR,
            restore_priority="required",
        ),
        RuntimeGroupSpec(
            name="faiss_index",
            path=FAISS_INDEX_DIR,
            restore_priority="required",
            known_files=(
                RuntimeFileSpec("index_faiss", FAISS_INDEX_DIR / "index.faiss"),
                RuntimeFileSpec("index_pkl", FAISS_INDEX_DIR / "index.pkl"),
                RuntimeFileSpec("active_papers_json", ACTIVE_PAPERS_FILE),
            ),
        ),
        RuntimeGroupSpec(
            name="models",
            path=PROJECT_ROOT / "models",
            restore_priority="runtime_dependency",
        ),
    )


def collect_runtime_backup_manifest(
    *,
    project_root: Path = PROJECT_ROOT,
    groups: tuple[RuntimeGroupSpec, ...] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Return a no-secret manifest for backing up excluded runtime state."""
    manifest_groups = [
        _group_manifest(group, project_root=project_root)
        for group in (groups if groups is not None else default_runtime_groups())
    ]
    env_file = project_root / ".env"
    return {
        "schema_version": 1,
        "generated_at": generated_at or _utc_now(),
        "mode": "local_runtime_backup_manifest",
        "content_exported": False,
        "secrets_exported": False,
        "delete_enabled": False,
        "hash_algorithm": "sha256",
        "project_root": _relative_path(project_root, project_root=project_root),
        "env_file_present": env_file.exists(),
        "env_file_content_exported": False,
        "total_files": sum(group["files"] for group in manifest_groups),
        "total_bytes": sum(group["bytes"] for group in manifest_groups),
        "groups": manifest_groups,
    }


def format_runtime_backup_manifest_markdown(manifest: dict[str, Any]) -> str:
    """Render a backup manifest as no-secret Markdown."""
    lines = [
        "# FluxMind Runtime Backup Manifest",
        "",
        "No runtime file contents or secrets are exported by this manifest.",
        "",
        f"- Generated at: {manifest.get('generated_at', '')}",
        f"- Mode: {manifest.get('mode', '')}",
        f"- Content exported: {str(manifest.get('content_exported', False)).lower()}",
        f"- Secrets exported: {str(manifest.get('secrets_exported', False)).lower()}",
        f"- Delete enabled: {str(manifest.get('delete_enabled', False)).lower()}",
        f"- Hash algorithm: {manifest.get('hash_algorithm', '')}",
        f"- Env file present: {str(manifest.get('env_file_present', False)).lower()}",
        f"- Env file content exported: {str(manifest.get('env_file_content_exported', False)).lower()}",
        f"- Total files: {manifest.get('total_files', 0)}",
        f"- Total bytes: {manifest.get('total_bytes', 0)}",
        "",
        "## Runtime Groups",
        "",
    ]
    for group in manifest.get("groups", []):
        lines.append(
            f"- {group.get('name', '')}: path={group.get('path', '')}, "
            f"priority={group.get('restore_priority', '')}, "
            f"exists={str(group.get('exists', False)).lower()}, "
            f"files={group.get('files', 0)}, bytes={group.get('bytes', 0)}"
        )
        for file_info in group.get("known_files", []):
            sha = file_info.get("sha256") or "missing"
            lines.append(
                f"  - {file_info.get('name', '')}: path={file_info.get('path', '')}, "
                f"exists={str(file_info.get('exists', False)).lower()}, "
                f"bytes={file_info.get('bytes', 0)}, sha256={sha}"
            )
    return "\n".join(lines)


def collect_runtime_restore_check(
    manifest: dict[str, Any],
    *,
    project_root: Path = PROJECT_ROOT,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Dry-run check that a target runtime tree matches a no-secret manifest."""
    checked_files = 0
    missing_groups = 0
    mismatched_groups = 0
    missing_files = 0
    mismatched_files = 0
    group_results: list[dict[str, Any]] = []
    manifest_errors = _restore_manifest_errors(manifest)
    manifest_groups = manifest.get("groups", [])
    if not isinstance(manifest_groups, list):
        manifest_groups = []

    for group in manifest_groups:
        if not isinstance(group, dict):
            manifest_errors.append("group entry must be an object")
            continue
        expected_group_exists = bool(group.get("exists"))
        group_path = _manifest_path(str(group.get("path", "")), project_root=project_root)
        group_exists = group_path.exists() and group_path.is_dir()
        expected_files = _manifest_int(group.get("files"))
        expected_bytes = _manifest_int(group.get("bytes"))
        actual_files, actual_group_bytes = _directory_totals(group_path) if group_exists else (0, 0)
        file_results: list[dict[str, Any]] = []
        group_ok = True
        group_status = "ok"
        if expected_group_exists and not group_exists:
            missing_groups += 1
            group_ok = False
            group_status = "missing"
        elif expected_group_exists and (
            actual_files != expected_files or actual_group_bytes != expected_bytes
        ):
            mismatched_groups += 1
            group_ok = False
            group_status = (
                "file_count_mismatch"
                if actual_files != expected_files
                else "byte_count_mismatch"
            )
        elif not expected_group_exists and group_exists:
            group_status = "extra_present"

        for file_info in group.get("known_files", []):
            expected_file_exists = bool(file_info.get("exists"))
            file_path = _manifest_path(str(file_info.get("path", "")), project_root=project_root)
            exists = file_path.exists()
            is_file = file_path.is_file() and not file_path.is_symlink()
            expected_sha = file_info.get("sha256")
            expected_file_bytes = _manifest_int(file_info.get("bytes"))
            actual_file_bytes = file_path.stat().st_size if is_file else 0
            actual_sha = _sha256_file(file_path) if is_file else None
            status = "ok"
            if expected_file_exists:
                checked_files += 1
                if not is_file:
                    status = "missing"
                    missing_files += 1
                    group_ok = False
                elif expected_sha and actual_sha != expected_sha:
                    status = "sha256_mismatch"
                    mismatched_files += 1
                    group_ok = False
                elif actual_file_bytes != expected_file_bytes:
                    status = "byte_count_mismatch"
                    mismatched_files += 1
                    group_ok = False
            elif exists:
                status = "extra_present"

            file_results.append(
                {
                    "name": file_info.get("name", ""),
                    "path": _relative_path(file_path, project_root=project_root),
                    "expected_exists": expected_file_exists,
                    "exists": exists,
                    "is_file": is_file,
                    "expected_bytes": expected_file_bytes,
                    "bytes": actual_file_bytes,
                    "expected_sha256": expected_sha,
                    "sha256": actual_sha,
                    "status": status,
                    "ok": status in {"ok", "extra_present"},
                }
            )

        group_results.append(
            {
                "name": group.get("name", ""),
                "path": _relative_path(group_path, project_root=project_root),
                "expected_exists": expected_group_exists,
                "exists": group_exists,
                "expected_files": expected_files,
                "files": actual_files,
                "expected_bytes": expected_bytes,
                "bytes": actual_group_bytes,
                "restore_priority": group.get("restore_priority", ""),
                "status": group_status,
                "ok": group_ok,
                "known_files": file_results,
            }
        )

    ok = (
        missing_groups == 0
        and mismatched_groups == 0
        and missing_files == 0
        and mismatched_files == 0
        and not manifest_errors
    )
    return {
        "schema_version": 1,
        "generated_at": generated_at or _utc_now(),
        "mode": "local_runtime_restore_dry_run",
        "source_manifest_generated_at": manifest.get("generated_at"),
        "source_manifest_mode": manifest.get("mode"),
        "project_root": _relative_path(project_root, project_root=project_root),
        "content_restored": False,
        "delete_enabled": False,
        "ok": ok,
        "manifest_errors": manifest_errors,
        "checked_groups": len(group_results),
        "checked_files": checked_files,
        "missing_groups": missing_groups,
        "mismatched_groups": mismatched_groups,
        "missing_files": missing_files,
        "mismatched_files": mismatched_files,
        "groups": group_results,
    }


def format_runtime_restore_check_markdown(check: dict[str, Any]) -> str:
    """Render a restore dry-run check as no-secret Markdown."""
    lines = [
        "# FluxMind Runtime Restore Dry Run",
        "",
        "No files are copied, overwritten, deleted, or restored by this check.",
        "",
        f"- Generated at: {check.get('generated_at', '')}",
        f"- Source manifest generated at: {check.get('source_manifest_generated_at', '')}",
        f"- Mode: {check.get('mode', '')}",
        f"- Content restored: {str(check.get('content_restored', False)).lower()}",
        f"- Delete enabled: {str(check.get('delete_enabled', False)).lower()}",
        f"- OK: {str(check.get('ok', False)).lower()}",
        f"- Manifest errors: {len(check.get('manifest_errors', []))}",
        f"- Checked groups: {check.get('checked_groups', 0)}",
        f"- Checked files: {check.get('checked_files', 0)}",
        f"- Missing groups: {check.get('missing_groups', 0)}",
        f"- Mismatched groups: {check.get('mismatched_groups', 0)}",
        f"- Missing files: {check.get('missing_files', 0)}",
        f"- Mismatched files: {check.get('mismatched_files', 0)}",
        "",
    ]
    if check.get("manifest_errors"):
        lines.extend(["## Manifest Errors", ""])
        for error in check.get("manifest_errors", []):
            lines.append(f"- {error}")
        lines.append("")
    lines.extend(["## Runtime Groups", ""])
    for group in check.get("groups", []):
        lines.append(
            f"- {group.get('name', '')}: path={group.get('path', '')}, "
            f"status={group.get('status', '')}, "
            f"expected_exists={str(group.get('expected_exists', False)).lower()}, "
            f"exists={str(group.get('exists', False)).lower()}, "
            f"expected_files={group.get('expected_files', 0)}, "
            f"files={group.get('files', 0)}, "
            f"expected_bytes={group.get('expected_bytes', 0)}, "
            f"bytes={group.get('bytes', 0)}, "
            f"ok={str(group.get('ok', False)).lower()}"
        )
        for file_info in group.get("known_files", []):
            lines.append(
                f"  - {file_info.get('name', '')}: path={file_info.get('path', '')}, "
                f"status={file_info.get('status', '')}, "
                f"expected_bytes={file_info.get('expected_bytes', 0)}, "
                f"bytes={file_info.get('bytes', 0)}"
            )
    return "\n".join(lines)
