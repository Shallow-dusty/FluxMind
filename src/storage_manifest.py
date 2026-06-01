"""No-secret runtime backup manifest helpers."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import (
    ACTIVE_PAPERS_FILE,
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
