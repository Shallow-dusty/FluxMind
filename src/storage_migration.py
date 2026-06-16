"""Local runtime migration rehearsal helpers.

The rehearsal copies runtime state into a staging root, then verifies that the
staged tree matches a no-secret manifest and still satisfies local storage
schema checks. It is a local backup/restore drill, not external database or
object-storage activation.
"""

from __future__ import annotations

import hashlib
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src import config
from src.platform_migration import (
    collect_platform_migration_preflight,
    runtime_groups_for_root,
)
from src.storage_manifest import (
    collect_runtime_backup_manifest,
    collect_runtime_restore_check,
)
from src.storage_schema import storage_schema_status_for_root


STORAGE_MIGRATION_REHEARSAL_SCHEMA_VERSION = 1
OBJECT_STORAGE_MIGRATION_MANIFEST_SCHEMA_VERSION = 1
DEFAULT_OBJECT_KEY_PREFIX = "fluxmind-runtime"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _empty_directory(path: Path) -> None:
    if not path.exists():
        path.mkdir(parents=True)
        return
    for item in path.iterdir():
        if item.is_dir() and not item.is_symlink():
            shutil.rmtree(item)
        else:
            item.unlink()


def _directory_has_entries(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sanitize_object_key_prefix(value: str) -> str:
    prefix = (value or DEFAULT_OBJECT_KEY_PREFIX).strip().strip("/")
    if "://" in prefix:
        prefix = DEFAULT_OBJECT_KEY_PREFIX
    prefix = re.sub(r"[^A-Za-z0-9._/-]+", "-", prefix).strip("/")
    return prefix or DEFAULT_OBJECT_KEY_PREFIX


def _path_token(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _copy_runtime_group(
    *,
    name: str,
    source_path: Path,
    target_path: Path,
    restore_priority: str,
    include_runtime_dependencies: bool,
) -> dict[str, Any]:
    if restore_priority == "runtime_dependency" and not include_runtime_dependencies:
        return {
            "name": name,
            "restore_priority": restore_priority,
            "source_exists": source_path.exists(),
            "status": "skipped_runtime_dependency",
            "copied_files": 0,
            "copied_bytes": 0,
            "skipped_symlinks": 0,
            "errors": [],
        }
    if not source_path.exists():
        return {
            "name": name,
            "restore_priority": restore_priority,
            "source_exists": False,
            "status": "source_absent",
            "copied_files": 0,
            "copied_bytes": 0,
            "skipped_symlinks": 0,
            "errors": [],
        }
    if not source_path.is_dir():
        return {
            "name": name,
            "restore_priority": restore_priority,
            "source_exists": True,
            "status": "source_not_directory",
            "copied_files": 0,
            "copied_bytes": 0,
            "skipped_symlinks": 0,
            "errors": ["source_not_directory"],
        }

    copied_files = 0
    copied_bytes = 0
    skipped_symlinks = 0
    errors: list[str] = []
    target_path.mkdir(parents=True, exist_ok=True)
    for item in source_path.rglob("*"):
        relative = item.relative_to(source_path)
        target = target_path / relative
        if item.is_symlink():
            skipped_symlinks += 1
            continue
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        if not item.is_file():
            continue
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)
            copied_files += 1
            copied_bytes += item.stat().st_size
        except OSError:
            errors.append("copy_failed")
            break

    return {
        "name": name,
        "restore_priority": restore_priority,
        "source_exists": True,
        "status": "copied" if not errors else "copy_failed",
        "copied_files": copied_files,
        "copied_bytes": copied_bytes,
        "skipped_symlinks": skipped_symlinks,
        "errors": errors,
    }


def _manifest_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": manifest.get("schema_version"),
        "mode": manifest.get("mode"),
        "content_exported": bool(manifest.get("content_exported")),
        "secrets_exported": bool(manifest.get("secrets_exported")),
        "env_file_present": bool(manifest.get("env_file_present")),
        "env_file_content_exported": bool(manifest.get("env_file_content_exported")),
        "total_files": int(manifest.get("total_files", 0) or 0),
        "total_bytes": int(manifest.get("total_bytes", 0) or 0),
        "group_count": len(manifest.get("groups", [])),
    }


def _restore_summary(check: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": check.get("schema_version"),
        "mode": check.get("mode"),
        "ok": bool(check.get("ok")),
        "content_restored": bool(check.get("content_restored")),
        "delete_enabled": bool(check.get("delete_enabled")),
        "manifest_errors": list(check.get("manifest_errors", [])),
        "checked_groups": int(check.get("checked_groups", 0) or 0),
        "checked_files": int(check.get("checked_files", 0) or 0),
        "missing_groups": int(check.get("missing_groups", 0) or 0),
        "mismatched_groups": int(check.get("mismatched_groups", 0) or 0),
        "missing_files": int(check.get("missing_files", 0) or 0),
        "mismatched_files": int(check.get("mismatched_files", 0) or 0),
    }


def _schema_summary(status: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": status.get("schema_version"),
        "mode": status.get("mode"),
        "ok": bool(status.get("ok")),
        "store_count": int(status.get("store_count", 0) or 0),
        "problem_count": int(status.get("problem_count", 0) or 0),
        "stores": [
            {
                "name": store.get("name", ""),
                "kind": store.get("kind", ""),
                "exists": bool(store.get("exists")),
                "ok": bool(store.get("ok")),
                "errors": list(store.get("errors", [])),
            }
            for store in status.get("stores", [])
        ],
    }


def collect_object_storage_migration_manifest(
    *,
    project_root: Path = config.PROJECT_ROOT,
    groups: tuple[Any, ...] | None = None,
    include_runtime_dependencies: bool = False,
    key_prefix: str = DEFAULT_OBJECT_KEY_PREFIX,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Return an opaque object manifest for copying runtime files to object storage.

    The manifest deliberately omits source paths, filenames, buckets, endpoints,
    and credentials. It gives an operator enough hash/key/count evidence to
    validate a future upload plan without exposing runtime contents in reports.
    """
    root = project_root.resolve()
    runtime_groups = groups if groups is not None else runtime_groups_for_root(root)
    if not include_runtime_dependencies:
        runtime_groups = tuple(
            group for group in runtime_groups if group.restore_priority != "runtime_dependency"
        )
    prefix = _sanitize_object_key_prefix(key_prefix)
    objects: list[dict[str, Any]] = []
    group_summaries: list[dict[str, Any]] = []
    unique_keys: set[str] = set()

    for group in runtime_groups:
        group_path = group.path.resolve()
        group_objects = 0
        group_bytes = 0
        group_exists = group_path.exists()
        group_is_dir = group_path.is_dir() and not group_path.is_symlink()
        if group_is_dir:
            for item in sorted(group_path.rglob("*"), key=lambda path: path.as_posix()):
                if item.is_symlink() or not item.is_file():
                    continue
                relative_path = item.relative_to(group_path).as_posix()
                digest = _sha256_file(item)
                size = item.stat().st_size
                object_key = f"{prefix}/{group.name}/{digest[:2]}/{digest}"
                unique_keys.add(object_key)
                group_objects += 1
                group_bytes += size
                objects.append(
                    {
                        "group": group.name,
                        "restore_priority": group.restore_priority,
                        "object_key": object_key,
                        "bytes": size,
                        "sha256": digest,
                        "source_path_token": _path_token(f"{group.name}/{relative_path}"),
                    }
                )

        group_summaries.append(
            {
                "name": group.name,
                "restore_priority": group.restore_priority,
                "source_exists": group_exists,
                "source_is_directory": group_is_dir,
                "object_count": group_objects,
                "bytes": group_bytes,
            }
        )

    return {
        "schema_version": OBJECT_STORAGE_MIGRATION_MANIFEST_SCHEMA_VERSION,
        "generated_at": generated_at or _utc_now(),
        "mode": "object_storage_migration_manifest",
        "content_exported": False,
        "secrets_exported": False,
        "source_paths_exported": False,
        "filenames_exported": False,
        "bucket_exported": False,
        "external_connectivity_checked": False,
        "hash_algorithm": "sha256",
        "object_key_strategy": "grouped_by_content_sha256",
        "key_prefix": prefix,
        "group_count": len(group_summaries),
        "object_count": len(objects),
        "unique_object_count": len(unique_keys),
        "duplicate_content_references": len(objects) - len(unique_keys),
        "total_bytes": sum(int(item.get("bytes", 0) or 0) for item in objects),
        "groups": group_summaries,
        "objects": objects,
    }


def _object_manifest_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": manifest.get("schema_version"),
        "mode": manifest.get("mode", ""),
        "content_exported": bool(manifest.get("content_exported")),
        "secrets_exported": bool(manifest.get("secrets_exported")),
        "source_paths_exported": bool(manifest.get("source_paths_exported")),
        "filenames_exported": bool(manifest.get("filenames_exported")),
        "bucket_exported": bool(manifest.get("bucket_exported")),
        "external_connectivity_checked": bool(manifest.get("external_connectivity_checked")),
        "hash_algorithm": manifest.get("hash_algorithm", ""),
        "object_key_strategy": manifest.get("object_key_strategy", ""),
        "group_count": int(manifest.get("group_count", 0) or 0),
        "object_count": int(manifest.get("object_count", 0) or 0),
        "unique_object_count": int(manifest.get("unique_object_count", 0) or 0),
        "duplicate_content_references": int(
            manifest.get("duplicate_content_references", 0) or 0
        ),
        "total_bytes": int(manifest.get("total_bytes", 0) or 0),
    }


def run_storage_migration_rehearsal(
    *,
    project_root: Path = config.PROJECT_ROOT,
    staging_root: Path,
    overwrite_staging: bool = False,
    include_runtime_dependencies: bool = False,
    include_object_manifest: bool = False,
    object_key_prefix: str = DEFAULT_OBJECT_KEY_PREFIX,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Copy local runtime state into staging and verify it with no-secret checks."""
    source_root = project_root.resolve()
    target_root = staging_root.resolve()
    blockers: list[str] = []
    copy_groups: list[dict[str, Any]] = []
    staging_prepared = False
    staging_created = False

    if source_root == target_root or _is_relative_to(target_root, source_root):
        blockers.append("staging_root_inside_project")
    elif _directory_has_entries(target_root) and not overwrite_staging:
        blockers.append("staging_root_not_empty")
    else:
        if target_root.exists() and overwrite_staging:
            _empty_directory(target_root)
        elif not target_root.exists():
            target_root.mkdir(parents=True)
            staging_created = True
        staging_prepared = True

    runtime_groups = runtime_groups_for_root(source_root)
    if not include_runtime_dependencies:
        runtime_groups = tuple(
            group for group in runtime_groups if group.restore_priority != "runtime_dependency"
        )
    source_manifest = collect_runtime_backup_manifest(
        project_root=source_root,
        groups=runtime_groups,
        generated_at=generated_at,
    )
    source_preflight = collect_platform_migration_preflight(
        project_root=source_root,
        generated_at=generated_at,
    )

    if staging_prepared:
        for group in source_manifest.get("groups", []):
            relative_group = Path(str(group.get("path", "")))
            copy_groups.append(
                _copy_runtime_group(
                    name=str(group.get("name", "")),
                    source_path=source_root / relative_group,
                    target_path=target_root / relative_group,
                    restore_priority=str(group.get("restore_priority", "")),
                    include_runtime_dependencies=include_runtime_dependencies,
                )
            )

    copy_errors = [
        error
        for group in copy_groups
        for error in group.get("errors", [])
    ]
    blockers.extend(copy_errors)
    restore_check = collect_runtime_restore_check(
        source_manifest,
        project_root=target_root,
        generated_at=generated_at,
    ) if staging_prepared else {}
    staged_schema = storage_schema_status_for_root(target_root) if staging_prepared else {}
    object_manifest = (
        collect_object_storage_migration_manifest(
            project_root=target_root,
            include_runtime_dependencies=include_runtime_dependencies,
            key_prefix=object_key_prefix,
            generated_at=generated_at,
        )
        if staging_prepared and include_object_manifest
        else {}
    )
    if source_preflight.get("preflight_ok") is not True:
        blockers.append("source_preflight_failed")
    if staging_prepared and not restore_check.get("ok"):
        blockers.append("staged_restore_check_failed")
    if staging_prepared and not staged_schema.get("ok"):
        blockers.append("staged_storage_schema_drift")

    copied_files = sum(int(group.get("copied_files", 0) or 0) for group in copy_groups)
    copied_bytes = sum(int(group.get("copied_bytes", 0) or 0) for group in copy_groups)
    skipped_symlinks = sum(int(group.get("skipped_symlinks", 0) or 0) for group in copy_groups)
    rehearsal_ok = not blockers and staging_prepared
    return {
        "schema_version": STORAGE_MIGRATION_REHEARSAL_SCHEMA_VERSION,
        "generated_at": generated_at or _utc_now(),
        "mode": "local_runtime_migration_rehearsal",
        "rehearsal_ok": rehearsal_ok,
        "activation_enabled": False,
        "external_connectivity_checked": False,
        "content_copied_to_staging": bool(staging_prepared),
        "content_exported_in_report": False,
        "secrets_copied": False,
        "secrets_exported": False,
        "staging_root_created": staging_created,
        "staging_root_overwritten": bool(overwrite_staging and staging_prepared),
        "staging_root_retained": True,
        "include_runtime_dependencies": include_runtime_dependencies,
        "blockers": sorted(set(blockers)),
        "summary": {
            "source_preflight_ok": bool(source_preflight.get("preflight_ok")),
            "source_activation_ready": bool(source_preflight.get("activation_ready")),
            "copy_group_count": len(copy_groups),
            "copied_files": copied_files,
            "copied_bytes": copied_bytes,
            "skipped_symlinks": skipped_symlinks,
            "restore_check_ok": bool(restore_check.get("ok")),
            "staged_storage_schema_ok": bool(staged_schema.get("ok")),
            "object_manifest_ready": bool(object_manifest.get("mode")),
            "object_manifest_objects": int(object_manifest.get("object_count", 0) or 0),
            "object_manifest_unique_objects": int(
                object_manifest.get("unique_object_count", 0) or 0
            ),
        },
        "source_preflight": {
            "preflight_ok": bool(source_preflight.get("preflight_ok")),
            "activation_ready": bool(source_preflight.get("activation_ready")),
            "local_blockers": list(source_preflight.get("blockers", {}).get("local_preflight", [])),
            "activation_blockers": list(source_preflight.get("blockers", {}).get("activation", [])),
        },
        "source_manifest": _manifest_summary(source_manifest),
        "copy": {
            "groups": copy_groups,
            "copied_files": copied_files,
            "copied_bytes": copied_bytes,
            "skipped_symlinks": skipped_symlinks,
        },
        "staged_restore_check": _restore_summary(restore_check),
        "staged_storage_schema": _schema_summary(staged_schema),
        "object_storage_manifest": object_manifest,
        "object_storage_manifest_summary": _object_manifest_summary(object_manifest)
        if object_manifest
        else {},
    }


def _format_bool(value: Any) -> str:
    return "true" if bool(value) else "false"


def format_storage_migration_rehearsal_markdown(status: dict[str, Any]) -> str:
    """Render a no-secret local migration rehearsal report."""
    summary = status.get("summary", {})
    object_summary = status.get("object_storage_manifest_summary", {})
    lines = [
        "# FluxMind Runtime Migration Rehearsal",
        "",
        "No runtime contents, job payloads, external URLs, bucket names, queue names, credentials, or secrets are exported in this report.",
        "",
        f"- Generated at: {status.get('generated_at', '')}",
        f"- Mode: {status.get('mode', '')}",
        f"- Rehearsal OK: {_format_bool(status.get('rehearsal_ok', False))}",
        f"- Activation enabled: {_format_bool(status.get('activation_enabled', False))}",
        f"- External connectivity checked: {_format_bool(status.get('external_connectivity_checked', False))}",
        f"- Content copied to staging: {_format_bool(status.get('content_copied_to_staging', False))}",
        f"- Content exported in report: {_format_bool(status.get('content_exported_in_report', False))}",
        f"- Secrets copied: {_format_bool(status.get('secrets_copied', False))}",
        f"- Secrets exported: {_format_bool(status.get('secrets_exported', False))}",
        f"- Include runtime dependencies: {_format_bool(status.get('include_runtime_dependencies', False))}",
        "",
        "## Summary",
        "",
        f"- Source preflight OK: {_format_bool(summary.get('source_preflight_ok', False))}",
        f"- Source activation ready: {_format_bool(summary.get('source_activation_ready', False))}",
        f"- Copy groups: {summary.get('copy_group_count', 0)}",
        f"- Copied files: {summary.get('copied_files', 0)}",
        f"- Copied bytes: {summary.get('copied_bytes', 0)}",
        f"- Skipped symlinks: {summary.get('skipped_symlinks', 0)}",
        f"- Restore check OK: {_format_bool(summary.get('restore_check_ok', False))}",
        f"- Staged storage schema OK: {_format_bool(summary.get('staged_storage_schema_ok', False))}",
        f"- Object manifest ready: {_format_bool(summary.get('object_manifest_ready', False))}",
        f"- Object manifest objects: {summary.get('object_manifest_objects', 0)}",
        f"- Object manifest unique objects: {summary.get('object_manifest_unique_objects', 0)}",
        "",
        "## Blockers",
        "",
        f"- {', '.join(status.get('blockers', [])) or 'none'}",
        "",
        "## Copied Groups",
        "",
    ]
    for group in status.get("copy", {}).get("groups", []):
        lines.append(
            f"- {group.get('name', '')}: status={group.get('status', '')}, "
            f"priority={group.get('restore_priority', '')}, "
            f"source_exists={_format_bool(group.get('source_exists', False))}, "
            f"files={group.get('copied_files', 0)}, "
            f"bytes={group.get('copied_bytes', 0)}, "
            f"skipped_symlinks={group.get('skipped_symlinks', 0)}, "
            f"errors={','.join(group.get('errors', [])) or 'none'}"
        )
    if object_summary:
        lines.extend(
            [
                "",
                "## Object Storage Migration Manifest",
                "",
                "Object keys, hashes, and byte counts are available in JSON output. Source paths, filenames, buckets, endpoints, credentials, and contents are not exported.",
                "",
                f"- Mode: {object_summary.get('mode', '')}",
                f"- Content exported: {_format_bool(object_summary.get('content_exported', False))}",
                f"- Secrets exported: {_format_bool(object_summary.get('secrets_exported', False))}",
                f"- Source paths exported: {_format_bool(object_summary.get('source_paths_exported', False))}",
                f"- Filenames exported: {_format_bool(object_summary.get('filenames_exported', False))}",
                f"- Bucket exported: {_format_bool(object_summary.get('bucket_exported', False))}",
                f"- Object count: {object_summary.get('object_count', 0)}",
                f"- Unique object count: {object_summary.get('unique_object_count', 0)}",
                f"- Duplicate content references: {object_summary.get('duplicate_content_references', 0)}",
                f"- Total bytes: {object_summary.get('total_bytes', 0)}",
            ]
        )
    return "\n".join(lines)


def format_object_storage_migration_manifest_markdown(manifest: dict[str, Any]) -> str:
    """Render an object-storage migration manifest summary as no-secret Markdown."""
    lines = [
        "# FluxMind Object Storage Migration Manifest",
        "",
        "No runtime contents, source paths, filenames, buckets, endpoints, credentials, or secrets are exported.",
        "",
        f"- Generated at: {manifest.get('generated_at', '')}",
        f"- Mode: {manifest.get('mode', '')}",
        f"- Content exported: {_format_bool(manifest.get('content_exported', False))}",
        f"- Secrets exported: {_format_bool(manifest.get('secrets_exported', False))}",
        f"- Source paths exported: {_format_bool(manifest.get('source_paths_exported', False))}",
        f"- Filenames exported: {_format_bool(manifest.get('filenames_exported', False))}",
        f"- Bucket exported: {_format_bool(manifest.get('bucket_exported', False))}",
        f"- External connectivity checked: {_format_bool(manifest.get('external_connectivity_checked', False))}",
        f"- Hash algorithm: {manifest.get('hash_algorithm', '')}",
        f"- Object key strategy: {manifest.get('object_key_strategy', '')}",
        f"- Object count: {manifest.get('object_count', 0)}",
        f"- Unique object count: {manifest.get('unique_object_count', 0)}",
        f"- Duplicate content references: {manifest.get('duplicate_content_references', 0)}",
        f"- Total bytes: {manifest.get('total_bytes', 0)}",
        "",
        "## Runtime Groups",
        "",
    ]
    for group in manifest.get("groups", []):
        lines.append(
            f"- {group.get('name', '')}: priority={group.get('restore_priority', '')}, "
            f"source_exists={_format_bool(group.get('source_exists', False))}, "
            f"objects={group.get('object_count', 0)}, bytes={group.get('bytes', 0)}"
        )
    return "\n".join(lines)
