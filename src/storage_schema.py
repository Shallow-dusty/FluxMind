"""No-secret local storage schema inventory.

This module checks shape only. It does not read or return stored row contents,
prompts, answers, filenames, owner IDs, request IDs, or source paths.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config import (
    ARTIFACTS_DIR,
    API_KEY_REGISTRY_FILE,
    CHUNK_METADATA_DB_FILE,
    CORPUS_METADATA_DB_FILE,
    CORPUS_METADATA_FILE,
    CORPUS_PROFILES_FILE,
    JOBS_DB_FILE,
    PRODUCT_REGISTRY_FILE,
    RUNTIME_EVENTS_FILE,
    SHARE_LINK_TOKEN_STORE_FILE,
)


STORAGE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class JsonStoreSpec:
    name: str
    path: Path
    required_keys: tuple[str, ...]
    expected_version: int = STORAGE_SCHEMA_VERSION
    required: bool = False


@dataclass(frozen=True)
class JsonlStoreSpec:
    name: str
    path: Path
    required_fields: tuple[str, ...]
    required: bool = False
    sample_limit: int = 25


@dataclass(frozen=True)
class SqliteTableSpec:
    name: str
    columns: tuple[str, ...]


@dataclass(frozen=True)
class SqliteStoreSpec:
    name: str
    path: Path
    tables: tuple[SqliteTableSpec, ...]
    required: bool = False


PAPER_COLUMNS = (
    "source_path",
    "paper_id",
    "filename",
    "source_kind",
    "checksum_sha256",
    "title",
    "authors",
    "year",
    "topic",
    "doi",
    "arxiv_id",
    "venue",
    "topic_tags",
    "source_url",
    "pdf_url",
    "license",
    "active",
    "indexed_status",
    "chunk_count",
    "parse_error",
    "index_error",
    "updated_at",
    "payload",
)

CHUNK_COLUMNS = (
    "chunk_id",
    "source_path",
    "source",
    "page",
    "chunk_index",
    "content_sha256",
    "char_count",
    "preview",
    "updated_at",
)

JOB_COLUMNS = (
    "job_id",
    "kind",
    "status",
    "created_at",
    "updated_at",
    "request_id",
    "attempts",
    "not_before",
    "deadline_at",
    "worker_id",
    "leased_at",
    "lease_expires_at",
    "idempotency_key",
    "max_attempts",
    "retry_backoff_s",
    "dead_lettered_at",
    "owner_id",
    "owner_label",
    "ownership_source",
    "payload",
)

JOB_IDEMPOTENCY_COLUMNS = ("kind", "idempotency_key", "job_id", "created_at")

ARTIFACT_COLUMNS = (
    "artifact_id",
    "job_id",
    "job_kind",
    "kind",
    "uri",
    "mime_type",
    "title",
    "metadata",
    "owner_id",
    "owner_label",
    "ownership_source",
    "payload",
)

API_KEY_COLUMNS = (
    "key_id",
    "token_hash",
    "owner_id",
    "owner_label",
    "description",
    "created_at",
    "revoked_at",
    "last_used_at",
    "use_count",
)

PRODUCT_USER_COLUMNS = ("user_id", "label", "status", "created_at", "updated_at")
PRODUCT_WORKSPACE_COLUMNS = (
    "workspace_id",
    "label",
    "owner_user_id",
    "status",
    "created_at",
    "updated_at",
)
PRODUCT_WORKSPACE_MEMBER_COLUMNS = (
    "workspace_id",
    "user_id",
    "role",
    "status",
    "created_at",
    "updated_at",
)
PRODUCT_QUOTA_COLUMNS = ("workspace_id", "metric", "limit_value", "window_s", "updated_at")
PRODUCT_USAGE_COLUMNS = (
    "event_id",
    "workspace_id",
    "user_id",
    "metric",
    "amount",
    "source",
    "created_at",
)
PRODUCT_BILLING_COLUMNS = (
    "workspace_id",
    "billing_mode",
    "status",
    "attribution_enabled",
    "updated_at",
)
SHARE_LINK_COLUMNS = (
    "link_id",
    "token_hash",
    "workspace_id",
    "created_by_user_id",
    "resource_kind",
    "resource_ref",
    "description",
    "created_at",
    "expires_at",
    "revoked_at",
    "redeemed_at",
    "redeem_count",
    "max_redemptions",
)

RUNTIME_EVENT_FIELDS = ("event_id", "kind", "code", "message", "created_at", "metadata")


def default_json_store_specs() -> tuple[JsonStoreSpec, ...]:
    return (
        JsonStoreSpec("corpus_metadata_json", CORPUS_METADATA_FILE, ("version", "papers")),
        JsonStoreSpec("corpus_profiles_json", CORPUS_PROFILES_FILE, ("version", "profiles")),
    )


def default_jsonl_store_specs() -> tuple[JsonlStoreSpec, ...]:
    return (
        JsonlStoreSpec("runtime_events_jsonl", RUNTIME_EVENTS_FILE, RUNTIME_EVENT_FIELDS),
    )


def default_sqlite_store_specs() -> tuple[SqliteStoreSpec, ...]:
    return (
        SqliteStoreSpec(
            "corpus_metadata_sqlite",
            CORPUS_METADATA_DB_FILE,
            (SqliteTableSpec("papers", PAPER_COLUMNS),),
        ),
        SqliteStoreSpec(
            "chunk_metadata_sqlite",
            CHUNK_METADATA_DB_FILE,
            (SqliteTableSpec("chunks", CHUNK_COLUMNS),),
        ),
        SqliteStoreSpec(
            "jobs_sqlite",
            JOBS_DB_FILE,
            (
                SqliteTableSpec("jobs", JOB_COLUMNS),
                SqliteTableSpec("job_idempotency", JOB_IDEMPOTENCY_COLUMNS),
            ),
        ),
        SqliteStoreSpec(
            "artifacts_sqlite",
            ARTIFACTS_DIR / "artifacts.sqlite3",
            (SqliteTableSpec("artifacts", ARTIFACT_COLUMNS),),
        ),
        SqliteStoreSpec(
            "api_key_registry_sqlite",
            API_KEY_REGISTRY_FILE,
            (SqliteTableSpec("api_keys", API_KEY_COLUMNS),),
        ),
        SqliteStoreSpec(
            "product_registry_sqlite",
            PRODUCT_REGISTRY_FILE,
            (
                SqliteTableSpec("product_users", PRODUCT_USER_COLUMNS),
                SqliteTableSpec("workspaces", PRODUCT_WORKSPACE_COLUMNS),
                SqliteTableSpec("workspace_members", PRODUCT_WORKSPACE_MEMBER_COLUMNS),
                SqliteTableSpec("quota_limits", PRODUCT_QUOTA_COLUMNS),
                SqliteTableSpec("usage_events", PRODUCT_USAGE_COLUMNS),
                SqliteTableSpec("billing_accounts", PRODUCT_BILLING_COLUMNS),
            ),
        ),
        SqliteStoreSpec(
            "share_link_registry_sqlite",
            SHARE_LINK_TOKEN_STORE_FILE,
            (SqliteTableSpec("share_links", SHARE_LINK_COLUMNS),),
        ),
    )


def _store_result(
    *,
    name: str,
    kind: str,
    required: bool,
    exists: bool,
    errors: list[str],
    **extra: Any,
) -> dict[str, Any]:
    return {
        "name": name,
        "kind": kind,
        "required": required,
        "exists": exists,
        "ok": not errors and (exists or not required),
        "errors": errors,
        **extra,
    }


def inspect_json_store(spec: JsonStoreSpec) -> dict[str, Any]:
    errors: list[str] = []
    if not spec.path.exists():
        if spec.required:
            errors.append("missing_required_store")
        return _store_result(
            name=spec.name,
            kind="json",
            required=spec.required,
            exists=False,
            errors=errors,
            expected_schema_version=spec.expected_version,
            schema_version=None,
            required_keys=list(spec.required_keys),
            missing_keys=[],
        )

    try:
        payload = json.loads(spec.path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = None
        errors.append("invalid_json")

    schema_version = payload.get("version") if isinstance(payload, dict) else None
    missing_keys = [
        key for key in spec.required_keys if not isinstance(payload, dict) or key not in payload
    ]
    if schema_version != spec.expected_version:
        errors.append("schema_version_mismatch")
    if missing_keys:
        errors.append("missing_required_keys")

    return _store_result(
        name=spec.name,
        kind="json",
        required=spec.required,
        exists=True,
        errors=errors,
        expected_schema_version=spec.expected_version,
        schema_version=schema_version,
        required_keys=list(spec.required_keys),
        missing_keys=missing_keys,
    )


def _read_jsonl_sample(path: Path, sample_limit: int) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line for line in lines if line.strip()][-sample_limit:]


def inspect_jsonl_store(spec: JsonlStoreSpec) -> dict[str, Any]:
    errors: list[str] = []
    if not spec.path.exists():
        if spec.required:
            errors.append("missing_required_store")
        return _store_result(
            name=spec.name,
            kind="jsonl",
            required=spec.required,
            exists=False,
            errors=errors,
            required_fields=list(spec.required_fields),
            sampled_events=0,
            invalid_lines=0,
            missing_fields=[],
        )

    invalid_lines = 0
    missing_fields: set[str] = set()
    sampled = 0
    try:
        lines = _read_jsonl_sample(spec.path, spec.sample_limit)
    except OSError:
        lines = []
        errors.append("unreadable_jsonl")

    for line in lines:
        sampled += 1
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            invalid_lines += 1
            continue
        if not isinstance(payload, dict):
            invalid_lines += 1
            continue
        for field in spec.required_fields:
            if field not in payload:
                missing_fields.add(field)

    if invalid_lines:
        errors.append("invalid_jsonl_lines")
    if missing_fields:
        errors.append("missing_required_fields")

    return _store_result(
        name=spec.name,
        kind="jsonl",
        required=spec.required,
        exists=True,
        errors=errors,
        required_fields=list(spec.required_fields),
        sampled_events=sampled,
        invalid_lines=invalid_lines,
        missing_fields=sorted(missing_fields),
    )


def _connect_readonly(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def inspect_sqlite_store(spec: SqliteStoreSpec) -> dict[str, Any]:
    errors: list[str] = []
    table_results: list[dict[str, Any]] = []
    if not spec.path.exists():
        if spec.required:
            errors.append("missing_required_store")
        return _store_result(
            name=spec.name,
            kind="sqlite",
            required=spec.required,
            exists=False,
            errors=errors,
            expected_table_count=len(spec.tables),
            table_count=0,
            missing_tables=[],
            tables=[],
        )

    try:
        with _connect_readonly(spec.path) as conn:
            existing_tables = {
                row["name"]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            }
            for table in spec.tables:
                if table.name not in existing_tables:
                    table_results.append(
                        {
                            "name": table.name,
                            "exists": False,
                            "column_count": 0,
                            "required_column_count": len(table.columns),
                            "missing_columns": list(table.columns),
                        }
                    )
                    continue
                columns = {
                    row["name"] for row in conn.execute(f"PRAGMA table_info({table.name})").fetchall()
                }
                missing_columns = [column for column in table.columns if column not in columns]
                table_results.append(
                    {
                        "name": table.name,
                        "exists": True,
                        "column_count": len(columns),
                        "required_column_count": len(table.columns),
                        "missing_columns": missing_columns,
                    }
                )
    except sqlite3.Error:
        errors.append("invalid_sqlite")

    missing_tables = [
        table["name"] for table in table_results if not table.get("exists")
    ]
    missing_columns_total = sum(len(table.get("missing_columns", [])) for table in table_results)
    if missing_tables:
        errors.append("missing_required_tables")
    if missing_columns_total:
        errors.append("missing_required_columns")

    return _store_result(
        name=spec.name,
        kind="sqlite",
        required=spec.required,
        exists=True,
        errors=errors,
        expected_table_count=len(spec.tables),
        table_count=len(table_results),
        missing_tables=missing_tables,
        tables=table_results,
    )


def storage_schema_status(
    *,
    json_stores: tuple[JsonStoreSpec, ...] | None = None,
    jsonl_stores: tuple[JsonlStoreSpec, ...] | None = None,
    sqlite_stores: tuple[SqliteStoreSpec, ...] | None = None,
) -> dict[str, Any]:
    """Return local storage schema readiness without exposing stored contents."""
    stores = [
        *(inspect_json_store(spec) for spec in (json_stores or default_json_store_specs())),
        *(inspect_jsonl_store(spec) for spec in (jsonl_stores or default_jsonl_store_specs())),
        *(inspect_sqlite_store(spec) for spec in (sqlite_stores or default_sqlite_store_specs())),
    ]
    problem_count = sum(len(store.get("errors", [])) for store in stores)
    return {
        "schema_version": STORAGE_SCHEMA_VERSION,
        "mode": "local_storage_schema_inventory",
        "ok": problem_count == 0 and all(store.get("ok") for store in stores),
        "store_count": len(stores),
        "problem_count": problem_count,
        "stores": stores,
    }


def storage_schema_status_for_root(project_root: Path) -> dict[str, Any]:
    """Return storage schema readiness for a project root using standard paths."""
    root = project_root.resolve()
    metadata_dir = root / "metadata"
    jobs_dir = root / "jobs"
    artifacts_dir = root / "artifacts"
    return storage_schema_status(
        json_stores=(
            JsonStoreSpec("corpus_metadata_json", metadata_dir / "corpus.json", ("version", "papers")),
            JsonStoreSpec(
                "corpus_profiles_json",
                metadata_dir / "corpus_profiles.json",
                ("version", "profiles"),
            ),
        ),
        jsonl_stores=(
            JsonlStoreSpec("runtime_events_jsonl", metadata_dir / "runtime_events.jsonl", RUNTIME_EVENT_FIELDS),
        ),
        sqlite_stores=(
            SqliteStoreSpec(
                "corpus_metadata_sqlite",
                metadata_dir / "corpus.sqlite3",
                (SqliteTableSpec("papers", PAPER_COLUMNS),),
            ),
            SqliteStoreSpec(
                "chunk_metadata_sqlite",
                metadata_dir / "chunks.sqlite3",
                (SqliteTableSpec("chunks", CHUNK_COLUMNS),),
            ),
            SqliteStoreSpec(
                "jobs_sqlite",
                jobs_dir / "jobs.sqlite3",
                (
                    SqliteTableSpec("jobs", JOB_COLUMNS),
                    SqliteTableSpec("job_idempotency", JOB_IDEMPOTENCY_COLUMNS),
                ),
            ),
            SqliteStoreSpec(
                "artifacts_sqlite",
                artifacts_dir / "artifacts.sqlite3",
                (SqliteTableSpec("artifacts", ARTIFACT_COLUMNS),),
            ),
            SqliteStoreSpec(
                "api_key_registry_sqlite",
                metadata_dir / "api_keys.sqlite3",
                (SqliteTableSpec("api_keys", API_KEY_COLUMNS),),
            ),
            SqliteStoreSpec(
                "product_registry_sqlite",
                metadata_dir / "product_registry.sqlite3",
                (
                    SqliteTableSpec("product_users", PRODUCT_USER_COLUMNS),
                    SqliteTableSpec("workspaces", PRODUCT_WORKSPACE_COLUMNS),
                    SqliteTableSpec("workspace_members", PRODUCT_WORKSPACE_MEMBER_COLUMNS),
                    SqliteTableSpec("quota_limits", PRODUCT_QUOTA_COLUMNS),
                    SqliteTableSpec("usage_events", PRODUCT_USAGE_COLUMNS),
                    SqliteTableSpec("billing_accounts", PRODUCT_BILLING_COLUMNS),
                ),
            ),
            SqliteStoreSpec(
                "share_link_registry_sqlite",
                metadata_dir / "share_links.sqlite3",
                (SqliteTableSpec("share_links", SHARE_LINK_COLUMNS),),
            ),
        ),
    )


def _format_bool(value: Any) -> str:
    return "true" if bool(value) else "false"


def format_storage_schema_markdown(status: dict[str, Any]) -> str:
    """Render a no-secret storage schema inventory as Markdown."""
    lines = [
        "# FluxMind Storage Schema",
        "",
        "No stored row contents, prompts, answers, filenames, owner IDs, request IDs, source paths, or runtime file contents are exported.",
        "",
        f"- Schema version: {status.get('schema_version', 0)}",
        f"- Mode: {status.get('mode', 'local_storage_schema_inventory')}",
        f"- OK: {_format_bool(status.get('ok', False))}",
        f"- Store count: {status.get('store_count', 0)}",
        f"- Problem count: {status.get('problem_count', 0)}",
        "",
        "## Stores",
        "",
    ]
    for store in status.get("stores", []):
        errors = ", ".join(store.get("errors", [])) or "none"
        lines.append(
            f"- {store.get('name', '')}: kind={store.get('kind', '')}, "
            f"exists={_format_bool(store.get('exists', False))}, "
            f"ok={_format_bool(store.get('ok', False))}, errors={errors}"
        )
        if store.get("kind") == "sqlite":
            for table in store.get("tables", []):
                lines.append(
                    f"  - table {table.get('name', '')}: "
                    f"exists={_format_bool(table.get('exists', False))}, "
                    f"columns={table.get('column_count', 0)}/"
                    f"{table.get('required_column_count', 0)}, "
                    f"missing={len(table.get('missing_columns', []))}"
                )
        elif store.get("kind") == "json":
            lines.append(
                f"  - version={store.get('schema_version')}, "
                f"expected={store.get('expected_schema_version')}, "
                f"missing_keys={len(store.get('missing_keys', []))}"
            )
        elif store.get("kind") == "jsonl":
            lines.append(
                f"  - sampled_events={store.get('sampled_events', 0)}, "
                f"invalid_lines={store.get('invalid_lines', 0)}, "
                f"missing_fields={len(store.get('missing_fields', []))}"
            )
    return "\n".join(lines)
