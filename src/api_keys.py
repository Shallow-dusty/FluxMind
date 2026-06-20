"""Local no-secret API key lifecycle registry."""

from __future__ import annotations

import hashlib
import secrets
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src import config
from src.jobs import normalize_ownership


API_KEY_REGISTRY_SCHEMA_VERSION = 1
API_KEY_TOKEN_PREFIX = "fmk_"
SUPPORTED_LOCAL_API_KEY_REGISTRY_BACKENDS = {"sqlite"}
DISABLED_API_KEY_REGISTRY_BACKENDS = {"", "none", "disabled", "local-disabled"}
SAFE_BACKEND_CHARS = set("abcdefghijklmnopqrstuvwxyz0123456789_.-")


@dataclass(frozen=True)
class ApiKeyRecord:
    key_id: str
    owner_id: str
    owner_label: str
    description: str
    created_at: str
    revoked_at: str | None
    last_used_at: str | None
    use_count: int

    @property
    def active(self) -> bool:
        return self.revoked_at is None

    def to_public_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("owner_id", None)
        payload.pop("owner_label", None)
        payload.pop("description", None)
        payload["active"] = self.active
        payload.update(
            {
                "owner_id_present": bool(self.owner_id),
                "owner_id_fingerprint": _safe_value_fingerprint(self.owner_id),
                "owner_label_present": bool(self.owner_label),
                "owner_label_fingerprint": _safe_value_fingerprint(self.owner_label),
                "description_present": bool(self.description),
                "description_fingerprint": _safe_value_fingerprint(self.description),
                "content_exported": False,
                "token_hash_exported": False,
                "owner_exported": False,
                "description_exported": False,
            }
        )
        return payload


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_backend_name(value: str | None) -> str:
    backend = (value or "none").strip().lower() or "none"
    if len(backend) > 64:
        return "custom"
    if any(char not in SAFE_BACKEND_CHARS for char in backend):
        return "custom"
    return backend


def _registry_db_path(db_path: Path | None = None) -> Path:
    if db_path is not None:
        return db_path
    return getattr(config, "API_KEY_REGISTRY_FILE", config.METADATA_DIR / "api_keys.sqlite3")


def api_key_token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _safe_value_fingerprint(value: str) -> str:
    if not value:
        return ""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def generate_api_key_token() -> str:
    return API_KEY_TOKEN_PREFIX + secrets.token_urlsafe(32)


def generate_key_id() -> str:
    return "key_" + secrets.token_urlsafe(12).replace("-", "").replace("_", "")[:16]


class LocalApiKeyRegistry:
    """SQLite-backed local API key registry.

    Only token hashes are persisted. The raw token is returned once from
    `create_key()` and is never included in list/status/verify outputs.
    """

    def __init__(self, db_path: Path | None = None):
        self.db_path = _registry_db_path(db_path)

    def ensure_schema(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS api_keys (
                    key_id TEXT PRIMARY KEY,
                    token_hash TEXT NOT NULL UNIQUE,
                    owner_id TEXT NOT NULL,
                    owner_label TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    revoked_at TEXT,
                    last_used_at TEXT,
                    use_count INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_api_keys_token_hash ON api_keys(token_hash)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_api_keys_owner_id ON api_keys(owner_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_api_keys_revoked_at ON api_keys(revoked_at)")

    def create_key(
        self,
        *,
        owner_id: str | None = None,
        owner_label: str | None = None,
        description: str | None = None,
    ) -> dict[str, Any]:
        self.ensure_schema()
        ownership = normalize_ownership(owner_id=owner_id, owner_label=owner_label)
        token = generate_api_key_token()
        record = ApiKeyRecord(
            key_id=generate_key_id(),
            owner_id=ownership["owner_id"],
            owner_label=ownership["owner_label"],
            description=(description or "").strip()[:256],
            created_at=_utc_now(),
            revoked_at=None,
            last_used_at=None,
            use_count=0,
        )
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO api_keys (
                    key_id, token_hash, owner_id, owner_label, description,
                    created_at, revoked_at, last_used_at, use_count
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.key_id,
                    api_key_token_hash(token),
                    record.owner_id,
                    record.owner_label,
                    record.description,
                    record.created_at,
                    record.revoked_at,
                    record.last_used_at,
                    record.use_count,
                ),
            )
        return {"token": token, "key": record.to_public_dict()}

    def list_keys(self, *, include_revoked: bool = False) -> list[ApiKeyRecord]:
        self.ensure_schema()
        sql = """
            SELECT key_id, owner_id, owner_label, description, created_at,
                   revoked_at, last_used_at, use_count
            FROM api_keys
        """
        if not include_revoked:
            sql += " WHERE revoked_at IS NULL"
        sql += " ORDER BY created_at DESC, key_id DESC"
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(sql).fetchall()
        return [
            ApiKeyRecord(
                key_id=row[0],
                owner_id=row[1],
                owner_label=row[2],
                description=row[3],
                created_at=row[4],
                revoked_at=row[5],
                last_used_at=row[6],
                use_count=int(row[7] or 0),
            )
            for row in rows
        ]

    def verify_token(self, token: str, *, update_usage: bool = False) -> ApiKeyRecord | None:
        if not token:
            return None
        self.ensure_schema()
        token_hash = api_key_token_hash(token)
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT key_id, owner_id, owner_label, description, created_at,
                       revoked_at, last_used_at, use_count
                FROM api_keys
                WHERE token_hash = ? AND revoked_at IS NULL
                """,
                (token_hash,),
            ).fetchone()
            if row is None:
                return None
            last_used_at = row[6]
            use_count = int(row[7] or 0)
            if update_usage:
                last_used_at = _utc_now()
                use_count += 1
                conn.execute(
                    """
                    UPDATE api_keys
                    SET last_used_at = ?, use_count = ?
                    WHERE key_id = ?
                    """,
                    (last_used_at, use_count, row[0]),
                )
        return ApiKeyRecord(
            key_id=row[0],
            owner_id=row[1],
            owner_label=row[2],
            description=row[3],
            created_at=row[4],
            revoked_at=row[5],
            last_used_at=last_used_at,
            use_count=use_count,
        )

    def revoke_key(self, key_id: str) -> ApiKeyRecord | None:
        self.ensure_schema()
        key_id = key_id.strip()
        revoked_at = _utc_now()
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT key_id, owner_id, owner_label, description, created_at,
                       revoked_at, last_used_at, use_count
                FROM api_keys
                WHERE key_id = ?
                """,
                (key_id,),
            ).fetchone()
            if row is None:
                return None
            final_revoked_at = row[5] or revoked_at
            if row[5] is None:
                conn.execute(
                    "UPDATE api_keys SET revoked_at = ? WHERE key_id = ?",
                    (final_revoked_at, key_id),
                )
        return ApiKeyRecord(
            key_id=row[0],
            owner_id=row[1],
            owner_label=row[2],
            description=row[3],
            created_at=row[4],
            revoked_at=final_revoked_at,
            last_used_at=row[6],
            use_count=int(row[7] or 0),
        )

    def status(self) -> dict[str, Any]:
        self.ensure_schema()
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM api_keys").fetchone()[0]
            active = conn.execute(
                "SELECT COUNT(*) FROM api_keys WHERE revoked_at IS NULL"
            ).fetchone()[0]
        return {
            "schema_version": API_KEY_REGISTRY_SCHEMA_VERSION,
            "backend": "sqlite",
            "available": True,
            "active_key_count": int(active or 0),
            "revoked_key_count": int((total or 0) - (active or 0)),
            "total_key_count": int(total or 0),
            "content_exported": False,
            "secrets_exported": False,
        }


def api_key_registry_backend_status(
    *,
    backend: str | None = None,
    db_path: Path | None = None,
) -> dict[str, Any]:
    safe_backend = _safe_backend_name(backend if backend is not None else config.API_KEY_REGISTRY_BACKEND)
    configured = safe_backend not in DISABLED_API_KEY_REGISTRY_BACKENDS
    supported = safe_backend in SUPPORTED_LOCAL_API_KEY_REGISTRY_BACKENDS
    if not configured:
        return {
            "schema_version": API_KEY_REGISTRY_SCHEMA_VERSION,
            "backend": safe_backend,
            "configured": False,
            "supported": False,
            "available": False,
            "reason": "api_key_registry_not_configured",
            "active_key_count": 0,
            "revoked_key_count": 0,
            "total_key_count": 0,
            "content_exported": False,
            "secrets_exported": False,
            "paths_exported": False,
        }
    if not supported:
        return {
            "schema_version": API_KEY_REGISTRY_SCHEMA_VERSION,
            "backend": safe_backend,
            "configured": True,
            "supported": False,
            "available": False,
            "reason": "api_key_registry_backend_not_local",
            "active_key_count": 0,
            "revoked_key_count": 0,
            "total_key_count": 0,
            "content_exported": False,
            "secrets_exported": False,
            "paths_exported": False,
        }
    try:
        status = LocalApiKeyRegistry(db_path=db_path).status()
    except (OSError, sqlite3.Error):
        return {
            "schema_version": API_KEY_REGISTRY_SCHEMA_VERSION,
            "backend": safe_backend,
            "configured": True,
            "supported": True,
            "available": False,
            "reason": "api_key_registry_unavailable",
            "active_key_count": 0,
            "revoked_key_count": 0,
            "total_key_count": 0,
            "content_exported": False,
            "secrets_exported": False,
            "paths_exported": False,
        }
    return {
        **status,
        "configured": True,
        "supported": True,
        "reason": "available",
        "paths_exported": False,
    }


def verify_configured_api_key_token(
    token: str,
    *,
    update_usage: bool = False,
) -> ApiKeyRecord | None:
    status = api_key_registry_backend_status()
    if not status.get("available") or status.get("backend") != "sqlite":
        return None
    return LocalApiKeyRegistry().verify_token(token, update_usage=update_usage)
