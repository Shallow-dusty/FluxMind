"""Local hash-only share-link token registry."""

from __future__ import annotations

import hashlib
import secrets
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from src import config
from src.product_registry import SAFE_BACKEND_CHARS, SAFE_ID_CHARS


SHARE_LINK_REGISTRY_SCHEMA_VERSION = 1
SHARE_LINK_TOKEN_PREFIX = "fms_"
SUPPORTED_LOCAL_SHARE_LINK_BACKENDS = {"sqlite"}
DISABLED_SHARE_LINK_BACKENDS = {"", "none", "disabled", "local-disabled"}
SAFE_RESOURCE_KINDS = {"corpus_profile", "paper", "artifact", "job", "report"}


@dataclass(frozen=True)
class ShareLinkRecord:
    link_id: str
    workspace_id: str
    created_by_user_id: str
    resource_kind: str
    resource_ref: str
    description: str
    created_at: str
    expires_at: str
    revoked_at: str | None
    redeemed_at: str | None
    redeem_count: int
    max_redemptions: int

    @property
    def active(self) -> bool:
        return self.revoked_at is None and not self.expired

    @property
    def expired(self) -> bool:
        if not self.expires_at:
            return False
        try:
            expires = datetime.fromisoformat(self.expires_at)
        except ValueError:
            return True
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        return expires <= datetime.now(timezone.utc)

    @property
    def exhausted(self) -> bool:
        return self.max_redemptions > 0 and self.redeem_count >= self.max_redemptions

    def to_public_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("resource_ref", None)
        payload.pop("description", None)
        payload.pop("created_by_user_id", None)
        payload.update(
            {
                "active": self.active and not self.exhausted,
                "expired": self.expired,
                "exhausted": self.exhausted,
                "created_by_user_present": bool(self.created_by_user_id),
                "created_by_user_fingerprint": _safe_value_fingerprint(
                    self.created_by_user_id
                ),
                "description_present": bool(self.description),
                "description_fingerprint": _safe_value_fingerprint(self.description),
                "resource_ref_present": bool(self.resource_ref),
                "resource_ref_fingerprint": share_link_resource_fingerprint(
                    self.resource_kind,
                    self.resource_ref,
                ),
                "content_exported": False,
                "secrets_exported": False,
                "share_token_exported": False,
                "share_url_exported": False,
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
    return getattr(
        config,
        "SHARE_LINK_TOKEN_STORE_FILE",
        config.METADATA_DIR / "share_links.sqlite3",
    )


def _safe_identifier(value: str | None, *, prefix: str) -> str:
    candidate = (value or "").strip()
    if not candidate:
        return _generated_id(prefix)
    candidate = candidate[:128]
    if any(char not in SAFE_ID_CHARS for char in candidate):
        return _generated_id(prefix)
    return candidate


def _safe_resource_ref(value: str | None) -> str:
    ref = (value or "").strip()
    if not ref:
        return ""
    return ref[:512]


def _safe_label(value: str | None) -> str:
    return (value or "").strip()[:256]


def _generated_id(prefix: str) -> str:
    return prefix + "_" + secrets.token_urlsafe(12).replace("-", "").replace("_", "")[:16]


def _safe_resource_kind(value: str | None) -> str:
    kind = (value or "corpus_profile").strip().lower().replace("-", "_")
    if kind not in SAFE_RESOURCE_KINDS:
        return "corpus_profile"
    return kind


def _expiry_from_ttl(expires_in_s: int | None) -> str:
    ttl = 7 * 24 * 60 * 60 if expires_in_s is None else max(60, min(int(expires_in_s), 31_536_000))
    return (datetime.now(timezone.utc) + timedelta(seconds=ttl)).isoformat()


def share_link_token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def share_link_resource_fingerprint(resource_kind: str, resource_ref: str) -> str:
    if not resource_ref:
        return ""
    digest = hashlib.sha256(f"{resource_kind}:{resource_ref}".encode("utf-8")).hexdigest()
    return digest[:16]


def _safe_value_fingerprint(value: str) -> str:
    if not value:
        return ""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def generate_share_link_token() -> str:
    return SHARE_LINK_TOKEN_PREFIX + secrets.token_urlsafe(32)


def generate_share_link_id() -> str:
    return _generated_id("share")


class LocalShareLinkRegistry:
    """SQLite-backed local share-link token store.

    Tokens are persisted only as SHA-256 hashes. Public projections omit the
    raw token, URL, resource reference, paths, prompts, answers, and content.
    """

    def __init__(self, db_path: Path | None = None):
        self.db_path = _registry_db_path(db_path)

    def ensure_schema(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS share_links (
                    link_id TEXT PRIMARY KEY,
                    token_hash TEXT NOT NULL UNIQUE,
                    workspace_id TEXT NOT NULL,
                    created_by_user_id TEXT NOT NULL,
                    resource_kind TEXT NOT NULL,
                    resource_ref TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    revoked_at TEXT,
                    redeemed_at TEXT,
                    redeem_count INTEGER NOT NULL DEFAULT 0,
                    max_redemptions INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_share_links_token_hash ON share_links(token_hash)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_share_links_workspace ON share_links(workspace_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_share_links_resource ON share_links(resource_kind, resource_ref)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_share_links_revoked ON share_links(revoked_at)")

    def create_link(
        self,
        *,
        workspace_id: str,
        created_by_user_id: str,
        resource_kind: str = "corpus_profile",
        resource_ref: str,
        description: str | None = None,
        expires_in_s: int | None = None,
        max_redemptions: int = 0,
    ) -> dict[str, Any]:
        self.ensure_schema()
        safe_workspace_id = _safe_identifier(workspace_id, prefix="ws")
        safe_user_id = _safe_identifier(created_by_user_id, prefix="user")
        safe_kind = _safe_resource_kind(resource_kind)
        safe_ref = _safe_resource_ref(resource_ref)
        if not safe_ref:
            raise ValueError("resource_ref_required")
        token = generate_share_link_token()
        record = ShareLinkRecord(
            link_id=generate_share_link_id(),
            workspace_id=safe_workspace_id,
            created_by_user_id=safe_user_id,
            resource_kind=safe_kind,
            resource_ref=safe_ref,
            description=_safe_label(description),
            created_at=_utc_now(),
            expires_at=_expiry_from_ttl(expires_in_s),
            revoked_at=None,
            redeemed_at=None,
            redeem_count=0,
            max_redemptions=max(0, int(max_redemptions)),
        )
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO share_links (
                    link_id, token_hash, workspace_id, created_by_user_id,
                    resource_kind, resource_ref, description, created_at,
                    expires_at, revoked_at, redeemed_at, redeem_count,
                    max_redemptions
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.link_id,
                    share_link_token_hash(token),
                    record.workspace_id,
                    record.created_by_user_id,
                    record.resource_kind,
                    record.resource_ref,
                    record.description,
                    record.created_at,
                    record.expires_at,
                    record.revoked_at,
                    record.redeemed_at,
                    record.redeem_count,
                    record.max_redemptions,
                ),
            )
        return {"token": token, "share_link": record.to_public_dict()}

    def list_links(
        self,
        *,
        workspace_id: str | None = None,
        include_revoked: bool = False,
        limit: int = 50,
    ) -> list[ShareLinkRecord]:
        self.ensure_schema()
        clauses: list[str] = []
        params: list[Any] = []
        if workspace_id:
            clauses.append("workspace_id = ?")
            params.append(_safe_identifier(workspace_id, prefix="ws"))
        if not include_revoked:
            clauses.append("revoked_at IS NULL")
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        limit_int = min(200, max(1, int(limit)))
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                f"""
                SELECT link_id, workspace_id, created_by_user_id, resource_kind,
                       resource_ref, description, created_at, expires_at,
                       revoked_at, redeemed_at, redeem_count, max_redemptions
                FROM share_links
                {where}
                ORDER BY created_at DESC, link_id DESC
                LIMIT ?
                """,
                (*params, limit_int),
            ).fetchall()
        return [_record_from_row(row) for row in rows]

    def get_link(self, link_id: str) -> ShareLinkRecord | None:
        self.ensure_schema()
        safe_link_id = _safe_identifier(link_id, prefix="share")
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT link_id, workspace_id, created_by_user_id, resource_kind,
                       resource_ref, description, created_at, expires_at,
                       revoked_at, redeemed_at, redeem_count, max_redemptions
                FROM share_links
                WHERE link_id = ?
                """,
                (safe_link_id,),
            ).fetchone()
        return _record_from_row(row) if row is not None else None

    def revoke_link(self, link_id: str) -> ShareLinkRecord | None:
        self.ensure_schema()
        safe_link_id = _safe_identifier(link_id, prefix="share")
        now = _utc_now()
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT link_id, workspace_id, created_by_user_id, resource_kind,
                       resource_ref, description, created_at, expires_at,
                       revoked_at, redeemed_at, redeem_count, max_redemptions
                FROM share_links
                WHERE link_id = ?
                """,
                (safe_link_id,),
            ).fetchone()
            if row is None:
                return None
            final_revoked_at = row[8] or now
            if row[8] is None:
                conn.execute(
                    "UPDATE share_links SET revoked_at = ? WHERE link_id = ?",
                    (final_revoked_at, safe_link_id),
                )
        return ShareLinkRecord(
            link_id=row[0],
            workspace_id=row[1],
            created_by_user_id=row[2],
            resource_kind=row[3],
            resource_ref=row[4],
            description=row[5],
            created_at=row[6],
            expires_at=row[7],
            revoked_at=final_revoked_at,
            redeemed_at=row[9],
            redeem_count=int(row[10] or 0),
            max_redemptions=int(row[11] or 0),
        )

    def resolve_token(self, token: str, *, record_redeem: bool = False) -> dict[str, Any]:
        if not token:
            return _invalid_resolution("share_link_token_missing")
        self.ensure_schema()
        token_hash = share_link_token_hash(token)
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT link_id, workspace_id, created_by_user_id, resource_kind,
                       resource_ref, description, created_at, expires_at,
                       revoked_at, redeemed_at, redeem_count, max_redemptions
                FROM share_links
                WHERE token_hash = ?
                """,
                (token_hash,),
            ).fetchone()
            if row is None:
                return _invalid_resolution("share_link_not_found")
            record = _record_from_row(row)
            reason = _record_block_reason(record)
            if reason:
                return {
                    "valid": False,
                    "reason": reason,
                    "share_link": record.to_public_dict(),
                    "content_exported": False,
                    "secrets_exported": False,
                    "share_token_exported": False,
                    "share_url_exported": False,
                }
            if record_redeem:
                redeemed_at = _utc_now()
                redeem_count = record.redeem_count + 1
                conn.execute(
                    """
                    UPDATE share_links
                    SET redeemed_at = ?, redeem_count = ?
                    WHERE link_id = ?
                    """,
                    (redeemed_at, redeem_count, record.link_id),
                )
                record = ShareLinkRecord(
                    **{
                        **asdict(record),
                        "redeemed_at": redeemed_at,
                        "redeem_count": redeem_count,
                    }
                )
        return {
            "valid": True,
            "reason": "allowed",
            "share_link": record.to_public_dict(),
            "content_exported": False,
            "secrets_exported": False,
            "share_token_exported": False,
            "share_url_exported": False,
        }

    def status(self) -> dict[str, Any]:
        self.ensure_schema()
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM share_links").fetchone()[0]
            revoked = conn.execute(
                "SELECT COUNT(*) FROM share_links WHERE revoked_at IS NOT NULL"
            ).fetchone()[0]
            expired = conn.execute(
                "SELECT COUNT(*) FROM share_links WHERE expires_at <= ?",
                (_utc_now(),),
            ).fetchone()[0]
            active = conn.execute(
                """
                SELECT COUNT(*)
                FROM share_links
                WHERE revoked_at IS NULL
                  AND expires_at > ?
                  AND (max_redemptions = 0 OR redeem_count < max_redemptions)
                """,
                (_utc_now(),),
            ).fetchone()[0]
            redeemed = conn.execute(
                "SELECT COALESCE(SUM(redeem_count), 0) FROM share_links"
            ).fetchone()[0]
        return {
            "schema_version": SHARE_LINK_REGISTRY_SCHEMA_VERSION,
            "backend": "sqlite",
            "available": True,
            "active_link_count": int(active or 0),
            "revoked_link_count": int(revoked or 0),
            "expired_link_count": int(expired or 0),
            "total_link_count": int(total or 0),
            "redeem_count": int(redeemed or 0),
            "content_exported": False,
            "secrets_exported": False,
            "paths_exported": False,
            "share_tokens_exported": False,
            "share_urls_exported": False,
        }


def _record_from_row(row: Any) -> ShareLinkRecord:
    return ShareLinkRecord(
        link_id=row[0],
        workspace_id=row[1],
        created_by_user_id=row[2],
        resource_kind=row[3],
        resource_ref=row[4],
        description=row[5],
        created_at=row[6],
        expires_at=row[7],
        revoked_at=row[8],
        redeemed_at=row[9],
        redeem_count=int(row[10] or 0),
        max_redemptions=int(row[11] or 0),
    )


def _record_block_reason(record: ShareLinkRecord) -> str:
    if record.revoked_at is not None:
        return "share_link_revoked"
    if record.expired:
        return "share_link_expired"
    if record.exhausted:
        return "share_link_redemption_limit_exceeded"
    return ""


def _invalid_resolution(reason: str) -> dict[str, Any]:
    return {
        "valid": False,
        "reason": reason,
        "share_link": {},
        "content_exported": False,
        "secrets_exported": False,
        "share_token_exported": False,
        "share_url_exported": False,
    }


def share_link_registry_backend_status(
    *,
    backend: str | None = None,
    db_path: Path | None = None,
) -> dict[str, Any]:
    safe_backend = _safe_backend_name(
        backend if backend is not None else config.SHARE_LINK_TOKEN_STORE_BACKEND
    )
    configured = safe_backend not in DISABLED_SHARE_LINK_BACKENDS
    supported = safe_backend in SUPPORTED_LOCAL_SHARE_LINK_BACKENDS
    if not configured:
        return {
            "schema_version": SHARE_LINK_REGISTRY_SCHEMA_VERSION,
            "backend": safe_backend,
            "configured": False,
            "supported": False,
            "available": False,
            "reason": "share_link_token_store_not_configured",
            "active_link_count": 0,
            "revoked_link_count": 0,
            "expired_link_count": 0,
            "total_link_count": 0,
            "redeem_count": 0,
            "content_exported": False,
            "secrets_exported": False,
            "paths_exported": False,
            "share_tokens_exported": False,
            "share_urls_exported": False,
        }
    if not supported:
        return {
            "schema_version": SHARE_LINK_REGISTRY_SCHEMA_VERSION,
            "backend": safe_backend,
            "configured": True,
            "supported": False,
            "available": False,
            "reason": "share_link_token_store_unavailable",
            "active_link_count": 0,
            "revoked_link_count": 0,
            "expired_link_count": 0,
            "total_link_count": 0,
            "redeem_count": 0,
            "content_exported": False,
            "secrets_exported": False,
            "paths_exported": False,
            "share_tokens_exported": False,
            "share_urls_exported": False,
        }
    try:
        status = LocalShareLinkRegistry(db_path=db_path).status()
    except (OSError, sqlite3.Error):
        return {
            "schema_version": SHARE_LINK_REGISTRY_SCHEMA_VERSION,
            "backend": safe_backend,
            "configured": True,
            "supported": True,
            "available": False,
            "reason": "share_link_token_store_unavailable",
            "active_link_count": 0,
            "revoked_link_count": 0,
            "expired_link_count": 0,
            "total_link_count": 0,
            "redeem_count": 0,
            "content_exported": False,
            "secrets_exported": False,
            "paths_exported": False,
            "share_tokens_exported": False,
            "share_urls_exported": False,
        }
    return {
        **status,
        "configured": True,
        "supported": True,
        "reason": "available",
    }
