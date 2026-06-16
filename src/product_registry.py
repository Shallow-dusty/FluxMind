"""Local no-secret product identity, workspace, quota, and billing ledger."""

from __future__ import annotations

import secrets
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from src import config


PRODUCT_REGISTRY_SCHEMA_VERSION = 1
SUPPORTED_LOCAL_PRODUCT_REGISTRY_BACKENDS = {"sqlite"}
DISABLED_PRODUCT_REGISTRY_BACKENDS = {"", "none", "disabled", "local-disabled"}
SAFE_BACKEND_CHARS = set("abcdefghijklmnopqrstuvwxyz0123456789_.-")
SAFE_ID_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-")
PRODUCT_ROLE_ORDER = ("owner", "admin", "member", "viewer")
PRODUCT_RBAC_ACTION_ROLES = {
    "query": {"owner", "admin", "member", "viewer"},
    "job_submit": {"owner", "admin", "member"},
    "corpus_write": {"owner", "admin"},
    "admin_write": {"owner", "admin"},
}


@dataclass(frozen=True)
class ProductUser:
    user_id: str
    label: str
    status: str
    created_at: str
    updated_at: str

    def to_public_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProductWorkspace:
    workspace_id: str
    label: str
    owner_user_id: str
    status: str
    created_at: str
    updated_at: str

    def to_public_dict(self) -> dict[str, Any]:
        return asdict(self)


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
    return getattr(config, "PRODUCT_REGISTRY_FILE", config.METADATA_DIR / "product_registry.sqlite3")


def _generated_id(prefix: str) -> str:
    return prefix + "_" + secrets.token_urlsafe(12).replace("-", "").replace("_", "")[:16]


def _safe_identifier(value: str | None, *, prefix: str) -> str:
    candidate = (value or "").strip()
    if not candidate:
        return _generated_id(prefix)
    candidate = candidate[:64]
    if any(char not in SAFE_ID_CHARS for char in candidate):
        return _generated_id(prefix)
    return candidate


def _safe_label(value: str | None, fallback: str) -> str:
    label = (value or "").strip()
    return (label or fallback)[:128]


def _ordered_roles(roles: set[str]) -> list[str]:
    return [role for role in PRODUCT_ROLE_ORDER if role in roles]


class LocalProductRegistry:
    """SQLite-backed local product registry.

    This registry is a local contract for users, workspaces, quotas, usage, and
    billing attribution. It does not store provider secrets, billing credentials,
    external account tokens, prompts, answers, or uploaded file contents.
    """

    def __init__(self, db_path: Path | None = None):
        self.db_path = _registry_db_path(db_path)

    def ensure_schema(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS product_users (
                    user_id TEXT PRIMARY KEY,
                    label TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS workspaces (
                    workspace_id TEXT PRIMARY KEY,
                    label TEXT NOT NULL,
                    owner_user_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS workspace_members (
                    workspace_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (workspace_id, user_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS quota_limits (
                    workspace_id TEXT NOT NULL,
                    metric TEXT NOT NULL,
                    limit_value INTEGER NOT NULL,
                    window_s INTEGER NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (workspace_id, metric)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS usage_events (
                    event_id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    metric TEXT NOT NULL,
                    amount INTEGER NOT NULL,
                    source TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS billing_accounts (
                    workspace_id TEXT PRIMARY KEY,
                    billing_mode TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attribution_enabled INTEGER NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_workspaces_owner ON workspaces(owner_user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_members_user ON workspace_members(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_workspace_metric ON usage_events(workspace_id, metric)")

    def upsert_user(
        self,
        *,
        user_id: str | None = None,
        label: str | None = None,
        status: str = "active",
    ) -> ProductUser:
        self.ensure_schema()
        safe_user_id = _safe_identifier(user_id, prefix="user")
        now = _utc_now()
        safe_label = _safe_label(label, safe_user_id)
        safe_status = status if status in {"active", "disabled"} else "active"
        with sqlite3.connect(self.db_path) as conn:
            existing = conn.execute(
                "SELECT created_at FROM product_users WHERE user_id = ?",
                (safe_user_id,),
            ).fetchone()
            created_at = existing[0] if existing else now
            conn.execute(
                """
                INSERT INTO product_users (user_id, label, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    label = excluded.label,
                    status = excluded.status,
                    updated_at = excluded.updated_at
                """,
                (safe_user_id, safe_label, safe_status, created_at, now),
            )
        return ProductUser(
            user_id=safe_user_id,
            label=safe_label,
            status=safe_status,
            created_at=created_at,
            updated_at=now,
        )

    def create_workspace(
        self,
        *,
        workspace_id: str | None = None,
        label: str | None = None,
        owner_user_id: str | None = None,
        owner_label: str | None = None,
    ) -> ProductWorkspace:
        owner = self.upsert_user(user_id=owner_user_id, label=owner_label)
        safe_workspace_id = _safe_identifier(workspace_id, prefix="ws")
        safe_label = _safe_label(label, safe_workspace_id)
        now = _utc_now()
        with sqlite3.connect(self.db_path) as conn:
            existing = conn.execute(
                "SELECT created_at FROM workspaces WHERE workspace_id = ?",
                (safe_workspace_id,),
            ).fetchone()
            created_at = existing[0] if existing else now
            conn.execute(
                """
                INSERT INTO workspaces (
                    workspace_id, label, owner_user_id, status, created_at, updated_at
                )
                VALUES (?, ?, ?, 'active', ?, ?)
                ON CONFLICT(workspace_id) DO UPDATE SET
                    label = excluded.label,
                    owner_user_id = excluded.owner_user_id,
                    status = 'active',
                    updated_at = excluded.updated_at
                """,
                (safe_workspace_id, safe_label, owner.user_id, created_at, now),
            )
            conn.execute(
                """
                INSERT INTO workspace_members (
                    workspace_id, user_id, role, status, created_at, updated_at
                )
                VALUES (?, ?, 'owner', 'active', ?, ?)
                ON CONFLICT(workspace_id, user_id) DO UPDATE SET
                    role = 'owner',
                    status = 'active',
                    updated_at = excluded.updated_at
                """,
                (safe_workspace_id, owner.user_id, now, now),
            )
        return ProductWorkspace(
            workspace_id=safe_workspace_id,
            label=safe_label,
            owner_user_id=owner.user_id,
            status="active",
            created_at=created_at,
            updated_at=now,
        )

    def add_member(
        self,
        *,
        workspace_id: str,
        user_id: str,
        label: str | None = None,
        role: str = "member",
    ) -> None:
        self.ensure_schema()
        user = self.upsert_user(user_id=user_id, label=label)
        safe_workspace_id = _safe_identifier(workspace_id, prefix="ws")
        safe_role = role if role in {"owner", "admin", "member", "viewer"} else "member"
        now = _utc_now()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO workspace_members (
                    workspace_id, user_id, role, status, created_at, updated_at
                )
                VALUES (?, ?, ?, 'active', ?, ?)
                ON CONFLICT(workspace_id, user_id) DO UPDATE SET
                    role = excluded.role,
                    status = 'active',
                    updated_at = excluded.updated_at
                """,
                (safe_workspace_id, user.user_id, safe_role, now, now),
            )

    def set_quota(
        self,
        *,
        workspace_id: str,
        metric: str,
        limit_value: int,
        window_s: int,
    ) -> dict[str, Any]:
        self.ensure_schema()
        safe_workspace_id = _safe_identifier(workspace_id, prefix="ws")
        safe_metric = _safe_identifier(metric, prefix="metric")
        limit_int = max(0, int(limit_value))
        window_int = max(0, int(window_s))
        now = _utc_now()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO quota_limits (
                    workspace_id, metric, limit_value, window_s, updated_at
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(workspace_id, metric) DO UPDATE SET
                    limit_value = excluded.limit_value,
                    window_s = excluded.window_s,
                    updated_at = excluded.updated_at
                """,
                (safe_workspace_id, safe_metric, limit_int, window_int, now),
            )
        return {
            "workspace_id": safe_workspace_id,
            "metric": safe_metric,
            "limit_value": limit_int,
            "window_s": window_int,
            "updated_at": now,
        }

    def record_usage(
        self,
        *,
        workspace_id: str,
        user_id: str,
        metric: str,
        amount: int,
        source: str = "manual",
    ) -> dict[str, Any]:
        self.ensure_schema()
        safe_workspace_id = _safe_identifier(workspace_id, prefix="ws")
        safe_user_id = _safe_identifier(user_id, prefix="user")
        safe_metric = _safe_identifier(metric, prefix="metric")
        amount_int = max(0, int(amount))
        safe_source = _safe_label(source, "manual")
        event_id = _generated_id("usage")
        created_at = _utc_now()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO usage_events (
                    event_id, workspace_id, user_id, metric, amount, source, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    safe_workspace_id,
                    safe_user_id,
                    safe_metric,
                    amount_int,
                    safe_source,
                    created_at,
                ),
            )
        return {
            "event_id": event_id,
            "workspace_id": safe_workspace_id,
            "user_id": safe_user_id,
            "metric": safe_metric,
            "amount": amount_int,
            "source": safe_source,
            "created_at": created_at,
        }

    def workspace_for_user(
        self,
        *,
        user_id: str,
        workspace_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Return an active workspace membership for a local product user."""
        self.ensure_schema()
        safe_user_id = _safe_identifier(user_id, prefix="user")
        requested_workspace_id = (
            _safe_identifier(workspace_id, prefix="ws") if workspace_id else None
        )
        where = [
            "m.user_id = ?",
            "m.status = 'active'",
            "w.status = 'active'",
        ]
        params: list[Any] = [safe_user_id]
        if requested_workspace_id:
            where.append("m.workspace_id = ?")
            params.append(requested_workspace_id)
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                f"""
                SELECT w.workspace_id, w.label, w.owner_user_id, m.user_id, m.role
                FROM workspace_members m
                JOIN workspaces w ON w.workspace_id = m.workspace_id
                WHERE {" AND ".join(where)}
                ORDER BY
                    CASE m.role
                        WHEN 'owner' THEN 0
                        WHEN 'admin' THEN 1
                        WHEN 'member' THEN 2
                        ELSE 3
                    END,
                    w.created_at ASC,
                    w.workspace_id ASC
                LIMIT 1
                """,
                params,
            ).fetchone()
        if row is None:
            return None
        return {
            "workspace_id": row[0],
            "workspace_label": row[1],
            "owner_user_id": row[2],
            "user_id": row[3],
            "role": row[4],
            "content_exported": False,
            "secrets_exported": False,
        }

    def permission_decision(
        self,
        *,
        user_id: str,
        action: str,
        workspace_id: str | None = None,
    ) -> dict[str, Any]:
        """Check local workspace role permissions without exporting secrets."""
        safe_action = _safe_identifier(action, prefix="action")
        required_roles = PRODUCT_RBAC_ACTION_ROLES.get(safe_action)
        safe_user_id = _safe_identifier(user_id, prefix="user")
        safe_workspace_hint = _safe_identifier(workspace_id, prefix="ws") if workspace_id else ""
        if required_roles is None:
            return {
                "allowed": False,
                "reason": "unsupported_product_action",
                "action": safe_action,
                "user_id": safe_user_id,
                "workspace_id": safe_workspace_hint,
                "role": "",
                "required_roles": [],
                "content_exported": False,
                "secrets_exported": False,
            }

        membership = self.workspace_for_user(user_id=safe_user_id, workspace_id=workspace_id)
        if membership is None:
            return {
                "allowed": False,
                "reason": "product_workspace_not_found",
                "action": safe_action,
                "user_id": safe_user_id,
                "workspace_id": safe_workspace_hint,
                "role": "",
                "required_roles": _ordered_roles(required_roles),
                "content_exported": False,
                "secrets_exported": False,
            }

        role = str(membership.get("role", ""))
        allowed = role in required_roles
        return {
            "allowed": allowed,
            "reason": "allowed" if allowed else "product_role_forbidden",
            "action": safe_action,
            "user_id": safe_user_id,
            "workspace_id": membership.get("workspace_id", ""),
            "workspace_label": membership.get("workspace_label", ""),
            "role": role,
            "required_roles": _ordered_roles(required_roles),
            "content_exported": False,
            "secrets_exported": False,
        }

    def quota_decision(
        self,
        *,
        workspace_id: str,
        user_id: str,
        metric: str,
        amount: int = 1,
        source: str = "runtime",
        record: bool = True,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Check a local quota window and optionally record no-secret usage."""
        self.ensure_schema()
        safe_workspace_id = _safe_identifier(workspace_id, prefix="ws")
        safe_user_id = _safe_identifier(user_id, prefix="user")
        safe_metric = _safe_identifier(metric, prefix="metric")
        amount_int = max(0, int(amount))
        safe_source = _safe_label(source, "runtime")
        now_dt = now or datetime.now(timezone.utc)
        if now_dt.tzinfo is None:
            now_dt = now_dt.replace(tzinfo=timezone.utc)
        with sqlite3.connect(self.db_path) as conn:
            quota = conn.execute(
                """
                SELECT limit_value, window_s
                FROM quota_limits
                WHERE workspace_id = ? AND metric = ?
                """,
                (safe_workspace_id, safe_metric),
            ).fetchone()
            if quota is None:
                usage_event_id = None
                if record:
                    usage_event_id = _generated_id("usage")
                    conn.execute(
                        """
                        INSERT INTO usage_events (
                            event_id, workspace_id, user_id, metric, amount, source, created_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            usage_event_id,
                            safe_workspace_id,
                            safe_user_id,
                            safe_metric,
                            amount_int,
                            safe_source,
                            now_dt.isoformat(),
                        ),
                    )
                return {
                    "enabled": True,
                    "allowed": True,
                    "limited": False,
                    "reason": "quota_not_configured",
                    "quota_configured": False,
                    "workspace_id": safe_workspace_id,
                    "user_id": safe_user_id,
                    "metric": safe_metric,
                    "amount": amount_int,
                    "limit_value": 0,
                    "used": 0,
                    "remaining": 0,
                    "window_s": 0,
                    "reset_after_s": 0,
                    "usage_event_id": usage_event_id,
                    "content_exported": False,
                    "secrets_exported": False,
                }
            limit_value = max(0, int(quota[0] or 0))
            window_s = max(1, int(quota[1] or 0))
            cutoff = (now_dt - timedelta(seconds=window_s)).isoformat()
            row = conn.execute(
                """
                SELECT COALESCE(SUM(amount), 0), COUNT(*), MIN(created_at)
                FROM usage_events
                WHERE workspace_id = ? AND metric = ? AND created_at >= ?
                """,
                (safe_workspace_id, safe_metric, cutoff),
            ).fetchone()
            used = int(row[0] or 0)
            oldest = row[2]
            allowed = used + amount_int <= limit_value
            usage_event_id = None
            if allowed and record:
                event_id = _generated_id("usage")
                created_at = now_dt.isoformat()
                conn.execute(
                    """
                    INSERT INTO usage_events (
                        event_id, workspace_id, user_id, metric, amount, source, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event_id,
                        safe_workspace_id,
                        safe_user_id,
                        safe_metric,
                        amount_int,
                        safe_source,
                        created_at,
                    ),
                )
                usage_event_id = event_id
                used += amount_int
            reset_after_s = window_s
            if oldest:
                try:
                    oldest_dt = datetime.fromisoformat(str(oldest))
                    if oldest_dt.tzinfo is None:
                        oldest_dt = oldest_dt.replace(tzinfo=timezone.utc)
                    reset_after_s = max(0, int((oldest_dt + timedelta(seconds=window_s) - now_dt).total_seconds()))
                except ValueError:
                    reset_after_s = window_s
        return {
            "enabled": True,
            "allowed": allowed,
            "limited": not allowed,
            "reason": "allowed" if allowed else "quota_exceeded",
            "quota_configured": True,
            "workspace_id": safe_workspace_id,
            "user_id": safe_user_id,
            "metric": safe_metric,
            "amount": amount_int,
            "limit_value": limit_value,
            "used": used,
            "remaining": max(0, limit_value - used),
            "window_s": window_s,
            "reset_after_s": reset_after_s,
            "usage_event_id": usage_event_id,
            "content_exported": False,
            "secrets_exported": False,
        }

    def set_billing_account(
        self,
        *,
        workspace_id: str,
        billing_mode: str = "local-ledger",
        status: str = "active",
        attribution_enabled: bool = True,
    ) -> dict[str, Any]:
        self.ensure_schema()
        safe_workspace_id = _safe_identifier(workspace_id, prefix="ws")
        safe_mode = _safe_identifier(billing_mode, prefix="billing")
        safe_status = status if status in {"active", "disabled"} else "active"
        now = _utc_now()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO billing_accounts (
                    workspace_id, billing_mode, status, attribution_enabled, updated_at
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(workspace_id) DO UPDATE SET
                    billing_mode = excluded.billing_mode,
                    status = excluded.status,
                    attribution_enabled = excluded.attribution_enabled,
                    updated_at = excluded.updated_at
                """,
                (
                    safe_workspace_id,
                    safe_mode,
                    safe_status,
                    1 if attribution_enabled else 0,
                    now,
                ),
            )
        return {
            "workspace_id": safe_workspace_id,
            "billing_mode": safe_mode,
            "status": safe_status,
            "attribution_enabled": bool(attribution_enabled),
            "updated_at": now,
        }

    def list_workspaces(self) -> list[ProductWorkspace]:
        self.ensure_schema()
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT workspace_id, label, owner_user_id, status, created_at, updated_at
                FROM workspaces
                ORDER BY created_at DESC, workspace_id DESC
                """
            ).fetchall()
        return [
            ProductWorkspace(
                workspace_id=row[0],
                label=row[1],
                owner_user_id=row[2],
                status=row[3],
                created_at=row[4],
                updated_at=row[5],
            )
            for row in rows
        ]

    def status(self) -> dict[str, Any]:
        self.ensure_schema()
        with sqlite3.connect(self.db_path) as conn:
            user_count = conn.execute("SELECT COUNT(*) FROM product_users").fetchone()[0]
            workspace_count = conn.execute("SELECT COUNT(*) FROM workspaces").fetchone()[0]
            member_count = conn.execute("SELECT COUNT(*) FROM workspace_members").fetchone()[0]
            quota_count = conn.execute("SELECT COUNT(*) FROM quota_limits").fetchone()[0]
            usage_count = conn.execute("SELECT COUNT(*) FROM usage_events").fetchone()[0]
            billing_count = conn.execute("SELECT COUNT(*) FROM billing_accounts").fetchone()[0]
            billing_attribution_count = conn.execute(
                "SELECT COUNT(*) FROM billing_accounts WHERE attribution_enabled = 1"
            ).fetchone()[0]
        return {
            "schema_version": PRODUCT_REGISTRY_SCHEMA_VERSION,
            "backend": "sqlite",
            "available": True,
            "user_count": int(user_count or 0),
            "workspace_count": int(workspace_count or 0),
            "member_count": int(member_count or 0),
            "quota_limit_count": int(quota_count or 0),
            "usage_event_count": int(usage_count or 0),
            "billing_account_count": int(billing_count or 0),
            "billing_attribution_count": int(billing_attribution_count or 0),
            "identity_available": True,
            "workspace_available": True,
            "rbac_available": True,
            "quota_store_available": True,
            "billing_ledger_available": True,
            "content_exported": False,
            "secrets_exported": False,
        }


def product_registry_backend_status(
    *,
    backend: str | None = None,
    db_path: Path | None = None,
) -> dict[str, Any]:
    safe_backend = _safe_backend_name(backend if backend is not None else config.PRODUCT_REGISTRY_BACKEND)
    configured = safe_backend not in DISABLED_PRODUCT_REGISTRY_BACKENDS
    supported = safe_backend in SUPPORTED_LOCAL_PRODUCT_REGISTRY_BACKENDS
    if not configured:
        return {
            "schema_version": PRODUCT_REGISTRY_SCHEMA_VERSION,
            "backend": safe_backend,
            "configured": False,
            "supported": False,
            "available": False,
            "reason": "product_registry_not_configured",
            "user_count": 0,
            "workspace_count": 0,
            "quota_limit_count": 0,
            "billing_account_count": 0,
            "rbac_available": False,
            "content_exported": False,
            "secrets_exported": False,
            "paths_exported": False,
        }
    if not supported:
        return {
            "schema_version": PRODUCT_REGISTRY_SCHEMA_VERSION,
            "backend": safe_backend,
            "configured": True,
            "supported": False,
            "available": False,
            "reason": "product_registry_backend_not_local",
            "user_count": 0,
            "workspace_count": 0,
            "quota_limit_count": 0,
            "billing_account_count": 0,
            "rbac_available": False,
            "content_exported": False,
            "secrets_exported": False,
            "paths_exported": False,
        }
    try:
        status = LocalProductRegistry(db_path=db_path).status()
    except (OSError, sqlite3.Error):
        return {
            "schema_version": PRODUCT_REGISTRY_SCHEMA_VERSION,
            "backend": safe_backend,
            "configured": True,
            "supported": True,
            "available": False,
            "reason": "product_registry_unavailable",
            "user_count": 0,
            "workspace_count": 0,
            "quota_limit_count": 0,
            "billing_account_count": 0,
            "rbac_available": False,
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
