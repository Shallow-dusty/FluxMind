"""Small local user and query-history store for the Streamlit research app."""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.config import USER_STORE_FILE


USER_ROLES = ("admin", "student")
_USER_ID_RE = re.compile(r"[a-z0-9][a-z0-9_.-]{1,31}")
_PASSWORD_ITERATIONS = 200_000


@dataclass(frozen=True)
class UserAccount:
    user_id: str
    display_name: str
    role: str
    active: bool
    created_at: str
    updated_at: str
    last_login_at: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class QueryHistoryEntry:
    history_id: int
    user_id: str
    question: str
    answer: str
    answer_mode: str
    created_at: str

    def to_dict(self) -> dict:
        return asdict(self)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_user_id(user_id: str) -> str:
    normalized = user_id.strip().lower()
    if not _USER_ID_RE.fullmatch(normalized):
        raise ValueError("User ID must be 2-32 letters, numbers, dots, dashes, or underscores.")
    return normalized


def _normalize_display_name(display_name: str, *, fallback: str) -> str:
    return display_name.strip()[:80] or fallback


def _validate_role(role: str) -> str:
    if role not in USER_ROLES:
        raise ValueError(f"Role must be one of: {', '.join(USER_ROLES)}.")
    return role


def _password_digest(password: str, salt: bytes) -> bytes:
    if len(password) < 8:
        raise ValueError("Password must contain at least 8 characters.")
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        _PASSWORD_ITERATIONS,
    )


def _account_from_row(row: sqlite3.Row) -> UserAccount:
    return UserAccount(
        user_id=row["user_id"],
        display_name=row["display_name"],
        role=row["role"],
        active=bool(row["active"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        last_login_at=row["last_login_at"],
    )


class LocalUserStore:
    """SQLite-backed accounts and private per-user query history."""

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or USER_STORE_FILE

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def ensure_schema(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ('admin', 'student')),
                    password_salt BLOB NOT NULL,
                    password_hash BLOB NOT NULL,
                    active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_login_at TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS query_history (
                    history_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL REFERENCES users(user_id),
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    answer_mode TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_query_history_user_created
                ON query_history(user_id, created_at DESC)
                """
            )

    def has_users(self) -> bool:
        self.ensure_schema()
        with self._connect() as conn:
            return conn.execute("SELECT 1 FROM users LIMIT 1").fetchone() is not None

    def create_initial_admin(
        self,
        *,
        user_id: str,
        display_name: str,
        password: str,
    ) -> UserAccount:
        if self.has_users():
            raise ValueError("Initial admin already exists.")
        return self.create_user(
            user_id=user_id,
            display_name=display_name,
            password=password,
            role="admin",
        )

    def create_user(
        self,
        *,
        user_id: str,
        display_name: str,
        password: str,
        role: str = "student",
    ) -> UserAccount:
        self.ensure_schema()
        normalized_id = _normalize_user_id(user_id)
        normalized_role = _validate_role(role)
        normalized_name = _normalize_display_name(display_name, fallback=normalized_id)
        salt = secrets.token_bytes(16)
        password_hash = _password_digest(password, salt)
        now = _utc_now()
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO users (
                        user_id, display_name, role, password_salt, password_hash,
                        active, created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, 1, ?, ?)
                    """,
                    (
                        normalized_id,
                        normalized_name,
                        normalized_role,
                        salt,
                        password_hash,
                        now,
                        now,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError("User ID already exists.") from exc
        account = self.get_user(normalized_id)
        assert account is not None
        return account

    def get_user(self, user_id: str) -> UserAccount | None:
        self.ensure_schema()
        try:
            normalized_id = _normalize_user_id(user_id)
        except ValueError:
            return None
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT user_id, display_name, role, active, created_at, updated_at, last_login_at
                FROM users
                WHERE user_id = ?
                """,
                (normalized_id,),
            ).fetchone()
        return _account_from_row(row) if row else None

    def authenticate(self, user_id: str, password: str) -> UserAccount | None:
        self.ensure_schema()
        try:
            normalized_id = _normalize_user_id(user_id)
        except ValueError:
            return None
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT user_id, display_name, role, password_salt, password_hash,
                       active, created_at, updated_at, last_login_at
                FROM users
                WHERE user_id = ?
                """,
                (normalized_id,),
            ).fetchone()
            if row is None or not row["active"]:
                return None
            try:
                candidate = _password_digest(password, row["password_salt"])
            except ValueError:
                return None
            if not hmac.compare_digest(candidate, row["password_hash"]):
                return None
            last_login_at = _utc_now()
            conn.execute(
                "UPDATE users SET last_login_at = ? WHERE user_id = ?",
                (last_login_at, normalized_id),
            )
        return self.get_user(normalized_id)

    def list_users(self, *, include_inactive: bool = False) -> list[UserAccount]:
        self.ensure_schema()
        where = "" if include_inactive else "WHERE active = 1"
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT user_id, display_name, role, active, created_at, updated_at, last_login_at
                FROM users
                {where}
                ORDER BY role ASC, display_name COLLATE NOCASE ASC
                """
            ).fetchall()
        return [_account_from_row(row) for row in rows]

    def update_user(
        self,
        user_id: str,
        *,
        display_name: str,
        role: str,
        active: bool,
    ) -> UserAccount:
        normalized_id = _normalize_user_id(user_id)
        normalized_role = _validate_role(role)
        normalized_name = _normalize_display_name(display_name, fallback=normalized_id)
        self.ensure_schema()
        with self._connect() as conn:
            current = conn.execute(
                "SELECT role, active FROM users WHERE user_id = ?",
                (normalized_id,),
            ).fetchone()
            if current is None:
                raise ValueError("User not found.")
            removing_admin = (
                current["role"] == "admin"
                and bool(current["active"])
                and (normalized_role != "admin" or not active)
            )
            if removing_admin:
                active_admins = conn.execute(
                    "SELECT COUNT(*) FROM users WHERE role = 'admin' AND active = 1"
                ).fetchone()[0]
                if active_admins <= 1:
                    raise ValueError("At least one active admin is required.")
            conn.execute(
                """
                UPDATE users
                SET display_name = ?, role = ?, active = ?, updated_at = ?
                WHERE user_id = ?
                """,
                (normalized_name, normalized_role, int(active), _utc_now(), normalized_id),
            )
        return self.get_user(normalized_id)

    def set_password(self, user_id: str, password: str) -> None:
        normalized_id = _normalize_user_id(user_id)
        salt = secrets.token_bytes(16)
        password_hash = _password_digest(password, salt)
        self.ensure_schema()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE users
                SET password_salt = ?, password_hash = ?, updated_at = ?
                WHERE user_id = ?
                """,
                (salt, password_hash, _utc_now(), normalized_id),
            )
        if cursor.rowcount == 0:
            raise ValueError("User not found.")

    def record_query(
        self,
        *,
        user_id: str,
        question: str,
        answer: str,
        answer_mode: str,
    ) -> QueryHistoryEntry:
        normalized_id = _normalize_user_id(user_id)
        created_at = _utc_now()
        self.ensure_schema()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO query_history (user_id, question, answer, answer_mode, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (normalized_id, question.strip(), answer.strip(), answer_mode, created_at),
            )
            history_id = int(cursor.lastrowid)
        return QueryHistoryEntry(
            history_id=history_id,
            user_id=normalized_id,
            question=question.strip(),
            answer=answer.strip(),
            answer_mode=answer_mode,
            created_at=created_at,
        )

    def list_history(self, user_id: str, *, limit: int = 50) -> list[QueryHistoryEntry]:
        normalized_id = _normalize_user_id(user_id)
        limit = max(1, min(int(limit), 200))
        self.ensure_schema()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT history_id, user_id, question, answer, answer_mode, created_at
                FROM query_history
                WHERE user_id = ?
                ORDER BY history_id DESC
                LIMIT ?
                """,
                (normalized_id, limit),
            ).fetchall()
        return [
            QueryHistoryEntry(
                history_id=row["history_id"],
                user_id=row["user_id"],
                question=row["question"],
                answer=row["answer"],
                answer_mode=row["answer_mode"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def clear_history(self, user_id: str) -> int:
        normalized_id = _normalize_user_id(user_id)
        self.ensure_schema()
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM query_history WHERE user_id = ?",
                (normalized_id,),
            )
        return cursor.rowcount
