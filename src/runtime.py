"""Runtime helpers for request IDs, logging, and user-facing errors."""

from __future__ import annotations

import logging
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import RUNTIME_EVENTS_FILE


logger = logging.getLogger("fluxmind")


def new_request_id() -> str:
    """Return a short stable-enough request ID for logs and responses."""
    return uuid.uuid4().hex[:12]


def estimate_text_tokens(text: str) -> int:
    """Return a rough no-secret token estimate when provider usage is unavailable."""
    normalized = " ".join(text.split())
    if not normalized:
        return 0
    return max(1, (len(normalized) + 3) // 4)


@dataclass(frozen=True)
class UserFacingError:
    """Normalized error payload safe to show to UI/API users."""

    code: str
    message: str
    status_code: int = 500


class FluxMindError(Exception):
    """Base error for failures that already have a user-facing shape."""

    def __init__(self, code: str, message: str, *, status_code: int = 500):
        super().__init__(message)
        self.user_error = UserFacingError(code, message, status_code)


class ProviderError(FluxMindError):
    """Raised when the LLM or another provider fails."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "provider_error",
        status_code: int = 502,
    ):
        super().__init__(code, message, status_code=status_code)


@dataclass(frozen=True)
class RuntimeEvent:
    """No-secret runtime event for admin/status history."""

    event_id: str
    kind: str
    code: str
    message: str
    created_at: str
    request_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_runtime_event(
    *,
    kind: str,
    code: str,
    message: str,
    request_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    path: Path | None = None,
) -> RuntimeEvent:
    """Append a no-secret runtime event to the local JSONL history."""
    event = RuntimeEvent(
        event_id=new_request_id(),
        kind=kind,
        code=code,
        message=message[:500],
        created_at=utc_now(),
        request_id=request_id,
        metadata=metadata or {},
    )
    target = path or RUNTIME_EVENTS_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")
    return event


def list_runtime_events(
    *,
    kind: str | None = None,
    code: str | None = None,
    q: str | None = None,
    limit: int = 50,
    path: Path | None = None,
) -> list[RuntimeEvent]:
    """Read latest no-secret runtime events from the local JSONL history."""
    target = path or RUNTIME_EVENTS_FILE
    if not target.exists():
        return []
    kind = kind.strip() if kind else None
    code = code.strip() if code else None
    query = (q or "").strip().casefold()
    events: list[RuntimeEvent] = []
    with target.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("runtime_events.invalid_json path=%s line=%s", target, line_number)
                continue
            if not isinstance(item, dict):
                logger.warning("runtime_events.invalid_type path=%s line=%s", target, line_number)
                continue
            if kind and item.get("kind") != kind:
                continue
            if code and item.get("code") != code:
                continue
            if query:
                searchable = " ".join(
                    str(value or "")
                    for value in (
                        item.get("event_id"),
                        item.get("kind"),
                        item.get("code"),
                        item.get("message"),
                        item.get("request_id"),
                        json.dumps(item.get("metadata") or {}, ensure_ascii=False, sort_keys=True),
                    )
                ).casefold()
                if query not in searchable:
                    continue
            try:
                events.append(RuntimeEvent(**item))
            except TypeError:
                logger.warning("runtime_events.invalid_event path=%s line=%s", target, line_number)
    return events[-limit:][::-1]


def normalize_exception(exc: Exception) -> UserFacingError:
    """Map internal exceptions to safe API/UI messages."""
    if isinstance(exc, FluxMindError):
        return exc.user_error

    text = str(exc).lower()
    if "timeout" in text or "timed out" in text:
        return UserFacingError(
            "provider_timeout",
            "The model provider timed out. Please retry the request.",
            504,
        )
    if "429" in text or "rate limit" in text or "quota" in text:
        return UserFacingError(
            "provider_rate_limited",
            "The model provider is rate limited. Please retry later.",
            429,
        )
    if "api key" in text or "authentication" in text or "unauthorized" in text:
        return UserFacingError(
            "provider_auth_failed",
            "The model provider rejected the configured credentials.",
            502,
        )
    if "upstream_empty_output" in text or "empty output" in text:
        return UserFacingError(
            "provider_empty_output",
            "The model provider returned an empty response. Please retry later.",
            502,
        )
    if "malformed" in text and ("stream" in text or "chunk" in text or "response" in text):
        return UserFacingError(
            "provider_malformed_response",
            "The model provider returned a malformed response.",
            502,
        )

    return UserFacingError(
        "internal_error",
        "FluxMind could not complete the request. Please retry or check logs.",
        500,
    )
