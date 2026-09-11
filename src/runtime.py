"""Small runtime primitives shared by the API, UI, jobs, and evaluation code."""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.config import RUNTIME_EVENTS_FILE


logger = logging.getLogger("fluxmind")
RUNTIME_EVENT_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$")


def new_request_id() -> str:
    """Return a short correlation ID for logs and responses."""
    return uuid.uuid4().hex[:12]


def normalize_runtime_event_request_id(value: Any) -> str | None:
    """Validate an externally supplied correlation ID."""
    text = str(value or "").strip()
    if not text:
        return None
    text = text[:64]
    if not RUNTIME_EVENT_REQUEST_ID_RE.fullmatch(text):
        return None
    return text


def runtime_ownership_metadata(ownership: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return ownership metadata used by operational events."""
    ownership = ownership or {}
    return {
        "owner_id": str(ownership.get("owner_id", "") or ""),
        "owner_label": str(ownership.get("owner_label", "") or ""),
        "ownership_source": str(ownership.get("ownership_source", "") or ""),
    }


def estimate_text_tokens(text: str) -> int:
    """Return a rough token estimate when provider usage is unavailable."""
    normalized = " ".join(text.split())
    if not normalized:
        return 0
    return max(1, (len(normalized) + 3) // 4)


@dataclass(frozen=True)
class UserFacingError:
    code: str
    message: str
    status_code: int = 500


class FluxMindError(Exception):
    """Base error for failures that already have a public shape."""

    def __init__(self, code: str, message: str, *, status_code: int = 500):
        super().__init__(message)
        self.user_error = UserFacingError(code, message, status_code)


class ProviderError(FluxMindError):
    def __init__(
        self,
        message: str,
        *,
        code: str = "provider_error",
        status_code: int = 502,
    ):
        super().__init__(code, message, status_code=status_code)


class ProviderQuotaGuardError(FluxMindError):
    def __init__(
        self,
        message: str,
        *,
        code: str = "provider_quota_guard_denied",
        status_code: int = 429,
        decision: dict[str, Any] | None = None,
    ):
        super().__init__(code, message, status_code=status_code)
        self.decision = decision or {}


@dataclass(frozen=True)
class RuntimeEvent:
    """One append-only operational event."""

    event_id: str
    kind: str
    code: str
    message: str
    created_at: str
    request_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


_RUNTIME_EVENT_FIELDS = set(RuntimeEvent.__dataclass_fields__)


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
    """Append an event.

    Call sites own the event schema and must pass operational metadata rather
    than prompts, answers, uploaded content, credentials, or arbitrary payloads.
    """
    normalized_request_id = normalize_runtime_event_request_id(request_id)
    event = RuntimeEvent(
        event_id=new_request_id(),
        kind=str(kind)[:80],
        code=str(code)[:120],
        message=str(message or "")[:500],
        created_at=utc_now(),
        request_id=normalized_request_id,
        metadata=dict(metadata or {}),
    )
    target = path or RUNTIME_EVENTS_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")
    return event


def runtime_event_to_dict(
    event: RuntimeEvent,
    *,
    include_request_id: bool = True,
) -> dict[str, Any]:
    """Convert an event to its API/UI representation."""
    payload = asdict(event)
    if not include_request_id:
        payload.pop("request_id", None)
    return payload


def list_runtime_events(
    *,
    kind: str | None = None,
    code: str | None = None,
    q: str | None = None,
    limit: int = 50,
    path: Path | None = None,
) -> list[RuntimeEvent]:
    """Read the newest matching events from the local JSONL history."""
    target = path or RUNTIME_EVENTS_FILE
    if not target.exists():
        return []

    expected_kind = kind.strip() if kind else None
    expected_code = code.strip() if code else None
    query = (q or "").strip().casefold()
    events: list[RuntimeEvent] = []

    with target.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise TypeError
                event = RuntimeEvent(
                    **{key: value for key, value in item.items() if key in _RUNTIME_EVENT_FIELDS}
                )
            except (json.JSONDecodeError, TypeError):
                logger.warning(
                    "runtime_events.invalid_event path=%s line=%s",
                    target,
                    line_number,
                )
                continue
            if expected_kind and event.kind != expected_kind:
                continue
            if expected_code and event.code != expected_code:
                continue
            if query:
                searchable = json.dumps(
                    runtime_event_to_dict(event),
                    ensure_ascii=False,
                    sort_keys=True,
                ).casefold()
                if query not in searchable:
                    continue
            events.append(event)

    bounded_limit = max(1, int(limit))
    return events[-bounded_limit:][::-1]


def normalize_exception(exc: Exception) -> UserFacingError:
    """Map internal exceptions to stable API/UI messages."""
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
        str(exc).strip() or exc.__class__.__name__,
        500,
    )
