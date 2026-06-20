"""Runtime helpers for request IDs, logging, and user-facing errors."""

from __future__ import annotations

import logging
import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.config import RUNTIME_EVENTS_FILE


logger = logging.getLogger("fluxmind")
SENSITIVE_RUNTIME_EVENT_METADATA_KEYS = {
    "api_key",
    "answer",
    "authorization",
    "body",
    "content",
    "credential",
    "file_content",
    "filename",
    "headers",
    "owner_id",
    "owner_label",
    "path",
    "prompt",
    "request_body",
    "request_id",
    "source_path",
    "source_paths",
    "text",
    "token",
    "uri",
}
SENSITIVE_RUNTIME_EVENT_METADATA_WORDS = {
    "authorization",
    "body",
    "content",
    "filename",
    "headers",
    "log",
    "logs",
    "password",
    "passwd",
    "path",
    "paths",
    "payload",
    "result",
    "secret",
    "stderr",
    "stdout",
    "text",
    "uri",
    "url",
    "urls",
}
SENSITIVE_RUNTIME_EVENT_CONTENT_WORDS = {
    "answer",
    "prompt",
    "question",
}
SENSITIVE_RUNTIME_EVENT_SUBJECT_WORDS = {
    "account",
    "auth",
    "customer",
    "member",
    "owner",
    "tenant",
    "user",
    "workspace",
}
SENSITIVE_RUNTIME_EVENT_IDENTIFIER_WORDS = {
    "email",
    "emails",
    "id",
    "ids",
    "label",
    "labels",
    "name",
    "names",
}
SAFE_CONTENT_METADATA_WORDS = {
    "coverage",
    "count",
    "counts",
    "mode",
    "rate",
    "score",
    "tokens",
}
SAFE_CREDENTIAL_METADATA_WORDS = {
    "backend",
    "configured",
    "count",
    "counts",
    "enabled",
    "limit",
    "limits",
    "present",
    "registry",
    "remaining",
    "source",
    "status",
    "tokens",
    "type",
    "window",
}
SENSITIVE_RUNTIME_EVENT_METADATA_COMPACT_KEYS = {
    "apikey",
    "authorizationheader",
    "authkeyid",
    "customerid",
    "keyid",
    "memberuserid",
    "owneruserid",
    "ownerid",
    "ownerlabel",
    "privatekey",
    "requestid",
    "secretkey",
    "tenantid",
    "userid",
    "useremail",
    "userlabel",
    "username",
    "workspaceid",
    "workspacelabel",
    "workspacename",
}
SENSITIVE_RUNTIME_EVENT_MESSAGE_PATTERNS = (
    re.compile(r"https?://\S+", re.IGNORECASE),
    re.compile(r"\b[A-Za-z]:\\[^\s]+"),
    re.compile(r"(?<!\w)/(?:[^\s/]+/)+[^\s]+"),
    re.compile(
        r"\b(?:api[_ -]?key|apikey|access[_ -]?token|accesstoken|token(?:[_ -]?value)?|secret(?:[_ -]?value)?|password|passwd|authorization(?:[_ -]?header)?|credential(?:[_ -]?value)?|rawprompt|prompt|finalanswer|answer|source[_ -]?path|sourcepath|file[_ -]?path|filepath)\b\s*[:=]",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:owner|user|workspace|tenant|member|customer)[_ -]?(?:id|ids|label|labels|name|names|email|emails)\b\s*[:=]",
        re.IGNORECASE,
    ),
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b", re.IGNORECASE),
    re.compile(r"\bbearer\s+\S+", re.IGNORECASE),
)
SAFE_RUNTIME_EVENT_MESSAGE_REDACTION = "Runtime event message redacted for no-secret projection."
RUNTIME_EVENT_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$")
SENSITIVE_RUNTIME_EVENT_REQUEST_ID_RE = re.compile(
    r"(authorization|bearer|api[-_\s]?key|token|secret|sk-[A-Za-z0-9])",
    re.IGNORECASE,
)


def new_request_id() -> str:
    """Return a short stable-enough request ID for logs and responses."""
    return uuid.uuid4().hex[:12]


def sanitize_runtime_event_request_id(value: Any) -> tuple[str | None, bool, bool]:
    """Return a request ID safe for admin-facing event projections.

    The boolean tuple tail is ``(present, redacted)`` so callers can preserve
    observability without echoing unsafe correlation values.
    """
    text = str(value or "").strip()
    if not text:
        return None, False, False
    text = text[:64]
    if SENSITIVE_RUNTIME_EVENT_REQUEST_ID_RE.search(text):
        return None, True, True
    if not RUNTIME_EVENT_REQUEST_ID_RE.fullmatch(text):
        return None, True, True
    return text, True, False


def runtime_ownership_metadata(ownership: Mapping[str, Any] | None) -> dict[str, Any]:
    """Project owner metadata for runtime events without owner identifiers."""
    ownership = ownership or {}
    return {
        "owner_id_present": bool(str(ownership.get("owner_id", "") or "").strip()),
        "owner_label_present": bool(str(ownership.get("owner_label", "") or "").strip()),
        "ownership_source": str(ownership.get("ownership_source", "") or ""),
    }


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


class ProviderQuotaGuardError(FluxMindError):
    """Raised when the local provider quota/cost guard denies a call."""

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
    """No-secret runtime event for admin/status history."""

    event_id: str
    kind: str
    code: str
    message: str
    created_at: str
    request_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    metadata_redacted_fields: int = 0
    message_redacted: bool = False


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
    safe_message, message_redacted = sanitize_runtime_event_message(message)
    safe_metadata, metadata_redacted = sanitize_runtime_event_metadata(metadata or {})
    safe_request_id, _request_id_present, _request_id_redacted = sanitize_runtime_event_request_id(
        request_id
    )
    event = RuntimeEvent(
        event_id=new_request_id(),
        kind=kind,
        code=code,
        message=safe_message,
        created_at=utc_now(),
        request_id=safe_request_id,
        metadata=safe_metadata,
        metadata_redacted_fields=metadata_redacted,
        message_redacted=message_redacted,
    )
    target = path or RUNTIME_EVENTS_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")
    return event


def _metadata_key_words(key: Any) -> set[str]:
    expanded = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(key).strip())
    normalized = expanded.casefold()
    return {word for word in re.split(r"[^a-z0-9]+", normalized) if word}


def runtime_event_metadata_key_is_sensitive(key: Any) -> bool:
    """Return whether a metadata key should be omitted from admin projections."""
    normalized = str(key).strip().casefold()
    if normalized in SENSITIVE_RUNTIME_EVENT_METADATA_KEYS:
        return True
    words = _metadata_key_words(key)
    if words & SENSITIVE_RUNTIME_EVENT_METADATA_WORDS:
        return True
    if words & SENSITIVE_RUNTIME_EVENT_CONTENT_WORDS and not (
        words & SAFE_CONTENT_METADATA_WORDS
    ):
        return True
    if (words & SENSITIVE_RUNTIME_EVENT_SUBJECT_WORDS) and (
        words & SENSITIVE_RUNTIME_EVENT_IDENTIFIER_WORDS
    ):
        return True
    compact = re.sub(r"[^a-z0-9]+", "", normalized)
    if compact in SENSITIVE_RUNTIME_EVENT_METADATA_COMPACT_KEYS:
        return True
    if "apikey" in compact and not (words & SAFE_CREDENTIAL_METADATA_WORDS):
        return True
    if (
        (
            "token" in words
            or ("token" in compact and not compact.endswith("tokens"))
        )
        and not (words & SAFE_CREDENTIAL_METADATA_WORDS)
    ):
        return True
    if ("credential" in words or "credential" in compact) and not (
        words & SAFE_CREDENTIAL_METADATA_WORDS
    ):
        return True
    if any(word in compact for word in SENSITIVE_RUNTIME_EVENT_CONTENT_WORDS) and not any(
        word in compact for word in SAFE_CONTENT_METADATA_WORDS
    ):
        return True
    if "path" in compact or "url" in compact:
        return True
    if "secret" in compact or "authorization" in compact:
        return True
    return False


def sanitize_runtime_event_metadata(value: Any) -> tuple[Any, int]:
    """Return runtime-event metadata safe for admin-facing projections."""
    redacted = 0
    if isinstance(value, dict):
        clean: dict[str, Any] = {}
        for key, item in value.items():
            if runtime_event_metadata_key_is_sensitive(key):
                redacted += 1
                continue
            clean_value, nested_redacted = sanitize_runtime_event_metadata(item)
            clean[str(key)] = clean_value
            redacted += nested_redacted
        return clean, redacted
    if isinstance(value, list):
        clean_items = []
        for item in value:
            clean_item, nested_redacted = sanitize_runtime_event_metadata(item)
            clean_items.append(clean_item)
            redacted += nested_redacted
        return clean_items, redacted
    return value, redacted


def sanitize_runtime_event_message(message: Any) -> tuple[str, bool]:
    """Return a no-secret event message for admin-facing projections."""
    text = str(message or "")[:500]
    if any(pattern.search(text) for pattern in SENSITIVE_RUNTIME_EVENT_MESSAGE_PATTERNS):
        return SAFE_RUNTIME_EVENT_MESSAGE_REDACTION, True
    return text, False


def runtime_event_to_safe_dict(
    event: RuntimeEvent,
    *,
    include_request_id: bool = True,
) -> dict[str, Any]:
    """Project a runtime event without sensitive metadata values."""
    payload = asdict(event)
    try:
        stored_metadata_redacted = max(0, int(payload.get("metadata_redacted_fields") or 0))
    except (TypeError, ValueError):
        stored_metadata_redacted = 0
    stored_message_redacted = bool(payload.get("message_redacted", False))
    metadata, redacted = sanitize_runtime_event_metadata(payload.get("metadata") or {})
    payload["metadata"] = metadata
    message, message_redacted = sanitize_runtime_event_message(payload.get("message", ""))
    payload["message"] = message
    safe_request_id, request_id_present, request_id_redacted = sanitize_runtime_event_request_id(
        payload.get("request_id")
    )
    if not include_request_id:
        payload.pop("request_id", None)
        payload["request_id_present"] = request_id_present
        if request_id_redacted:
            payload["request_id_redacted"] = True
    elif safe_request_id:
        payload["request_id"] = safe_request_id
    else:
        payload.pop("request_id", None)
        if request_id_present:
            payload["request_id_present"] = True
        if request_id_redacted:
            payload["request_id_redacted"] = True
    total_redacted = stored_metadata_redacted + redacted
    if total_redacted:
        payload["metadata_redacted_fields"] = total_redacted
    else:
        payload.pop("metadata_redacted_fields", None)
    if stored_message_redacted or message_redacted:
        payload["message_redacted"] = True
    else:
        payload.pop("message_redacted", None)
    return payload


def _runtime_event_matches_safe_query(event: RuntimeEvent, query: str) -> bool:
    """Search only the admin-safe projection of a runtime event."""
    safe_event = runtime_event_to_safe_dict(event, include_request_id=True)
    searchable = json.dumps(safe_event, ensure_ascii=False, sort_keys=True).casefold()
    return query in searchable


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
            try:
                event = RuntimeEvent(**item)
            except TypeError:
                logger.warning("runtime_events.invalid_event path=%s line=%s", target, line_number)
                continue
            if query and not _runtime_event_matches_safe_query(event, query):
                continue
            events.append(event)
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
