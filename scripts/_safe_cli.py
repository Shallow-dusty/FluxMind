"""Small helpers for no-secret CLI output."""

from __future__ import annotations

import re


SENSITIVE_CLI_ERROR_PATTERNS = (
    re.compile(r"https?://\S+", re.IGNORECASE),
    re.compile(r"\b[A-Za-z]:\\[^\s'\"]+"),
    re.compile(r"(?<!\w)/(?:[^\s/'\"]+/)+[^\s'\"]+"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b", re.IGNORECASE),
    re.compile(r"\bbearer\s+\S+", re.IGNORECASE),
    re.compile(
        r"\b(?:api[_ -]?key|apikey|access[_ -]?token|accesstoken|token|secret|password|passwd|authorization|credential)\b\s*[:=]\s*\S+",
        re.IGNORECASE,
    ),
)
CLI_ERROR_REDACTION = "[redacted]"


def sanitize_cli_error_message(message: str) -> str:
    text = str(message or "").strip()
    for pattern in SENSITIVE_CLI_ERROR_PATTERNS:
        text = pattern.sub(CLI_ERROR_REDACTION, text)
    text = " ".join(text.split())
    return text[:240]


def format_os_error(exc: BaseException) -> str:
    """Return an exception summary without file paths or raw filenames."""
    message = getattr(exc, "strerror", None) or str(exc) or exc.__class__.__name__
    return sanitize_cli_error_message(message) or exc.__class__.__name__


def format_cli_error(exc: BaseException) -> str:
    """Return a public CLI exception summary without paths, URLs, or tokens."""
    message = str(exc) or exc.__class__.__name__
    return sanitize_cli_error_message(message) or exc.__class__.__name__
