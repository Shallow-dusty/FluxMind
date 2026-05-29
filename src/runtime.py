"""Runtime helpers for request IDs, logging, and user-facing errors."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass


logger = logging.getLogger("fluxmind")


def new_request_id() -> str:
    """Return a short stable-enough request ID for logs and responses."""
    return uuid.uuid4().hex[:12]


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

    def __init__(self, message: str, *, code: str = "provider_error"):
        super().__init__(code, message, status_code=502)


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

    return UserFacingError(
        "internal_error",
        "FluxMind could not complete the request. Please retry or check logs.",
        500,
    )
