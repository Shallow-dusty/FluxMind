"""Small helpers for consistent CLI error output."""

from __future__ import annotations


def format_error(exc: BaseException) -> str:
    return str(exc).strip() or exc.__class__.__name__
