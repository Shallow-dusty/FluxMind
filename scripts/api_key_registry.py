#!/usr/bin/env python3
"""Manage the local no-secret FluxMind API key registry."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._safe_cli import format_os_error  # noqa: E402
from src.api_keys import (  # noqa: E402
    LocalApiKeyRegistry,
    api_key_registry_backend_status,
)


def add_output_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--format", choices=("json", "markdown"), default=argparse.SUPPRESS)
    parser.add_argument("--output", default=argparse.SUPPRESS)


def emit_output(output: str, output_path: str | None) -> None:
    if output_path:
        Path(output_path).write_text(output, encoding="utf-8")
    else:
        print(output, end="")


def render_markdown(payload: dict) -> str:
    lines = ["# FluxMind API Key Registry", ""]
    if "token" in payload:
        lines.extend(
            [
                "## Created Key",
                "",
                f"- Key ID: {payload.get('key', {}).get('key_id', '')}",
                f"- Owner fingerprint: {payload.get('key', {}).get('owner_id_fingerprint', '')}",
                "- Token: shown once in JSON output only",
            ]
        )
    elif "keys" in payload:
        lines.extend(["## Keys", ""])
        keys = payload.get("keys", [])
        if not keys:
            lines.append("- none")
        for key in keys:
            lines.append(
                f"- {key.get('key_id', '')}: owner_fingerprint={key.get('owner_id_fingerprint', '')}, "
                f"active={str(key.get('active', False)).lower()}, "
                f"use_count={key.get('use_count', 0)}"
            )
    else:
        lines.extend(
            [
                f"- Backend: {payload.get('backend', '')}",
                f"- Available: {str(payload.get('available', False)).lower()}",
                f"- Active keys: {payload.get('active_key_count', 0)}",
                f"- Revoked keys: {payload.get('revoked_key_count', 0)}",
                f"- Secrets exported: {str(payload.get('secrets_exported', False)).lower()}",
            ]
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, help="Optional registry SQLite path.")
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    parser.add_argument("--output", help="Optional output file path.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create")
    add_output_args(create)
    create.add_argument("--owner-id")
    create.add_argument("--owner-label")
    create.add_argument("--description", default="")

    list_cmd = subparsers.add_parser("list")
    add_output_args(list_cmd)
    list_cmd.add_argument("--include-revoked", action="store_true")

    revoke = subparsers.add_parser("revoke")
    add_output_args(revoke)
    revoke.add_argument("key_id")

    verify = subparsers.add_parser("verify")
    add_output_args(verify)
    verify.add_argument("token")

    status_cmd = subparsers.add_parser("status")
    add_output_args(status_cmd)

    args = parser.parse_args()
    if args.command == "create" and args.format != "json":
        print(
            "error: create outputs the one-time token and therefore requires --format json",
            file=sys.stderr,
        )
        return 2
    registry = LocalApiKeyRegistry(db_path=args.db)

    try:
        if args.command == "create":
            payload = registry.create_key(
                owner_id=args.owner_id,
                owner_label=args.owner_label,
                description=args.description,
            )
        elif args.command == "list":
            payload = {
                "keys": [
                    record.to_public_dict()
                    for record in registry.list_keys(include_revoked=args.include_revoked)
                ],
                "content_exported": False,
                "secrets_exported": False,
            }
        elif args.command == "revoke":
            record = registry.revoke_key(args.key_id)
            if record is None:
                payload = {"ok": False, "reason": "key_not_found", "secrets_exported": False}
            else:
                payload = {"ok": True, "key": record.to_public_dict(), "secrets_exported": False}
        elif args.command == "verify":
            record = registry.verify_token(args.token)
            if record is None:
                payload = {"valid": False, "secrets_exported": False}
            else:
                payload = {"valid": True, "key": record.to_public_dict(), "secrets_exported": False}
        else:
            payload = api_key_registry_backend_status(
                backend="sqlite" if args.db else None,
                db_path=args.db,
            )
    except (OSError, sqlite3.Error) as exc:
        print(f"error: {format_os_error(exc)}", file=sys.stderr)
        return 2

    if args.format == "markdown":
        output = render_markdown(payload) + "\n"
    else:
        output = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    try:
        emit_output(output, args.output)
    except OSError as exc:
        print(f"error: {format_os_error(exc)}", file=sys.stderr)
        return 2

    if args.command == "verify":
        return 0 if payload.get("valid") else 1
    if args.command == "revoke":
        return 0 if payload.get("ok") else 1
    if args.command == "status":
        return 0 if payload.get("available") or not payload.get("configured") else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
