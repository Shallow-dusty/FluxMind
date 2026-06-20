#!/usr/bin/env python3
"""Manage the local no-secret FluxMind share-link registry."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._safe_cli import format_os_error, sanitize_cli_error_message  # noqa: E402
from src.share_links import (  # noqa: E402
    LocalShareLinkRegistry,
    share_link_registry_backend_status,
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
    lines = ["# FluxMind Share Link Registry", ""]
    if "token" in payload:
        link = payload.get("share_link", {}) or {}
        lines.extend(
            [
                "## Created Share Link",
                "",
                f"- Link ID: {link.get('link_id', '')}",
                f"- Workspace present: {str(link.get('workspace_present', False)).lower()}",
                f"- Workspace fingerprint: {link.get('workspace_fingerprint', '')}",
                f"- Resource kind: {link.get('resource_kind', '')}",
                "- Token: shown once in JSON output only",
                f"- Share token exported: {str(link.get('share_token_exported', False)).lower()}",
            ]
        )
    elif "share_links" in payload:
        lines.extend(["## Share Links", ""])
        links = payload.get("share_links", [])
        if not links:
            lines.append("- none")
        for link in links:
            lines.append(
                f"- {link.get('link_id', '')}: "
                f"workspace_present={str(link.get('workspace_present', False)).lower()}, "
                f"kind={link.get('resource_kind', '')}, "
                f"active={str(link.get('active', False)).lower()}, "
                f"redeem_count={link.get('redeem_count', 0)}"
            )
    elif "resolution" in payload:
        resolution = payload.get("resolution", {}) or {}
        link = resolution.get("share_link", {}) or {}
        lines.extend(
            [
                "## Resolution",
                "",
                f"- Valid: {str(resolution.get('valid', False)).lower()}",
                f"- Reason: {resolution.get('reason', '')}",
                f"- Link ID: {link.get('link_id', '')}",
                f"- Resource kind: {link.get('resource_kind', '')}",
                f"- Resource fingerprint: {link.get('resource_ref_fingerprint', '')}",
                f"- Share token exported: {str(resolution.get('share_token_exported', False)).lower()}",
            ]
        )
    elif "share_link" in payload:
        link = payload.get("share_link", {}) or {}
        lines.extend(
            [
                "## Share Link",
                "",
                f"- Link ID: {link.get('link_id', '')}",
                f"- Active: {str(link.get('active', False)).lower()}",
                f"- Revoked at: {link.get('revoked_at', '') or 'none'}",
                f"- Share token exported: {str(link.get('share_token_exported', False)).lower()}",
            ]
        )
    else:
        lines.extend(
            [
                f"- Backend: {payload.get('backend', '')}",
                f"- Available: {str(payload.get('available', False)).lower()}",
                f"- Active links: {payload.get('active_link_count', 0)}",
                f"- Revoked links: {payload.get('revoked_link_count', 0)}",
                f"- Expired links: {payload.get('expired_link_count', 0)}",
                f"- Total links: {payload.get('total_link_count', 0)}",
                f"- Secrets exported: {str(payload.get('secrets_exported', False)).lower()}",
                f"- Share tokens exported: {str(payload.get('share_tokens_exported', False)).lower()}",
                f"- Share URLs exported: {str(payload.get('share_urls_exported', False)).lower()}",
            ]
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, help="Optional share-link SQLite path.")
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    parser.add_argument("--output", help="Optional output file path.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create")
    add_output_args(create)
    create.add_argument("--workspace-id", required=True)
    create.add_argument("--created-by-user-id", required=True)
    create.add_argument("--resource-kind", default="corpus_profile")
    create.add_argument("--resource-ref", required=True)
    create.add_argument("--description", default="")
    create.add_argument("--expires-in-s", type=int)
    create.add_argument("--max-redemptions", type=int, default=0)

    list_cmd = subparsers.add_parser("list")
    add_output_args(list_cmd)
    list_cmd.add_argument("--workspace-id")
    list_cmd.add_argument("--include-revoked", action="store_true")
    list_cmd.add_argument("--limit", type=int, default=50)

    revoke = subparsers.add_parser("revoke")
    add_output_args(revoke)
    revoke.add_argument("link_id")

    resolve = subparsers.add_parser("resolve")
    add_output_args(resolve)
    resolve.add_argument("token")
    resolve.add_argument("--record-redeem", action="store_true")

    status_cmd = subparsers.add_parser("status")
    add_output_args(status_cmd)

    args = parser.parse_args()
    if args.command == "create" and args.format != "json":
        print(
            "error: create outputs the one-time share token and therefore requires --format json",
            file=sys.stderr,
        )
        return 2

    registry = LocalShareLinkRegistry(db_path=args.db)
    try:
        if args.command == "create":
            payload = registry.create_link(
                workspace_id=args.workspace_id,
                created_by_user_id=args.created_by_user_id,
                resource_kind=args.resource_kind,
                resource_ref=args.resource_ref,
                description=args.description,
                expires_in_s=args.expires_in_s,
                max_redemptions=args.max_redemptions,
            )
        elif args.command == "list":
            payload = {
                "share_links": [
                    record.to_public_dict()
                    for record in registry.list_links(
                        workspace_id=args.workspace_id,
                        include_revoked=args.include_revoked,
                        limit=args.limit,
                    )
                ],
                "content_exported": False,
                "secrets_exported": False,
                "share_tokens_exported": False,
                "share_urls_exported": False,
            }
        elif args.command == "revoke":
            record = registry.revoke_link(args.link_id)
            if record is None:
                payload = {
                    "ok": False,
                    "reason": "share_link_not_found",
                    "secrets_exported": False,
                    "share_tokens_exported": False,
                }
            else:
                payload = {
                    "ok": True,
                    "share_link": record.to_public_dict(),
                    "secrets_exported": False,
                    "share_tokens_exported": False,
                    "share_urls_exported": False,
                }
        elif args.command == "resolve":
            payload = {
                "resolution": registry.resolve_token(
                    args.token,
                    record_redeem=args.record_redeem,
                )
            }
        else:
            payload = share_link_registry_backend_status(
                backend="sqlite" if args.db else None,
                db_path=args.db,
            )
    except ValueError as exc:
        print(f"error: {sanitize_cli_error_message(str(exc))}", file=sys.stderr)
        return 2
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

    if args.command == "resolve":
        return 0 if payload.get("resolution", {}).get("valid") else 1
    if args.command == "revoke":
        return 0 if payload.get("ok") else 1
    if args.command == "status":
        return 0 if payload.get("available") or not payload.get("configured") else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
