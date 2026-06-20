#!/usr/bin/env python3
"""Manage the local no-secret FluxMind product registry."""

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
from src.product_registry import (  # noqa: E402
    LocalProductRegistry,
    product_registry_backend_status,
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
    lines = ["# FluxMind Product Registry", ""]
    if "workspace" in payload:
        workspace = payload.get("workspace", {})
        lines.extend(
            [
                "## Workspace",
                "",
                f"- Workspace ID: {workspace.get('workspace_id', '')}",
                f"- Owner user ID: {workspace.get('owner_user_id', '')}",
                f"- Secrets exported: {str(payload.get('secrets_exported', False)).lower()}",
            ]
        )
    elif "member" in payload:
        member = payload.get("member", {})
        lines.extend(
            [
                "## Member",
                "",
                f"- Workspace ID: {member.get('workspace_id', '')}",
                f"- User ID: {member.get('user_id', '')}",
                f"- Role: {member.get('role', '')}",
                f"- Secrets exported: {str(payload.get('secrets_exported', False)).lower()}",
            ]
        )
    elif "permission" in payload:
        decision = payload.get("permission", {})
        lines.extend(
            [
                "## Permission",
                "",
                f"- Allowed: {str(decision.get('allowed', False)).lower()}",
                f"- Reason: {decision.get('reason', '')}",
                f"- Workspace ID: {decision.get('workspace_id', '')}",
                f"- User ID: {decision.get('user_id', '')}",
                f"- Role: {decision.get('role', '')}",
                f"- Action: {decision.get('action', '')}",
                f"- Required roles: {', '.join(decision.get('required_roles', [])) or 'none'}",
                f"- Secrets exported: {str(payload.get('secrets_exported', False)).lower()}",
            ]
        )
    elif "usage_event" in payload:
        event = payload.get("usage_event", {})
        lines.extend(
            [
                "## Usage Event",
                "",
                f"- Event ID: {event.get('event_id', '')}",
                f"- Workspace ID: {event.get('workspace_id', '')}",
                f"- Metric: {event.get('metric', '')}",
                f"- Amount: {event.get('amount', 0)}",
                f"- Secrets exported: {str(payload.get('secrets_exported', False)).lower()}",
            ]
        )
    elif "workspaces" in payload:
        lines.extend(["## Workspaces", ""])
        workspaces = payload.get("workspaces", [])
        if not workspaces:
            lines.append("- none")
        for workspace in workspaces:
            lines.append(
                f"- {workspace.get('workspace_id', '')}: "
                f"owner={workspace.get('owner_user_id', '')}, "
                f"status={workspace.get('status', '')}"
            )
    else:
        lines.extend(
            [
                f"- Backend: {payload.get('backend', '')}",
                f"- Available: {str(payload.get('available', False)).lower()}",
                f"- Users: {payload.get('user_count', 0)}",
                f"- Workspaces: {payload.get('workspace_count', 0)}",
                f"- RBAC available: {str(payload.get('rbac_available', False)).lower()}",
                f"- Quota limits: {payload.get('quota_limit_count', 0)}",
                f"- Usage events: {payload.get('usage_event_count', 0)}",
                f"- Billing accounts: {payload.get('billing_account_count', 0)}",
                f"- Secrets exported: {str(payload.get('secrets_exported', False)).lower()}",
            ]
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, help="Optional product registry SQLite path.")
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    parser.add_argument("--output", help="Optional output file path.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    status_cmd = subparsers.add_parser("status")
    add_output_args(status_cmd)

    bootstrap = subparsers.add_parser("bootstrap-local")
    add_output_args(bootstrap)
    bootstrap.add_argument("--user-id", default="local-user")
    bootstrap.add_argument("--user-label", default="Local user")
    bootstrap.add_argument("--workspace-id", default="local-workspace")
    bootstrap.add_argument("--workspace-label", default="Local workspace")
    bootstrap.add_argument("--quota-metric", default="requests")
    bootstrap.add_argument("--quota-limit", type=int, default=1000)
    bootstrap.add_argument("--quota-window-s", type=int, default=86400)
    bootstrap.add_argument("--billing-mode", default="local-ledger")

    quota = subparsers.add_parser("set-quota")
    add_output_args(quota)
    quota.add_argument("--workspace-id", required=True)
    quota.add_argument("--metric", required=True)
    quota.add_argument("--limit", type=int, required=True)
    quota.add_argument("--window-s", type=int, required=True)

    usage = subparsers.add_parser("record-usage")
    add_output_args(usage)
    usage.add_argument("--workspace-id", required=True)
    usage.add_argument("--user-id", required=True)
    usage.add_argument("--metric", required=True)
    usage.add_argument("--amount", type=int, required=True)
    usage.add_argument("--source", default="manual")

    member = subparsers.add_parser("add-member")
    add_output_args(member)
    member.add_argument("--workspace-id", required=True)
    member.add_argument("--user-id", required=True)
    member.add_argument("--user-label")
    member.add_argument("--role", choices=("owner", "admin", "member", "viewer"), default="member")

    permission = subparsers.add_parser("check-permission")
    add_output_args(permission)
    permission.add_argument("--workspace-id")
    permission.add_argument("--user-id", required=True)
    permission.add_argument("--action", required=True)

    list_cmd = subparsers.add_parser("list-workspaces")
    add_output_args(list_cmd)

    args = parser.parse_args()
    registry = LocalProductRegistry(db_path=args.db)

    try:
        if args.command == "status":
            payload = product_registry_backend_status(
                backend="sqlite" if args.db else None,
                db_path=args.db,
            )
        elif args.command == "bootstrap-local":
            workspace = registry.create_workspace(
                workspace_id=args.workspace_id,
                label=args.workspace_label,
                owner_user_id=args.user_id,
                owner_label=args.user_label,
            )
            quota_payload = registry.set_quota(
                workspace_id=workspace.workspace_id,
                metric=args.quota_metric,
                limit_value=args.quota_limit,
                window_s=args.quota_window_s,
            )
            billing = registry.set_billing_account(
                workspace_id=workspace.workspace_id,
                billing_mode=args.billing_mode,
                attribution_enabled=True,
            )
            payload = {
                "workspace": workspace.to_public_dict(),
                "quota": quota_payload,
                "billing": billing,
                "content_exported": False,
                "secrets_exported": False,
            }
        elif args.command == "set-quota":
            payload = {
                "quota": registry.set_quota(
                    workspace_id=args.workspace_id,
                    metric=args.metric,
                    limit_value=args.limit,
                    window_s=args.window_s,
                ),
                "content_exported": False,
                "secrets_exported": False,
            }
        elif args.command == "record-usage":
            payload = {
                "usage_event": registry.record_usage(
                    workspace_id=args.workspace_id,
                    user_id=args.user_id,
                    metric=args.metric,
                    amount=args.amount,
                    source=args.source,
                ),
                "content_exported": False,
                "secrets_exported": False,
            }
        elif args.command == "add-member":
            registry.add_member(
                workspace_id=args.workspace_id,
                user_id=args.user_id,
                label=args.user_label,
                role=args.role,
            )
            payload = {
                "member": {
                    "workspace_id": args.workspace_id,
                    "user_id": args.user_id,
                    "role": args.role,
                },
                "content_exported": False,
                "secrets_exported": False,
            }
        elif args.command == "check-permission":
            payload = {
                "permission": registry.permission_decision(
                    workspace_id=args.workspace_id,
                    user_id=args.user_id,
                    action=args.action,
                ),
                "content_exported": False,
                "secrets_exported": False,
            }
        else:
            payload = {
                "workspaces": [workspace.to_public_dict() for workspace in registry.list_workspaces()],
                "content_exported": False,
                "secrets_exported": False,
            }
    except (OSError, sqlite3.Error, ValueError) as exc:
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

    if args.command == "status":
        return 0 if payload.get("available") or not payload.get("configured") else 1
    if args.command == "check-permission":
        return 0 if payload.get("permission", {}).get("allowed", False) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
