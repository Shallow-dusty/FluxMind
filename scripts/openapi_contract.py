#!/usr/bin/env python3
"""Check FluxMind OpenAPI contract readiness without exporting the raw schema."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._safe_cli import format_cli_error, format_os_error  # noqa: E402
from src.openapi_contract import (  # noqa: E402
    collect_openapi_contract,
    format_openapi_contract_markdown,
    format_openapi_contract_snapshot_verify_markdown,
    verify_openapi_contract_snapshot,
)


def read_json_input(path: str | None) -> dict | None:
    if not path:
        return None
    if path == "-":
        data = json.loads(sys.stdin.read())
    else:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("OpenAPI contract snapshot JSON must be an object")
    return data


def emit_output(output: str, output_path: str | None) -> None:
    if output_path:
        Path(output_path).write_text(output, encoding="utf-8")
    else:
        print(output, end="")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--format",
        choices=("json", "markdown"),
        default="json",
        help="Output format. Defaults to JSON.",
    )
    parser.add_argument("--output", help="Optional output file path.")
    parser.add_argument(
        "--require-local-contract",
        action="store_true",
        help="Exit nonzero unless the local OpenAPI contract is ready.",
    )
    parser.add_argument(
        "--verify-snapshot",
        help="Optional prior no-secret OpenAPI contract JSON report path, or '-' for stdin.",
    )
    parser.add_argument(
        "--require-no-drift",
        action="store_true",
        help="Exit nonzero unless --verify-snapshot matches the current no-secret contract report.",
    )
    args = parser.parse_args()
    if args.require_no_drift and not args.verify_snapshot:
        print("error: --require-no-drift requires --verify-snapshot", file=sys.stderr)
        return 2

    try:
        import api

        current = collect_openapi_contract(api.app.openapi())
        snapshot = read_json_input(args.verify_snapshot)
        status = (
            verify_openapi_contract_snapshot(current, snapshot)
            if snapshot is not None
            else current
        )
        if args.format == "markdown":
            if snapshot is not None:
                output = format_openapi_contract_snapshot_verify_markdown(status) + "\n"
            else:
                output = format_openapi_contract_markdown(status) + "\n"
        else:
            output = json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        emit_output(output, args.output)
    except OSError as exc:
        print(f"error: {format_os_error(exc)}", file=sys.stderr)
        return 2
    except json.JSONDecodeError as exc:
        print(f"error: {format_cli_error(exc)}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"error: {format_cli_error(exc)}", file=sys.stderr)
        return 2

    if args.require_local_contract and not current.get("local_contract_ready"):
        return 1
    if args.require_no_drift and not status.get("ok", False):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
