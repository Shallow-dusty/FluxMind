#!/usr/bin/env python3
"""Check FluxMind local storage schema readiness without exporting contents."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._safe_cli import format_os_error
from src.storage_schema import format_storage_schema_markdown, storage_schema_status_for_root


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
        "--target-root",
        default=str(PROJECT_ROOT),
        help="Project root whose local runtime stores should be checked.",
    )
    args = parser.parse_args()

    try:
        status = storage_schema_status_for_root(Path(args.target_root))
        if args.format == "markdown":
            output = format_storage_schema_markdown(status) + "\n"
        else:
            output = json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        emit_output(output, args.output)
    except OSError as exc:
        print(f"error: {format_os_error(exc)}", file=sys.stderr)
        return 2
    return 0 if status.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
