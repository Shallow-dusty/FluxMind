#!/usr/bin/env python3
"""Run a no-secret local provider runtime rehearsal."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._safe_cli import format_os_error  # noqa: E402
from src.provider_runtime_rehearsal import (  # noqa: E402
    collect_provider_runtime_rehearsal,
    format_provider_runtime_rehearsal_markdown,
)


def emit_output(output: str, output_path: str | None) -> None:
    if output_path:
        Path(output_path).write_text(output, encoding="utf-8")
    else:
        print(output, end="")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        help="Optional rehearsal root. Defaults to a temporary directory that is removed after the run.",
    )
    parser.add_argument(
        "--format",
        choices=("json", "markdown"),
        default="json",
        help="Output format. Defaults to JSON.",
    )
    parser.add_argument("--output", help="Optional output file path.")
    parser.add_argument(
        "--require-local-foundation",
        action="store_true",
        help="Exit nonzero unless the local provider runtime rehearsal passes.",
    )
    args = parser.parse_args()

    try:
        status = collect_provider_runtime_rehearsal(root=args.root)
        if args.format == "markdown":
            output = format_provider_runtime_rehearsal_markdown(status) + "\n"
        else:
            output = json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        emit_output(output, args.output)
    except OSError as exc:
        print(f"error: {format_os_error(exc)}", file=sys.stderr)
        return 2

    if args.require_local_foundation and not status.get("ok"):
        return 1
    return 0 if status.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
