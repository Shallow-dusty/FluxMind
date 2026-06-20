#!/usr/bin/env python3
"""Run the no-secret FluxMind provider activation readiness check."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._safe_cli import format_os_error  # noqa: E402
from src.provider_readiness import (  # noqa: E402
    collect_provider_readiness,
    format_provider_readiness_markdown,
)


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
        "--require-activation",
        action="store_true",
        help="Exit nonzero unless real provider activation targets are ready.",
    )
    args = parser.parse_args()

    try:
        status = collect_provider_readiness()
        if args.format == "markdown":
            output = format_provider_readiness_markdown(status) + "\n"
        else:
            output = json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        emit_output(output, args.output)
    except OSError as exc:
        print(f"error: {format_os_error(exc)}", file=sys.stderr)
        return 2

    if not status.get("local_foundation_ready"):
        return 1
    if args.require_activation and not status.get("activation_ready"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
