#!/usr/bin/env python3
"""Print a no-secret backup manifest for FluxMind runtime state."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.storage_manifest import (
    collect_runtime_backup_manifest,
    format_runtime_backup_manifest_markdown,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--format",
        choices=("json", "markdown"),
        default="json",
        help="Output format. Defaults to JSON.",
    )
    parser.add_argument("--output", help="Optional output file path.")
    args = parser.parse_args()

    manifest = collect_runtime_backup_manifest()
    if args.format == "markdown":
        output = format_runtime_backup_manifest_markdown(manifest) + "\n"
    else:
        output = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"

    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
    else:
        print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
