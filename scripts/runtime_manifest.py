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
    collect_runtime_restore_check,
    format_runtime_backup_manifest_markdown,
    format_runtime_restore_check_markdown,
)


def load_json_manifest(path_arg: str) -> dict:
    if path_arg == "-":
        return json.load(sys.stdin)
    return json.loads(Path(path_arg).read_text(encoding="utf-8"))


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
        "--restore-check",
        metavar="MANIFEST_JSON",
        help="Dry-run verify this no-secret runtime manifest against a target root. Use '-' for stdin.",
    )
    parser.add_argument(
        "--target-root",
        default=str(PROJECT_ROOT),
        help="Root for resolving project-relative manifest paths during --restore-check.",
    )
    args = parser.parse_args()

    try:
        if args.restore_check:
            manifest = load_json_manifest(args.restore_check)
            restore_check = collect_runtime_restore_check(
                manifest,
                project_root=Path(args.target_root),
            )
            if args.format == "markdown":
                output = format_runtime_restore_check_markdown(restore_check) + "\n"
            else:
                output = json.dumps(
                    restore_check,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                ) + "\n"
            emit_output(output, args.output)
            return 0 if restore_check.get("ok") else 1

        manifest = collect_runtime_backup_manifest()
        if args.format == "markdown":
            output = format_runtime_backup_manifest_markdown(manifest) + "\n"
        else:
            output = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        emit_output(output, args.output)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
