#!/usr/bin/env python3
"""Run a local no-secret FluxMind runtime migration rehearsal."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.storage_migration import (
    format_storage_migration_rehearsal_markdown,
    run_storage_migration_rehearsal,
)


def emit_output(output: str, output_path: str | None) -> None:
    if output_path:
        Path(output_path).write_text(output, encoding="utf-8")
    else:
        print(output, end="")


def _render(status: dict, output_format: str) -> str:
    if output_format == "markdown":
        return format_storage_migration_rehearsal_markdown(status) + "\n"
    return json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


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
        help="Project root whose runtime state should be rehearsed.",
    )
    parser.add_argument(
        "--staging-root",
        help="Optional staging root to retain. Defaults to a temporary directory that is cleaned up.",
    )
    parser.add_argument(
        "--overwrite-staging",
        action="store_true",
        help="Allow clearing an existing staging root before the rehearsal.",
    )
    parser.add_argument(
        "--include-runtime-dependencies",
        action="store_true",
        help="Also copy runtime_dependency groups such as local models.",
    )
    args = parser.parse_args()

    try:
        if args.staging_root:
            status = run_storage_migration_rehearsal(
                project_root=Path(args.target_root),
                staging_root=Path(args.staging_root),
                overwrite_staging=args.overwrite_staging,
                include_runtime_dependencies=args.include_runtime_dependencies,
            )
            output = _render(status, args.format)
            emit_output(output, args.output)
            return 0 if status.get("rehearsal_ok") else 1

        with tempfile.TemporaryDirectory(prefix="fluxmind-migration-rehearsal-") as tmp_dir:
            status = run_storage_migration_rehearsal(
                project_root=Path(args.target_root),
                staging_root=Path(tmp_dir),
                overwrite_staging=False,
                include_runtime_dependencies=args.include_runtime_dependencies,
            )
            status["staging_root_retained"] = False
            output = _render(status, args.format)
            emit_output(output, args.output)
            return 0 if status.get("rehearsal_ok") else 1
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
