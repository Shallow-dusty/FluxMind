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

from scripts._safe_cli import format_os_error
from src.storage_migration import (
    format_job_store_migration_verify_markdown,
    format_object_storage_migration_verify_markdown,
    format_storage_migration_rehearsal_markdown,
    run_storage_migration_rehearsal,
    verify_job_store_migration_manifest,
    verify_object_storage_migration_manifest,
)


def emit_output(output: str, output_path: str | None) -> None:
    if output_path:
        Path(output_path).write_text(output, encoding="utf-8")
    else:
        print(output, end="")


def _render(status: dict, output_format: str) -> str:
    if output_format == "markdown":
        if status.get("mode") == "job_store_migration_manifest_verify":
            return format_job_store_migration_verify_markdown(status) + "\n"
        if status.get("mode") == "object_storage_migration_manifest_verify":
            return format_object_storage_migration_verify_markdown(status) + "\n"
        return format_storage_migration_rehearsal_markdown(status) + "\n"
    return json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _read_json(path: str) -> dict:
    if path == "-":
        return json.loads(sys.stdin.read())
    return json.loads(Path(path).read_text(encoding="utf-8"))


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
    parser.add_argument(
        "--include-object-manifest",
        action="store_true",
        help="Include an opaque object-storage migration manifest for staged runtime files.",
    )
    parser.add_argument(
        "--include-job-store-manifest",
        action="store_true",
        help="Include a no-secret durable job-store migration manifest for staged job state.",
    )
    parser.add_argument(
        "--object-key-prefix",
        default="fluxmind-runtime",
        help="Opaque object-key prefix for --include-object-manifest output.",
    )
    parser.add_argument(
        "--verify-object-manifest",
        metavar="PATH",
        help="Verify an opaque object-storage migration manifest JSON against --target-root. Use '-' for stdin.",
    )
    parser.add_argument(
        "--verify-job-store-manifest",
        metavar="PATH",
        help="Verify a no-secret job-store migration manifest JSON against --target-root. Use '-' for stdin.",
    )
    args = parser.parse_args()

    try:
        if args.verify_object_manifest:
            manifest = _read_json(args.verify_object_manifest)
            status = verify_object_storage_migration_manifest(
                manifest,
                project_root=Path(args.target_root),
                include_runtime_dependencies=True
                if args.include_runtime_dependencies
                else None,
            )
            output = _render(status, args.format)
            emit_output(output, args.output)
            return 0 if status.get("ok") else 1

        if args.verify_job_store_manifest:
            manifest = _read_json(args.verify_job_store_manifest)
            status = verify_job_store_migration_manifest(
                manifest,
                project_root=Path(args.target_root),
            )
            output = _render(status, args.format)
            emit_output(output, args.output)
            return 0 if status.get("ok") else 1

        if args.staging_root:
            status = run_storage_migration_rehearsal(
                project_root=Path(args.target_root),
                staging_root=Path(args.staging_root),
                overwrite_staging=args.overwrite_staging,
                include_runtime_dependencies=args.include_runtime_dependencies,
                include_object_manifest=args.include_object_manifest,
                include_job_store_manifest=args.include_job_store_manifest,
                object_key_prefix=args.object_key_prefix,
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
                include_object_manifest=args.include_object_manifest,
                include_job_store_manifest=args.include_job_store_manifest,
                object_key_prefix=args.object_key_prefix,
            )
            status["staging_root_retained"] = False
            output = _render(status, args.format)
            emit_output(output, args.output)
            return 0 if status.get("rehearsal_ok") else 1
    except OSError as exc:
        print(f"error: {format_os_error(exc)}", file=sys.stderr)
        return 2
    except json.JSONDecodeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
