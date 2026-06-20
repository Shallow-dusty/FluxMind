#!/usr/bin/env python3
"""Run the no-secret FluxMind quality maturity readiness check."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._safe_cli import format_os_error  # noqa: E402
from src.quality_readiness import (  # noqa: E402
    collect_quality_readiness,
    format_quality_readiness_markdown,
)


def emit_output(output: str, output_path: str | None) -> None:
    if output_path:
        Path(output_path).write_text(output, encoding="utf-8")
    else:
        print(output, end="")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        type=Path,
        default=PROJECT_ROOT / "eval" / "rag_baseline.json",
        help="Evaluation config file. Defaults to eval/rag_baseline.json.",
    )
    parser.add_argument(
        "--live-report",
        action="append",
        type=Path,
        default=[],
        help="Optional no-secret JSON report from scripts/evaluate_rag.py --json-report.",
    )
    parser.add_argument(
        "--format",
        choices=("json", "markdown"),
        default="json",
        help="Output format. Defaults to JSON.",
    )
    parser.add_argument("--output", help="Optional output file path.")
    parser.add_argument(
        "--require-target",
        choices=("self_use", "small_group", "community"),
        help="Exit nonzero unless the selected maturity target is ready.",
    )
    args = parser.parse_args()

    try:
        status = collect_quality_readiness(
            eval_file=args.file,
            live_report_paths=args.live_report,
        )
        if args.format == "markdown":
            output = format_quality_readiness_markdown(status) + "\n"
        else:
            output = json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        emit_output(output, args.output)
    except OSError as exc:
        print(f"error: {format_os_error(exc)}", file=sys.stderr)
        return 2
    except json.JSONDecodeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if not status.get("local_foundation_ready"):
        return 1
    if args.require_target:
        target_key = {
            "self_use": "local_foundation_ready",
            "small_group": "small_group_ready",
            "community": "community_ready",
        }[args.require_target]
        if not status.get(target_key):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
