#!/usr/bin/env python3
"""Run the no-secret local FluxMind activation suite."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._safe_cli import format_os_error  # noqa: E402
from src.activation_suite import (  # noqa: E402
    collect_activation_suite,
    format_activation_suite_markdown,
)


def _load_openapi_schema() -> dict:
    import api

    return api.app.openapi()


def emit_output(output: str, output_path: str | None) -> None:
    if output_path:
        Path(output_path).write_text(output, encoding="utf-8")
    else:
        print(output, end="")


def _required_target_ready(status: dict, target: str | None) -> bool:
    if not target:
        return bool(status.get("local_foundation_ready"))
    target_key = {
        "local_foundation": "local_foundation_ready",
        "small_group": "small_group_ready",
        "community": "community_ready",
        "full_activation": "full_activation_ready",
    }[target]
    return bool(status.get(target_key))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        help="Optional suite state root. Defaults to a temporary directory that is removed after the run.",
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Project root whose local runtime state should be rehearsed.",
    )
    parser.add_argument(
        "--eval-file",
        type=Path,
        help="Evaluation config file. Defaults to --target-root/eval/rag_baseline.json.",
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
        choices=("local_foundation", "small_group", "community", "full_activation"),
        help="Exit nonzero unless the selected suite target is ready.",
    )
    args = parser.parse_args()

    try:
        eval_file = args.eval_file or (args.target_root / "eval" / "rag_baseline.json")
        status = collect_activation_suite(
            root=args.root,
            project_root=args.target_root,
            eval_file=eval_file,
            live_report_paths=args.live_report,
            openapi_schema=_load_openapi_schema(),
        )
        if args.format == "markdown":
            output = format_activation_suite_markdown(status) + "\n"
        else:
            output = json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        emit_output(output, args.output)
    except OSError as exc:
        print(f"error: {format_os_error(exc)}", file=sys.stderr)
        return 2
    except json.JSONDecodeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    return 0 if _required_target_ready(status, args.require_target) else 1


if __name__ == "__main__":
    raise SystemExit(main())
