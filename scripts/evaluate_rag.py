#!/usr/bin/env python3
"""Run FluxMind's offline RAG baseline checks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import evaluate_config, load_eval_config  # noqa: E402

DEFAULT_EVAL_FILE = PROJECT_ROOT / "eval" / "rag_baseline.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        type=Path,
        default=DEFAULT_EVAL_FILE,
        help="Offline evaluation JSON file",
    )
    args = parser.parse_args()

    config = load_eval_config(args.file)
    case_results, provider_results, recorded_results = evaluate_config(config)

    failures: list[str] = []
    for result in case_results:
        status = "ok" if result.ok else "fail"
        print(f"{status:4} eval case {result.case_id}: {result.message}")
        if not result.ok:
            failures.append(f"eval case {result.case_id}")

    for result in provider_results:
        status = "ok" if result.ok else "fail"
        print(
            f"{status:4} provider fixture {result.fixture_id}: "
            f"expected={result.expected_code} actual={result.actual_code}"
        )
        if not result.ok:
            failures.append(f"provider fixture {result.fixture_id}")

    for result in recorded_results:
        status = "ok" if result.ok else "fail"
        print(f"{status:4} recorded answer {result.case_id}: {result.message}")
        if not result.ok:
            failures.append(f"recorded answer {result.case_id}")

    if failures:
        print("\nFailed checks:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
