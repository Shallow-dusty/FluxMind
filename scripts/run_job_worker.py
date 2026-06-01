#!/usr/bin/env python3
"""Run FluxMind's explicit local durable job worker."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.jobs import LocalDurableJobWorker  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-id", help="Stable worker ID to persist on claimed jobs")
    parser.add_argument("--lease-seconds", type=int, default=3600)
    parser.add_argument("--max-jobs", type=int, default=1, help="Maximum jobs to run before exiting")
    parser.add_argument("--loop", action="store_true", help="Keep polling until --max-jobs is reached")
    parser.add_argument("--forever", action="store_true", help="Poll forever; intended for systemd worker services")
    parser.add_argument("--poll-interval-s", type=float, default=2.0)
    args = parser.parse_args()

    max_jobs = None if args.forever else args.max_jobs
    worker = LocalDurableJobWorker(
        worker_id=args.worker_id,
        lease_seconds=args.lease_seconds,
    )
    if args.loop or args.forever:
        results = worker.run_polling(
            poll_interval_s=args.poll_interval_s,
            max_jobs=max_jobs,
        )
    else:
        results = worker.run_until_empty(max_jobs=max_jobs)

    for record in results:
        print(
            f"job_id={record.job_id} kind={record.kind} "
            f"status={record.status} worker_id={record.worker_id}"
        )
    if not results:
        print("no_due_jobs=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
