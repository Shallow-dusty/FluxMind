#!/usr/bin/env python3
"""Safely sync FluxMind source to the deployed host.

The script is dry-run by default. Use --apply to perform the rsync.
Runtime state, secrets, models, and virtual environments are always excluded.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._safe_cli import format_os_error  # noqa: E402

DEFAULT_HOST = "root@100.100.233.26"
DEFAULT_REMOTE_PATH = "/opt/fluxmind/"
DEFAULT_RESTART_SERVICES = (
    "fluxmind-api.service",
    "fluxmind-ui.service",
    "fluxmind-worker.service",
)

DEPLOY_EXCLUDES = (
    ".git/",
    ".venv/",
    "venv/",
    "__pycache__/",
    ".pytest_cache/",
    ".mypy_cache/",
    ".ruff_cache/",
    ".coverage",
    ".env",
    ".cache/",
    "models/",
    "metadata/",
    "jobs/",
    "artifacts/",
    "papers/",
    "faiss_index/",
)

REQUIRED_RUNTIME_EXCLUDES = {
    ".env",
    ".coverage",
    ".cache/",
    "venv/",
    "models/",
    "metadata/",
    "jobs/",
    "artifacts/",
    "papers/",
    "faiss_index/",
}


def validate_project_root(root: Path = PROJECT_ROOT) -> None:
    required = ["api.py", "app.py", "src", "scripts", "docs"]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise RuntimeError(f"Not a FluxMind project root: missing {', '.join(missing)}")


def validate_excludes(excludes: tuple[str, ...] = DEPLOY_EXCLUDES) -> None:
    missing = sorted(REQUIRED_RUNTIME_EXCLUDES - set(excludes))
    if missing:
        raise RuntimeError(f"Refusing deploy sync; missing runtime excludes: {', '.join(missing)}")


def build_rsync_command(
    *,
    host: str,
    remote_path: str,
    apply: bool,
    root: Path = PROJECT_ROOT,
) -> list[str]:
    validate_project_root(root)
    validate_excludes()
    command = ["rsync", "-az", "--delete", "--itemize-changes", "--human-readable"]
    if not apply:
        command.append("--dry-run")
    for pattern in DEPLOY_EXCLUDES:
        command.extend(["--exclude", pattern])
    command.extend([f"{root.resolve().as_posix()}/", f"{host}:{remote_path}"])
    return command


def build_restart_command(*, services: tuple[str, ...]) -> str:
    service_list = " ".join(services)
    return f"systemctl restart {service_list} && systemctl is-active {service_list}"


def run_command(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(command))
    return subprocess.run(command, check=check, text=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--remote-path", default=DEFAULT_REMOTE_PATH)
    parser.add_argument("--apply", action="store_true", help="Perform the rsync; default is dry-run")
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Restart FluxMind API/UI/worker after a successful --apply sync",
    )
    parser.add_argument(
        "--service",
        action="append",
        default=[],
        help="Service to restart when --restart is set. Defaults to API/UI/worker.",
    )
    args = parser.parse_args()

    try:
        rsync_command = build_rsync_command(
            host=args.host,
            remote_path=args.remote_path,
            apply=args.apply,
        )
    except RuntimeError as exc:
        print(f"error: {format_os_error(exc)}", file=sys.stderr)
        return 2

    if not args.apply:
        print("dry_run=1 use --apply to sync")
    run_command(rsync_command)

    if args.restart:
        if not args.apply:
            print("skip_restart=1 dry-run mode")
            return 0
        services = tuple(args.service) if args.service else DEFAULT_RESTART_SERVICES
        run_command(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                args.host,
                build_restart_command(services=services),
            ]
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
