#!/usr/bin/env python3
"""FluxMind local/remote health checks.

The default mode is local and side-effect free: it verifies required files,
workspace numbering, importability, and optional local index metadata. Use
`--url` to add HTTP checks for deployed UI/API endpoints.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def check(condition: bool, label: str, failures: list[str]) -> None:
    status = "ok" if condition else "fail"
    print(f"{status:4} {label}")
    if not condition:
        failures.append(label)


def http_status(url: str, timeout: float, retries: int) -> int | None:
    request = urllib.request.Request(url, headers={"User-Agent": "FluxMindHealth/1.0"})
    last_status: int | None = None
    for _ in range(max(1, retries)):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.status
        except urllib.error.HTTPError as exc:
            return exc.code
        except OSError:
            last_status = None
    return last_status


def run_ssh(host: str, command: str, timeout: float) -> tuple[int, str]:
    proc = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={int(timeout)}",
            host,
            command,
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout + 15,
    )
    return proc.returncode, proc.stdout


def directory_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", action="append", default=[], help="HTTP(S) URL to check")
    parser.add_argument("--ssh-host", help="remote host for systemd/runtime checks")
    parser.add_argument("--retries", type=int, default=3, help="HTTP retry count")
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()

    failures: list[str] = []

    required = [
        "app.py",
        "api.py",
        "src/chain.py",
        "src/ingestion.py",
        "src/capabilities.py",
        "src/providers.py",
        "src/jobs.py",
        "src/metadata.py",
        "src/evaluation.py",
        "eval/rag_baseline.json",
        "scripts/evaluate_rag.py",
        "docs/DEPLOYMENT_STATUS.md",
        "docs/ARCHITECTURE.md",
        "docs/BACKLOG.md",
        "docs/PLATFORM_AUDIT_AND_ROADMAP.md",
        "papers/library/manifest.json",
    ]
    for relative in required:
        check((PROJECT_ROOT / relative).exists(), f"required file: {relative}", failures)

    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    check("11.FluxMind" in readme, "README records formal workspace index", failures)
    check("Previous temporary index `80` has been retired" in readme, "README records 80 retirement", failures)

    app_source = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")
    check("st.write_stream" not in app_source, "chat stream avoids st.write_stream", failures)
    check("notranslate" in app_source and 'translate", "no"' in app_source, "translation guard installed", failures)
    check("get_async_job_manager" in app_source, "Streamlit async job panel installed", failures)
    api_source = (PROJECT_ROOT / "api.py").read_text(encoding="utf-8")
    check("/corpus/papers" in api_source, "corpus metadata route installed", failures)
    check("/jobs/index/rebuild" in api_source, "index rebuild job route installed", failures)
    check("/jobs/async/index/rebuild" in api_source, "async index rebuild job route installed", failures)
    check("/jobs/{job_id}/retry" in api_source, "job retry route installed", failures)

    manifest = json.loads((PROJECT_ROOT / "papers/library/manifest.json").read_text(encoding="utf-8"))
    check(len(manifest) >= 6, "seed paper manifest has at least 6 entries", failures)
    if (PROJECT_ROOT / "artifacts").exists():
        print(f"info artifact bytes={directory_size_bytes(PROJECT_ROOT / 'artifacts')}")
    else:
        print("skip artifact directory is absent")
    if (PROJECT_ROOT / "jobs").exists():
        print(f"info job bytes={directory_size_bytes(PROJECT_ROOT / 'jobs')}")
    else:
        print("skip job directory is absent")

    index_file = PROJECT_ROOT / "faiss_index" / "index.faiss"
    if index_file.exists():
        check(index_file.stat().st_size > 0, "local FAISS index is non-empty", failures)
        print(f"info local FAISS index bytes={index_file.stat().st_size}")
    else:
        print("skip local FAISS index is absent")

    active_papers_file = PROJECT_ROOT / "faiss_index" / "active_papers.json"
    if active_papers_file.exists():
        active_papers = json.loads(active_papers_file.read_text(encoding="utf-8"))
        check(isinstance(active_papers, list), "active paper selection is a list", failures)
        print(f"info active papers={len(active_papers)}")
    else:
        print("skip active paper selection is absent")

    for url in args.url:
        status = http_status(url, args.timeout, args.retries)
        check(status == 200, f"{url} returns 200 (got {status})", failures)

    if args.ssh_host:
        command = (
            "set -e; "
            "systemctl is-active cloudflared-fluxmind-smy.service fluxmind-ui.service fluxmind-api.service docker.service; "
            "ss -ltnp | egrep '18501|18502'; "
            "curl -sS --max-time 10 http://127.0.0.1:18502/health; "
            "test -f /opt/fluxmind/app.py; "
            "grep -q 'render_streaming_response' /opt/fluxmind/app.py; "
            "grep -q 'get_async_job_manager' /opt/fluxmind/app.py; "
            "grep -q '/corpus/papers' /opt/fluxmind/api.py; "
            "grep -q '/jobs/async/index/rebuild' /opt/fluxmind/api.py; "
            "test -f /opt/fluxmind/src/capabilities.py; "
            "grep -E '^(LLM_MODEL|EMBEDDING_MODEL)=' /opt/fluxmind/.env; "
            "test -s /opt/fluxmind/faiss_index/index.faiss; "
            "python3 - <<'PY'\n"
            "import json\n"
            "from pathlib import Path\n"
            "root = Path('/opt/fluxmind')\n"
            "active = root / 'faiss_index' / 'active_papers.json'\n"
            "papers = json.loads(active.read_text()) if active.exists() else []\n"
            "print(f'active_papers={len(papers)}')\n"
            "print(f'faiss_index_bytes={(root / \"faiss_index\" / \"index.faiss\").stat().st_size}')\n"
            "PY\n"
            "journalctl -u fluxmind-api.service -u fluxmind-ui.service --since '30 minutes ago' --no-pager | "
            "egrep -i 'error|exception|traceback' | tail -20 || true; "
            "df -h / | sed -n '2p'"
        )
        code, output = run_ssh(args.ssh_host, command, args.timeout)
        if output.strip():
            print(output.rstrip())
        check(code == 0, f"{args.ssh_host} remote runtime checks", failures)

    if failures:
        print("\nFailed checks:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
