#!/usr/bin/env python3
"""FluxMind local and deployed-runtime health checks.

The local check validates the files, imports, corpus state, and persisted index
that the research workflow actually needs. ``--url`` adds simple HTTP checks;
``--ssh-host`` checks the deployed services and a small set of live API flows.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SSH_COMMAND_TIMEOUT_FLOOR_S = 180.0
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REQUIRED_FILES = (
    "app.py",
    "api.py",
    "src/chain.py",
    "src/users.py",
    "src/ingestion.py",
    "src/embeddings.py",
    "src/jobs.py",
    "src/providers.py",
    "src/artifacts.py",
    "src/execution_policy.py",
    "src/evaluation.py",
    "eval/rag_baseline.json",
    "papers/library/manifest.json",
    "scripts/evaluate_rag.py",
    "scripts/run_job_worker.py",
    "deploy/systemd/fluxmind-worker.service",
)

CORE_IMPORTS = (
    "src.chain",
    "src.users",
    "src.ingestion",
    "src.jobs",
    "src.providers",
    "src.artifacts",
    "src.evaluation",
)


def check(condition: bool, label: str, failures: list[str]) -> None:
    status = "ok" if condition else "fail"
    print(f"{status:4} {label}")
    if not condition:
        failures.append(label)


def http_status(url: str, timeout: float, retries: int) -> int | None:
    request = urllib.request.Request(url, headers={"User-Agent": "FluxMindHealth/1.0"})
    last_status: int | None = None
    for attempt in range(max(1, retries)):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.status
        except urllib.error.HTTPError as exc:
            last_status = exc.code
            if exc.code not in {429, 502, 503, 504}:
                return exc.code
        except OSError:
            last_status = None
        if attempt + 1 < max(1, retries):
            time.sleep(min(1.0, timeout))
    return last_status


def run_ssh(host: str, command: str, timeout: float) -> tuple[int, str]:
    command_timeout = max(timeout + 15, SSH_COMMAND_TIMEOUT_FLOOR_S)
    ssh_command = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={int(timeout)}",
        host,
        command,
    ]
    try:
        proc = subprocess.run(
            ssh_command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=command_timeout,
        )
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        detail = f"ssh command timed out after {command_timeout:.1f}s"
        return 124, f"{output.rstrip()}\n{detail}\n" if output else f"{detail}\n"
    except FileNotFoundError:
        return 127, "ssh executable not found\n"
    return proc.returncode, proc.stdout


def directory_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def check_python_sources(failures: list[str]) -> None:
    for relative in ("app.py", "api.py"):
        path = PROJECT_ROOT / relative
        if not path.exists():
            continue
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
        except (OSError, SyntaxError):
            check(False, f"Python syntax: {relative}", failures)
        else:
            check(True, f"Python syntax: {relative}", failures)

    for module_name in CORE_IMPORTS:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            check(False, f"import {module_name} ({exc.__class__.__name__})", failures)
        else:
            check(True, f"import {module_name}", failures)


def check_corpus_state(failures: list[str]) -> None:
    manifest_path = PROJECT_ROOT / "papers" / "library" / "manifest.json"
    try:
        manifest = load_json(manifest_path)
    except (OSError, ValueError):
        check(False, "paper manifest is valid JSON", failures)
    else:
        papers = manifest if isinstance(manifest, dict) else {}
        check(bool(papers), "paper manifest contains papers", failures)
        print(f"info curated papers={len(papers)}")

    index_dir = PROJECT_ROOT / "faiss_index"
    index_file = index_dir / "index.faiss"
    index_metadata = index_dir / "index.pkl"
    if not index_file.exists() and not index_metadata.exists():
        print("skip local FAISS index is absent")
        return
    check(index_file.is_file() and index_file.stat().st_size > 0, "local FAISS vectors are non-empty", failures)
    check(index_metadata.is_file() and index_metadata.stat().st_size > 0, "local FAISS metadata is non-empty", failures)
    print(f"info local FAISS bytes={directory_size_bytes(index_dir)}")

    active_path = index_dir / "active_papers.json"
    if not active_path.exists():
        print("skip active paper selection is absent")
        return
    try:
        active_papers = load_json(active_path)
    except (OSError, ValueError):
        check(False, "active paper selection is valid JSON", failures)
        return
    check(isinstance(active_papers, list) and bool(active_papers), "active paper selection is a non-empty list", failures)
    if not isinstance(active_papers, list):
        return
    missing_pdfs = [
        source_path
        for source_path in active_papers
        if not (PROJECT_ROOT / source_path).is_file()
    ]
    check(not missing_pdfs, "active paper files exist", failures)
    print(f"info active papers={len(active_papers)}")

    chunks_db = PROJECT_ROOT / "metadata" / "chunks.sqlite3"
    if not chunks_db.exists():
        print("skip chunk metadata database is absent")
        return
    try:
        with sqlite3.connect(f"file:{chunks_db}?mode=ro", uri=True) as conn:
            chunk_count = int(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
            chunk_sources = {
                row[0]
                for row in conn.execute(
                    "SELECT DISTINCT source_path FROM chunks WHERE source_path IS NOT NULL"
                )
            }
    except sqlite3.Error:
        check(False, "chunk metadata database is readable", failures)
        return
    check(chunk_count > 0, "chunk metadata contains rows", failures)
    check(set(active_papers) == chunk_sources, "chunk metadata matches active papers", failures)
    print(f"info chunks={chunk_count} chunk_sources={len(chunk_sources)}")


def check_user_store(failures: list[str]) -> None:
    users_db = PROJECT_ROOT / "metadata" / "users.sqlite3"
    if not users_db.exists():
        print("skip local user store awaits first Streamlit setup")
        return
    try:
        with sqlite3.connect(f"file:{users_db}?mode=ro", uri=True) as conn:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            user_count = int(conn.execute("SELECT COUNT(*) FROM users").fetchone()[0])
    except sqlite3.Error:
        check(False, "local user store is readable", failures)
        return
    check({"users", "query_history"}.issubset(tables), "local user store schema is ready", failures)
    print(f"info local users={user_count}")


def remote_command() -> str:
    """Return the compact deployed-runtime check executed over SSH."""
    return r"""set -e
systemctl is-active cloudflared-fluxmind-smy.service fluxmind-ui.service fluxmind-api.service fluxmind-worker.service docker.service
ss -ltn | grep -q ':18501'
ss -ltn | grep -q ':18502'
test -f /opt/fluxmind/app.py
test -f /opt/fluxmind/api.py
test -f /opt/fluxmind/src/users.py
test -f /opt/fluxmind/scripts/run_job_worker.py
test -s /opt/fluxmind/faiss_index/index.faiss
python3 - <<'PY'
import json
import sqlite3
from pathlib import Path
from urllib import request

root = Path("/opt/fluxmind")
token = ""
env_file = root / ".env"
if env_file.exists():
    for line in env_file.read_text(encoding="utf-8").splitlines():
        if line.startswith("FLUXMIND_API_TOKEN="):
            token = line.split("=", 1)[1].strip().strip("\"'")
            break
headers = {"X-API-Key": token} if token else {}

def api(path, *, payload=None):
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    call_headers = dict(headers)
    if body is not None:
        call_headers["Content-Type"] = "application/json"
    req = request.Request(
        "http://127.0.0.1:18502" + path,
        data=body,
        headers=call_headers,
        method="POST" if body is not None else "GET",
    )
    with request.urlopen(req, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))

health = api("/health")
ready = api("/ready")
corpus = api("/corpus/status").get("status", {})
retrieval = api(
    "/query/retrieve",
    payload={
        "question": "sliding mode control observer",
        "answer_mode": "literature_review",
    },
).get("retrieval", {})
openapi = api("/openapi.json")

if health.get("status") != "ok":
    raise SystemExit("API health failed")
if ready.get("status") != "ready":
    raise SystemExit("retrieval warmup is not ready")
if int(retrieval.get("context_count") or 0) <= 0:
    raise SystemExit("retrieval returned no context")
if retrieval.get("missing_source_page_refs"):
    raise SystemExit("retrieval returned incomplete source/page metadata")
if "/query/history/{user_id}" not in openapi.get("paths", {}):
    raise SystemExit("query-history route missing from OpenAPI")

active_path = root / "faiss_index" / "active_papers.json"
active = json.loads(active_path.read_text(encoding="utf-8"))
chunks_path = root / "metadata" / "chunks.sqlite3"
with sqlite3.connect(chunks_path) as conn:
    chunk_count = int(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
    sources = {
        row[0]
        for row in conn.execute(
            "SELECT DISTINCT source_path FROM chunks WHERE source_path IS NOT NULL"
        )
    }
if chunk_count <= 0 or set(active) != sources:
    raise SystemExit("active corpus and chunk metadata differ")

users_path = root / "metadata" / "users.sqlite3"
if users_path.exists():
    with sqlite3.connect(users_path) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    if not {"users", "query_history"}.issubset(tables):
        raise SystemExit("local user schema incomplete")

print(
    "runtime_smoke=ok "
    f"active_papers={len(active)} chunks={chunk_count} "
    f"retrieval_context={retrieval.get('context_count')} "
    f"corpus_status={corpus.get('status', 'unknown')}"
)
PY
journalctl -u fluxmind-api.service -u fluxmind-ui.service --since '30 minutes ago' --no-pager | egrep -i 'error|exception|traceback' | tail -20 || true
df -h / | sed -n '2p'"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", action="append", default=[], help="HTTP(S) URL to check")
    parser.add_argument("--ssh-host", help="remote host for systemd/runtime checks")
    parser.add_argument("--retries", type=int, default=3, help="HTTP retry count")
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()
    failures: list[str] = []

    for relative in REQUIRED_FILES:
        check((PROJECT_ROOT / relative).is_file(), f"required file: {relative}", failures)

    check_python_sources(failures)
    check_corpus_state(failures)
    check_user_store(failures)

    for runtime_dir in ("artifacts", "jobs"):
        path = PROJECT_ROOT / runtime_dir
        if path.exists():
            print(f"info {runtime_dir} bytes={directory_size_bytes(path)}")
        else:
            print(f"skip {runtime_dir} directory is absent")

    for url in args.url:
        status = http_status(url, args.timeout, args.retries)
        check(status == 200, f"{url} returns 200 (got {status})", failures)

    if args.ssh_host:
        code, output = run_ssh(args.ssh_host, remote_command(), args.timeout)
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
