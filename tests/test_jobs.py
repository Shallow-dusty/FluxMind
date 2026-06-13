import time
import sqlite3
import threading
from pathlib import Path

import pytest

from src.capabilities import CodeExecutionRequest, ImageGenerationRequest
from src.execution_policy import POLICY_VIOLATION_EXIT_CODE
from src.ingestion import IngestionCancelled
from src.jobs import (
    DEFAULT_OWNER_ID,
    AsyncJobManager,
    LocalDurableJobWorker,
    LocalJobRunner,
    LocalJobStore,
    future_utc,
    parse_utc,
)


@pytest.fixture(autouse=True)
def no_runtime_event_disk_writes(monkeypatch):
    monkeypatch.setattr("src.jobs.append_runtime_event", lambda **_kwargs: None)


def wait_for_status(store: LocalJobStore, job_id: str, statuses: set[str], timeout_s: float = 3):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        job = store.get(job_id)
        if job and job.status in statuses:
            return job
        time.sleep(0.02)
    job = store.get(job_id)
    raise AssertionError(f"Job {job_id} did not reach {statuses}; last={job.status if job else None}")


def test_mock_image_job_persists_succeeded_record(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("src.providers.ARTIFACTS_DIR", tmp_path / "artifacts")
    store = LocalJobStore(tmp_path / "jobs.jsonl")
    runner = LocalJobRunner(store)

    job = runner.run_mock_image(ImageGenerationRequest(prompt="SMC observer"))
    loaded = store.get(job.job_id)

    assert loaded is not None
    assert loaded.status == "succeeded"
    assert loaded.kind == "image_generation"
    assert loaded.artifacts[0]["mime_type"] == "image/svg+xml"
    assert [entry["status"] for entry in loaded.logs] == ["running", "succeeded"]


def test_local_python_job_persists_execution_result(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("src.providers.ARTIFACTS_DIR", tmp_path / "artifacts")
    store = LocalJobStore(tmp_path / "jobs.jsonl")
    runner = LocalJobRunner(store)

    job = runner.run_local_python(
        CodeExecutionRequest(
            language="python",
            entrypoint="main.py",
            files={
                "main.py": (
                    "from pathlib import Path\n"
                    "print('job-ok')\n"
                    "Path('result.txt').write_text('artifact-ok', encoding='utf-8')\n"
                )
            },
        )
    )

    assert job.status == "succeeded"
    assert job.result["stdout"] == "job-ok\n"
    assert job.artifacts[0]["title"] == "result.txt"
    assert store.get(job.job_id).result["exit_code"] == 0
    assert job.logs[-1]["metadata"]["exit_code"] == 0
    assert job.logs[-1]["metadata"]["artifact_count"] == 1


def test_local_python_job_records_no_secret_execution_event(tmp_path: Path, monkeypatch):
    events = []
    monkeypatch.setattr("src.jobs.append_runtime_event", lambda **kwargs: events.append(kwargs))
    store = LocalJobStore(tmp_path / "jobs.jsonl")
    runner = LocalJobRunner(store)

    job = runner.run_local_python(
        CodeExecutionRequest(
            language="python",
            entrypoint="main.py",
            files={"main.py": "print('event-ok')"},
        ),
        request_id="req-code-event",
        ownership={"owner_id": "lab-code", "owner_label": "Code Lab"},
    )

    assert job.status == "succeeded"
    assert len(events) == 1
    event = events[0]
    assert event["kind"] == "code_execution"
    assert event["code"] == "execution_succeeded"
    assert event["request_id"] == "req-code-event"
    metadata = event["metadata"]
    assert metadata["job_id"] == job.job_id
    assert metadata["status"] == "succeeded"
    assert metadata["language"] == "python"
    assert metadata["backend"] == "local"
    assert metadata["owner_id"] == "lab-code"
    assert metadata["provider_runtime"] == "python-local"
    assert metadata["execution_policy"] == "local-safe-v1"
    assert metadata["policy_violation"] == "false"
    assert metadata["exit_code"] == 0
    assert metadata["artifact_count"] == 0
    assert metadata["max_artifacts"] == "16"
    assert metadata["max_artifact_bytes"] == str(2 * 1024 * 1024)
    assert metadata["max_artifact_total_bytes"] == str(8 * 1024 * 1024)
    assert metadata["artifact_exported_count"] == "0"
    assert metadata["artifact_collection_truncated"] == "false"
    assert "event-ok" not in str(metadata)
    assert "main.py" not in str(metadata)


def test_code_execution_event_log_failure_does_not_fail_job(tmp_path: Path, monkeypatch):
    def fail_event_log(**_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("src.jobs.append_runtime_event", fail_event_log)
    store = LocalJobStore(tmp_path / "jobs.jsonl")
    runner = LocalJobRunner(store)

    job = runner.run_local_python(
        CodeExecutionRequest(
            language="python",
            entrypoint="main.py",
            files={"main.py": "print('still-ok')"},
        )
    )

    assert job.status == "succeeded"
    assert any(entry["status"] == "runtime_event_warning" for entry in job.logs)


def test_local_python_job_uses_configured_docker_backend(tmp_path: Path, monkeypatch):
    captured: dict[str, list[str]] = {}

    class FakePopen:
        returncode = 0

        def __init__(self, command, **_kwargs):
            captured["command"] = command
            mount = command[command.index("-v") + 1]
            workdir = Path(mount.split(":", 1)[0])
            (workdir / "docker-result.txt").write_text("job-docker-artifact", encoding="utf-8")

        def poll(self):
            return self.returncode

        def communicate(self, timeout=None):
            return "job-docker-ok\n", ""

    monkeypatch.setattr("src.jobs.CODE_EXECUTION_BACKEND", "docker")
    monkeypatch.setattr("src.jobs.DOCKER_EXECUTION_IMAGE", "python:3.12-slim")
    monkeypatch.setattr("src.providers.shutil.which", lambda _name: "/usr/bin/docker")
    monkeypatch.setattr("src.providers.subprocess.Popen", FakePopen)
    store = LocalJobStore(tmp_path / "jobs.jsonl")
    runner = LocalJobRunner(store, artifact_root=tmp_path / "artifacts")

    job = runner.run_local_python(
        CodeExecutionRequest(
            language="python",
            entrypoint="main.py",
            files={"main.py": "print('job-docker-ok')"},
        )
    )

    assert job.status == "succeeded"
    assert job.result["stdout"] == "job-docker-ok\n"
    assert job.result["runtime_metadata"]["provider_runtime"] == "docker-python"
    assert job.result["runtime_metadata"]["network_policy_enforced"] == "true"
    assert job.artifacts[0]["title"] == "docker-result.txt"
    assert job.artifacts[0]["metadata"]["runtime"] == "docker-python"
    assert captured["command"][captured["command"].index("--network") + 1] == "none"


def test_local_job_runner_uses_custom_artifact_root(tmp_path: Path):
    artifact_root = tmp_path / "eval-artifacts"
    store = LocalJobStore(tmp_path / "jobs.jsonl")
    runner = LocalJobRunner(store, artifact_root=artifact_root)

    job = runner.run_local_python(
        CodeExecutionRequest(
            language="python",
            entrypoint="main.py",
            files={
                "main.py": (
                    "from pathlib import Path\n"
                    "Path('isolated.txt').write_text('artifact-root-ok', encoding='utf-8')\n"
                )
            },
        )
    )

    assert job.status == "succeeded"
    assert job.artifacts[0]["uri"].startswith(artifact_root.resolve().as_uri())
    assert Path(job.artifacts[0]["uri"].removeprefix("file://")).exists()


def test_local_python_job_records_failure(tmp_path: Path):
    store = LocalJobStore(tmp_path / "jobs.jsonl")
    runner = LocalJobRunner(store)

    job = runner.run_local_python(
        CodeExecutionRequest(
            language="python",
            entrypoint="main.py",
            files={"main.py": "raise SystemExit(7)"},
        )
    )

    assert job.status == "failed"
    assert job.result["exit_code"] == 7
    assert job.error["code"] == "execution_failed"
    assert job.logs[-1]["status"] == "failed"
    assert job.logs[-1]["metadata"]["error_code"] == "execution_failed"


def test_local_python_job_records_policy_violation(tmp_path: Path, monkeypatch):
    events = []
    monkeypatch.setattr("src.jobs.append_runtime_event", lambda **kwargs: events.append(kwargs))
    store = LocalJobStore(tmp_path / "jobs.jsonl")
    runner = LocalJobRunner(store)

    job = runner.run_local_python(
        CodeExecutionRequest(
            language="python",
            entrypoint="main.py",
            files={"main.py": "import subprocess\nsubprocess.run(['echo', 'bad'])\n"},
        )
    )

    assert job.status == "failed"
    assert job.result["exit_code"] == POLICY_VIOLATION_EXIT_CODE
    assert job.result["runtime_metadata"]["policy_violation"] == "true"
    assert job.error["code"] == "execution_policy_violation"
    assert "subprocess" in job.error["message"]
    assert events[0]["kind"] == "code_execution"
    assert events[0]["code"] == "execution_policy_violation"
    assert events[0]["message"] == "Code execution job failed."
    assert events[0]["metadata"]["policy_violation"] == "true"
    assert events[0]["metadata"]["execution_policy_violations"] != "0"
    assert "subprocess" not in str(events[0]["metadata"])


def test_job_store_filters_latest_records(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("src.providers.ARTIFACTS_DIR", tmp_path / "artifacts")
    store = LocalJobStore(tmp_path / "jobs.jsonl")
    runner = LocalJobRunner(store)
    image_job = runner.run_mock_image(
        ImageGenerationRequest(prompt="SMC observer filter smoke"),
        request_id="req-image-filter",
    )
    failed_job = runner.run_local_python(
        CodeExecutionRequest(
            language="python",
            entrypoint="main.py",
            files={"main.py": "raise SystemExit(9)"},
        ),
        request_id="req-python-filter",
    )

    assert [job.job_id for job in store.list_latest(status="failed")] == [failed_job.job_id]
    assert [job.job_id for job in store.list_latest(kind="image_generation")] == [image_job.job_id]
    assert [job.job_id for job in store.list_latest(q="observer filter")] == [image_job.job_id]
    assert [job.job_id for job in store.list_latest(q="req-python-filter")] == [failed_job.job_id]
    assert store.list_latest(status="queued") == []


def test_job_store_persists_and_filters_local_ownership(tmp_path: Path):
    jobs_file = tmp_path / "jobs.jsonl"
    store = LocalJobStore(jobs_file)
    runner = LocalJobRunner(store)

    default_job = runner._enqueue("image_generation", {"prompt": "default owner"}, "req-default")
    owned_job = runner._enqueue(
        "code_execution",
        {"entrypoint": "main.py"},
        "req-owned",
        ownership={"owner_id": "lab-a", "owner_label": "Lab A"},
    )

    loaded_default = store.get(default_job.job_id)
    loaded_owned = store.get(owned_job.job_id)

    assert loaded_default.owner_id == DEFAULT_OWNER_ID
    assert loaded_default.ownership_source == "default"
    assert loaded_owned.owner_id == "lab-a"
    assert loaded_owned.owner_label == "Lab A"
    assert loaded_owned.ownership_source == "request"
    assert loaded_owned.logs[0]["metadata"]["owner_id"] == "lab-a"
    assert [job.job_id for job in store.list_latest(owner_id="lab-a")] == [owned_job.job_id]
    assert [job.job_id for job in store.list_latest(q="Lab A")] == [owned_job.job_id]

    with sqlite3.connect(jobs_file.with_suffix(".sqlite3")) as conn:
        row = conn.execute(
            "SELECT owner_id, owner_label, ownership_source FROM jobs WHERE job_id = ?",
            (owned_job.job_id,),
        ).fetchone()
    assert row == ("lab-a", "Lab A", "request")

    jobs_file.with_suffix(".sqlite3").unlink()
    reloaded = LocalJobStore(jobs_file)
    assert reloaded.get(owned_job.job_id).owner_id == "lab-a"


def test_local_python_job_records_timeout_code(tmp_path: Path):
    store = LocalJobStore(tmp_path / "jobs.jsonl")
    runner = LocalJobRunner(store)

    job = runner.run_local_python(
        CodeExecutionRequest(
            language="python",
            entrypoint="main.py",
            files={"main.py": "import time\ntime.sleep(2)"},
            timeout_s=1,
        )
    )

    assert job.status == "failed"
    assert job.result["exit_code"] == 124
    assert job.error["code"] == "execution_timeout"
    assert "timed out" in job.error["message"]


def test_local_octave_job_records_missing_runtime_failure(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("src.providers.shutil.which", lambda _name: None)
    store = LocalJobStore(tmp_path / "jobs.jsonl")
    runner = LocalJobRunner(store)

    job = runner.run_local_octave(
        CodeExecutionRequest(
            language="octave",
            entrypoint="main.m",
            files={"main.m": "disp('ok');"},
        )
    )

    assert job.status == "failed"
    assert job.kind == "code_execution"
    assert job.request["language"] == "octave"
    assert job.result["exit_code"] == 127
    assert job.error["code"] == "runtime_unavailable"
    assert "GNU Octave executable not found" in job.error["message"]


def test_index_rebuild_job_records_selected_pdfs(tmp_path: Path, monkeypatch):
    paper = tmp_path / "papers" / "library" / "paper.pdf"
    paper.parent.mkdir(parents=True)
    paper.write_bytes(b"%PDF-1.4")

    monkeypatch.setattr("src.jobs.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("src.jobs.ingestion.discover_pdfs", lambda: [paper])
    monkeypatch.setattr(
        "src.jobs.ingestion.rebuild_vector_store_from_pdfs",
        lambda paths, **_kwargs: (object(), 12),
    )

    store = LocalJobStore(tmp_path / "jobs.jsonl")
    runner = LocalJobRunner(store)
    job = runner.run_index_rebuild(["papers/library/paper.pdf"])

    assert job.status == "succeeded"
    assert job.kind == "index_rebuild"
    assert job.result["paper_count"] == 1
    assert job.result["chunk_count"] == 12


def test_index_rebuild_job_records_mid_rebuild_cancellation(tmp_path: Path, monkeypatch):
    paper = tmp_path / "papers" / "library" / "paper.pdf"
    paper.parent.mkdir(parents=True)
    paper.write_bytes(b"%PDF-1.4")

    def cancelled_rebuild(_paths, *, cancel_event):
        assert cancel_event is not None
        cancel_event.set()
        raise IngestionCancelled("Index rebuild was cancelled.")

    monkeypatch.setattr("src.jobs.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("src.jobs.ingestion.discover_pdfs", lambda: [paper])
    monkeypatch.setattr(
        "src.jobs.ingestion.rebuild_vector_store_from_pdfs",
        cancelled_rebuild,
    )

    store = LocalJobStore(tmp_path / "jobs.jsonl")
    runner = LocalJobRunner(store)
    job = runner.run_index_rebuild(["papers/library/paper.pdf"], cancel_event=threading.Event())

    assert job.status == "cancelled"
    assert job.error == {"code": "cancelled", "message": "Index rebuild was cancelled."}


def test_job_store_list_cancel_and_retry(tmp_path: Path):
    store = LocalJobStore(tmp_path / "jobs.jsonl")
    runner = LocalJobRunner(store)

    failed = runner.run_local_python(
        CodeExecutionRequest(
            language="python",
            entrypoint="main.py",
            files={"main.py": "raise SystemExit(2)"},
        )
    )
    retried = runner.retry(failed.job_id)

    assert retried is not None
    assert retried.job_id != failed.job_id
    assert [job.job_id for job in store.list_latest(limit=2)]
    assert store.cancel("missing") is None


def test_job_retry_inherits_local_ownership(tmp_path: Path):
    store = LocalJobStore(tmp_path / "jobs.jsonl")
    runner = LocalJobRunner(store)

    failed = runner.run_local_python(
        CodeExecutionRequest(
            language="python",
            entrypoint="main.py",
            files={"main.py": "raise SystemExit(2)"},
        ),
        ownership={"owner_id": "lab-retry", "owner_label": "Retry Lab"},
    )
    retried = runner.retry(failed.job_id)
    scheduled = runner.schedule_retry(failed.job_id, delay_s=30)

    assert retried is not None
    assert retried.owner_id == "lab-retry"
    assert retried.owner_label == "Retry Lab"
    assert retried.ownership_source == "inherited"
    assert scheduled is not None
    assert scheduled.owner_id == "lab-retry"
    assert scheduled.ownership_source == "inherited"


def test_job_store_cancel_appends_transition_log(tmp_path: Path):
    store = LocalJobStore(tmp_path / "jobs.jsonl")
    queued = LocalJobRunner(store)._enqueue("image_generation", {"prompt": "x"}, "req-cancel")

    cancelled = store.cancel(queued.job_id)

    assert cancelled.status == "cancelled"
    assert [entry["status"] for entry in cancelled.logs] == ["queued", "cancelled"]


def test_runner_schedules_retry_with_backoff_metadata(tmp_path: Path):
    store = LocalJobStore(tmp_path / "jobs.jsonl")
    runner = LocalJobRunner(store)
    failed = runner.run_local_python(
        CodeExecutionRequest(
            language="python",
            entrypoint="main.py",
            files={"main.py": "raise SystemExit(2)"},
        )
    )

    scheduled = runner.schedule_retry(failed.job_id, delay_s=30, request_id="req-backoff")

    assert scheduled is not None
    assert scheduled.status == "queued"
    assert scheduled.parent_job_id == failed.job_id
    assert scheduled.request_id == "req-backoff"
    assert scheduled.not_before is not None
    assert parse_utc(scheduled.not_before) > parse_utc(scheduled.created_at)
    assert scheduled.logs[-1]["status"] == "queued"


def test_runner_schedules_retry_with_queue_deadline(tmp_path: Path):
    store = LocalJobStore(tmp_path / "jobs.jsonl")
    runner = LocalJobRunner(store)
    failed = runner.run_local_python(
        CodeExecutionRequest(
            language="python",
            entrypoint="main.py",
            files={"main.py": "raise SystemExit(2)"},
        )
    )

    scheduled = runner.schedule_retry(failed.job_id, delay_s=30, queue_timeout_s=60)

    assert scheduled is not None
    assert scheduled.deadline_at is not None
    assert parse_utc(scheduled.deadline_at) > parse_utc(scheduled.created_at)


def test_job_store_finds_claimed_record_by_idempotency_key(tmp_path: Path):
    jobs_file = tmp_path / "jobs.jsonl"
    store = LocalJobStore(jobs_file)
    runner = LocalJobRunner(store)
    first = runner._enqueue(
        "code_execution",
        {"entrypoint": "main.py"},
        request_id="req-one",
        idempotency_key="idem-store",
    )

    found = store.find_by_idempotency_key(kind="code_execution", key="idem-store")

    assert found is not None
    assert found.job_id == first.job_id
    assert found.idempotency_key == "idem-store"
    assert store.find_by_idempotency_key(kind="index_rebuild", key="idem-store") is None

    jobs_file.with_suffix(".sqlite3").unlink()
    reloaded = LocalJobStore(jobs_file)
    reloaded_found = reloaded.find_by_idempotency_key(kind="code_execution", key="idem-store")

    assert reloaded_found is not None
    assert reloaded_found.job_id == first.job_id


def test_local_job_runner_reuses_idempotency_key_for_immediate_jobs(tmp_path: Path):
    store = LocalJobStore(tmp_path / "jobs.jsonl")
    runner = LocalJobRunner(store)

    first = runner.run_local_python(
        CodeExecutionRequest(
            language="python",
            entrypoint="main.py",
            files={"main.py": "print('first run')"},
        ),
        idempotency_key="idem-immediate",
    )
    second = runner.run_local_python(
        CodeExecutionRequest(
            language="python",
            entrypoint="main.py",
            files={"main.py": "raise RuntimeError('should not execute')"},
        ),
        idempotency_key="idem-immediate",
    )

    assert first.status == "succeeded"
    assert second.job_id == first.job_id
    assert second.status == "succeeded"
    assert len(store.list_latest(limit=10)) == 1


def test_async_manager_reuses_idempotency_key_for_duplicate_python_jobs(tmp_path: Path):
    store = LocalJobStore(tmp_path / "jobs.jsonl")
    manager = AsyncJobManager(store)
    request = CodeExecutionRequest(
        language="python",
        entrypoint="main.py",
        files={"main.py": "print('idempotent-async')"},
    )

    first = manager.enqueue_local_python(request, idempotency_key="idem-async")
    second = manager.enqueue_local_python(request, idempotency_key="idem-async")

    assert second.job_id == first.job_id
    assert second.idempotency_key == "idem-async"
    assert len(store.list_latest(limit=10)) == 1
    manager._queue.join()
    assert store.get(first.job_id).status == "succeeded"


def test_async_manager_preserves_distinct_jobs_for_different_or_missing_keys(tmp_path: Path):
    store = LocalJobStore(tmp_path / "jobs.jsonl")
    manager = AsyncJobManager(store)
    request = CodeExecutionRequest(
        language="python",
        entrypoint="main.py",
        files={"main.py": "print('distinct-async')"},
    )

    first = manager.enqueue_local_python(request, idempotency_key="idem-one")
    second = manager.enqueue_local_python(request, idempotency_key="idem-two")
    third = manager.enqueue_local_python(request)
    fourth = manager.enqueue_local_python(request)

    assert len({first.job_id, second.job_id, third.job_id, fourth.job_id}) == 4
    assert third.idempotency_key is None
    assert fourth.idempotency_key is None
    assert len(store.list_latest(limit=10)) == 4
    manager._queue.join()


def test_job_store_claims_due_job_with_worker_lease(tmp_path: Path):
    store = LocalJobStore(tmp_path / "jobs.jsonl")
    runner = LocalJobRunner(store)
    scheduled = runner._enqueue(
        "code_execution",
        {
            "language": "python",
            "entrypoint": "main.py",
            "files": {"main.py": "print('later')"},
            "timeout_s": 5,
            "memory_mb": 512,
        },
        "req-scheduled",
        not_before=future_utc(60),
    )
    due = runner._enqueue(
        "code_execution",
        {
            "language": "python",
            "entrypoint": "main.py",
            "files": {"main.py": "print('now')"},
            "timeout_s": 5,
            "memory_mb": 512,
        },
        "req-due",
    )

    claimed = store.claim_next_due_job(worker_id="worker-a", lease_seconds=30)

    assert claimed is not None
    assert claimed.job_id == due.job_id
    assert claimed.status == "queued"
    assert claimed.worker_id == "worker-a"
    assert claimed.leased_at is not None
    assert claimed.lease_expires_at is not None
    assert parse_utc(claimed.lease_expires_at) > parse_utc(claimed.leased_at)
    assert store.claim_next_due_job(worker_id="worker-b", lease_seconds=30) is None
    assert store.get(scheduled.job_id).worker_id is None
    with sqlite3.connect(store.db_path) as conn:
        row = conn.execute(
            "SELECT worker_id, leased_at, lease_expires_at FROM jobs WHERE job_id = ?",
            (due.job_id,),
        ).fetchone()
    assert row[0] == "worker-a"
    assert row[1] is not None
    assert row[2] is not None


def test_job_store_reclaims_expired_worker_lease(tmp_path: Path):
    store = LocalJobStore(tmp_path / "jobs.jsonl")
    runner = LocalJobRunner(store)
    due = runner._enqueue(
        "code_execution",
        {
            "language": "python",
            "entrypoint": "main.py",
            "files": {"main.py": "print('reclaim')"},
            "timeout_s": 5,
            "memory_mb": 512,
        },
        "req-reclaim",
    )
    claimed = store.claim_job(due.job_id, worker_id="worker-a", lease_seconds=30)
    claimed.lease_expires_at = "2000-01-01T00:00:00+00:00"
    store.append(claimed)

    reclaimed = store.claim_job(due.job_id, worker_id="worker-b", lease_seconds=30)

    assert reclaimed is not None
    assert reclaimed.worker_id == "worker-b"
    assert parse_utc(reclaimed.lease_expires_at) > parse_utc(reclaimed.leased_at)


def test_job_store_summarizes_worker_lease_health(tmp_path: Path):
    store = LocalJobStore(tmp_path / "jobs.jsonl")
    runner = LocalJobRunner(store)
    active = runner._enqueue("image_generation", {"prompt": "active"}, "req-active")
    expired = runner._enqueue("image_generation", {"prompt": "expired"}, "req-expired")
    store.claim_job(active.job_id, worker_id="worker-a", lease_seconds=30)
    expired_claim = store.claim_job(expired.job_id, worker_id="worker-b", lease_seconds=30)
    expired_claim.lease_expires_at = "2000-01-01T00:00:00+00:00"
    store.append(expired_claim)

    health = store.worker_lease_health()

    assert health["total_leased_jobs"] == 2
    assert health["worker_ids"] == ["worker-a", "worker-b"]
    assert health["by_worker"] == {"worker-a": 1, "worker-b": 1}
    assert health["active_worker_ids"] == ["worker-a"]
    assert health["expired_worker_ids"] == ["worker-b"]
    assert health["active_leases"] == 1
    assert health["expired_leases"] == 1
    assert {item["worker_id"] for item in health["latest"]} == {"worker-a", "worker-b"}


def test_job_store_releases_worker_lease(tmp_path: Path):
    store = LocalJobStore(tmp_path / "jobs.jsonl")
    due = LocalJobRunner(store)._enqueue(
        "image_generation",
        {"prompt": "release me"},
        "req-release",
    )
    claimed = store.claim_job(due.job_id, worker_id="worker-a", lease_seconds=30)

    released = store.release_job_lease(claimed.job_id, worker_id="worker-a")

    assert released.worker_id is None
    assert released.leased_at is None
    assert released.lease_expires_at is None
    assert store.claim_job(due.job_id, worker_id="worker-b", lease_seconds=30).worker_id == "worker-b"


def test_async_manager_runs_zero_delay_scheduled_retry(tmp_path: Path):
    store = LocalJobStore(tmp_path / "jobs.jsonl")
    manager = AsyncJobManager(store)
    failed = LocalJobRunner(store).run_local_python(
        CodeExecutionRequest(
            language="python",
            entrypoint="main.py",
            files={"main.py": "print('retry-ok')\nraise SystemExit(1)"},
        )
    )
    failed.request["files"] = {"main.py": "print('retry-ok')"}
    store.append(failed)

    scheduled = manager.schedule_retry(failed.job_id, delay_s=0)

    assert scheduled is not None
    assert scheduled.parent_job_id == failed.job_id
    manager._queue.join()
    finished = store.get(scheduled.job_id)
    assert finished.status == "succeeded"
    assert finished.result["stdout"] == "retry-ok\n"


def test_async_manager_recovers_queued_job_after_restart(tmp_path: Path):
    store = LocalJobStore(tmp_path / "jobs.jsonl")
    failed = LocalJobRunner(store).run_local_python(
        CodeExecutionRequest(
            language="python",
            entrypoint="main.py",
            files={"main.py": "raise SystemExit(1)"},
        )
    )
    failed.request["files"] = {"main.py": "print('recovered-ok')"}
    store.append(failed)
    scheduled = LocalJobRunner(store).schedule_retry(failed.job_id, delay_s=0)

    assert scheduled is not None
    assert store.queue_health()["due"] == 1

    manager = AsyncJobManager(store)
    manager._queue.join()
    recovered = wait_for_status(store, scheduled.job_id, {"succeeded"})

    assert recovered.result["stdout"] == "recovered-ok\n"
    assert store.queue_health()["queued"] == 0


def test_async_manager_marks_expired_queued_job_failed(tmp_path: Path):
    store = LocalJobStore(tmp_path / "jobs.jsonl")
    manager = AsyncJobManager(store, recover_existing=False)
    record = manager.runner._enqueue(
        "code_execution",
        {
            "language": "python",
            "entrypoint": "main.py",
            "files": {"main.py": "print('should-not-run')"},
            "timeout_s": 5,
            "memory_mb": 512,
        },
        "req-expired",
        deadline_at="2000-01-01T00:00:00+00:00",
    )

    assert store.queue_health()["expired"] == 1
    manager._enqueue(record)
    manager._queue.join()
    expired = store.get(record.job_id)

    assert expired.status == "failed"
    assert expired.result is None
    assert expired.error["code"] == "job_deadline_exceeded"


def test_job_store_mirrors_current_state_to_sqlite(tmp_path: Path):
    store = LocalJobStore(tmp_path / "jobs.jsonl")
    runner = LocalJobRunner(store)

    job = runner.run_local_python(
        CodeExecutionRequest(
            language="python",
            entrypoint="main.py",
            files={"main.py": "print('sqlite-ok')"},
        )
    )

    assert store.db_path.exists()
    with sqlite3.connect(store.db_path) as conn:
        row = conn.execute(
            "SELECT status, attempts FROM jobs WHERE job_id = ?",
            (job.job_id,),
        ).fetchone()
    assert row == ("succeeded", 1)
    assert store.get(job.job_id).result["stdout"] == "sqlite-ok\n"


def test_job_store_migrates_existing_jsonl_to_sqlite(tmp_path: Path):
    store = LocalJobStore(tmp_path / "jobs.jsonl")
    runner = LocalJobRunner(store)
    job = runner.run_local_python(
        CodeExecutionRequest(
            language="python",
            entrypoint="main.py",
            files={"main.py": "raise SystemExit(4)"},
        )
    )
    store.db_path.unlink()

    migrated = LocalJobStore(tmp_path / "jobs.jsonl")

    assert migrated.get(job.job_id).status == "failed"
    assert migrated.db_path.exists()
    assert migrated.list_latest(limit=1)[0].job_id == job.job_id


def test_async_manager_runs_queued_python_job(tmp_path: Path):
    store = LocalJobStore(tmp_path / "jobs.jsonl")
    manager = AsyncJobManager(store, worker_id="worker-test")

    queued = manager.enqueue_local_python(
        CodeExecutionRequest(
            language="python",
            entrypoint="main.py",
            files={"main.py": "print('async-ok')"},
        )
    )
    finished = wait_for_status(store, queued.job_id, {"succeeded"})

    assert queued.status == "queued"
    assert finished.worker_id == "worker-test"
    assert finished.leased_at is not None
    assert finished.lease_expires_at is not None
    assert finished.result["stdout"] == "async-ok\n"


def test_async_manager_requeues_until_dead_lettered(tmp_path: Path):
    store = LocalJobStore(tmp_path / "jobs.jsonl")
    manager = AsyncJobManager(store, worker_id="async-retry")

    queued = manager.enqueue_local_python(
        CodeExecutionRequest(
            language="python",
            entrypoint="main.py",
            files={"main.py": "raise SystemExit(5)"},
        ),
        max_attempts=2,
        retry_backoff_s=0,
    )
    manager._queue.join()
    dead = store.get(queued.job_id)

    assert dead.status == "dead_lettered"
    assert dead.attempts == 2
    assert dead.worker_id == "async-retry"
    assert dead.dead_lettered_at is not None
    assert [entry["status"] for entry in dead.logs] == [
        "queued",
        "running",
        "failed",
        "queued",
        "running",
        "failed",
        "dead_lettered",
    ]


def test_durable_worker_runs_due_python_job_without_async_manager(tmp_path: Path):
    store = LocalJobStore(tmp_path / "jobs.jsonl")
    queued = LocalJobRunner(store)._enqueue(
        "code_execution",
        {
            "language": "python",
            "entrypoint": "main.py",
            "files": {"main.py": "print('durable-worker-ok')"},
            "timeout_s": 5,
            "memory_mb": 512,
        },
        "req-durable-worker",
    )
    worker = LocalDurableJobWorker(store, worker_id="durable-test", lease_seconds=30)

    finished = worker.run_once()

    assert finished.job_id == queued.job_id
    assert finished.status == "succeeded"
    assert finished.worker_id == "durable-test"
    assert finished.result["stdout"] == "durable-worker-ok\n"
    assert [entry["status"] for entry in finished.logs] == ["queued", "running", "succeeded"]


def test_durable_worker_skips_scheduled_jobs_until_due(tmp_path: Path):
    store = LocalJobStore(tmp_path / "jobs.jsonl")
    LocalJobRunner(store)._enqueue(
        "image_generation",
        {"prompt": "not yet"},
        "req-not-yet",
        not_before=future_utc(60),
    )

    result = LocalDurableJobWorker(store, worker_id="durable-test").run_once()

    assert result is None
    assert store.queue_health()["scheduled"] == 1
    assert store.queue_health()["leased_queued"] == 0


def test_durable_worker_marks_expired_queued_job_failed(tmp_path: Path):
    store = LocalJobStore(tmp_path / "jobs.jsonl")
    queued = LocalJobRunner(store)._enqueue(
        "code_execution",
        {
            "language": "python",
            "entrypoint": "main.py",
            "files": {"main.py": "print('should-not-run')"},
            "timeout_s": 5,
            "memory_mb": 512,
        },
        "req-durable-expired",
        deadline_at="2000-01-01T00:00:00+00:00",
    )

    failed = LocalDurableJobWorker(store, worker_id="durable-test").run_once()

    assert failed.job_id == queued.job_id
    assert failed.status == "failed"
    assert failed.worker_id == "durable-test"
    assert failed.result is None
    assert failed.error["code"] == "job_deadline_exceeded"


def test_durable_worker_requeues_failed_job_before_dead_letter(tmp_path: Path):
    store = LocalJobStore(tmp_path / "jobs.jsonl")
    queued = LocalJobRunner(store)._enqueue(
        "code_execution",
        {
            "language": "python",
            "entrypoint": "main.py",
            "files": {"main.py": "raise SystemExit(2)"},
            "timeout_s": 5,
            "memory_mb": 512,
        },
        "req-auto-retry",
        max_attempts=2,
        retry_backoff_s=0,
    )
    worker = LocalDurableJobWorker(store, worker_id="durable-retry")

    retrying = worker.run_once()

    assert retrying.job_id == queued.job_id
    assert retrying.status == "queued"
    assert retrying.attempts == 1
    assert retrying.worker_id is None
    assert [entry["status"] for entry in retrying.logs] == ["queued", "running", "failed", "queued"]
    assert store.queue_health()["due"] == 1

    retrying.request["files"] = {"main.py": "print('retried-ok')"}
    store.append(retrying)
    finished = worker.run_once()

    assert finished.job_id == queued.job_id
    assert finished.status == "succeeded"
    assert finished.attempts == 2
    assert finished.result["stdout"] == "retried-ok\n"


def test_durable_worker_dead_letters_after_retry_policy_exhaustion(tmp_path: Path):
    store = LocalJobStore(tmp_path / "jobs.jsonl")
    queued = LocalJobRunner(store)._enqueue(
        "code_execution",
        {
            "language": "python",
            "entrypoint": "main.py",
            "files": {"main.py": "raise SystemExit(3)"},
            "timeout_s": 5,
            "memory_mb": 512,
        },
        "req-dead-letter",
        max_attempts=2,
        retry_backoff_s=0,
    )
    worker = LocalDurableJobWorker(store, worker_id="durable-dead")

    results = worker.run_until_empty(max_jobs=2)
    dead = store.get(queued.job_id)

    assert [result.status for result in results] == ["queued", "dead_lettered"]
    assert dead.status == "dead_lettered"
    assert dead.attempts == 2
    assert dead.max_attempts == 2
    assert dead.dead_lettered_at is not None
    assert dead.error["code"] == "execution_failed"
    assert store.queue_health()["queued"] == 0
    assert dead.logs[-1]["status"] == "dead_lettered"


def test_durable_worker_cancels_running_python_job_from_store_state(tmp_path: Path):
    store = LocalJobStore(tmp_path / "jobs.jsonl")
    queued = LocalJobRunner(store)._enqueue(
        "code_execution",
        {
            "language": "python",
            "entrypoint": "main.py",
            "files": {
                "main.py": (
                    "import time\n"
                    "time.sleep(5)\n"
                    "print('too-late')\n"
                )
            },
            "timeout_s": 10,
            "memory_mb": 512,
        },
        "req-durable-cancel",
    )
    worker = LocalDurableJobWorker(
        store,
        worker_id="durable-cancel-test",
        cancel_poll_interval_s=0.05,
    )
    result_holder: dict[str, object] = {}

    thread = threading.Thread(
        target=lambda: result_holder.setdefault("job", worker.run_once()),
        daemon=True,
    )
    thread.start()
    wait_for_status(store, queued.job_id, {"running"})

    store.cancel(queued.job_id)
    thread.join(timeout=3)

    assert not thread.is_alive()
    cancelled = store.get(queued.job_id)
    assert cancelled.status == "cancelled"
    assert cancelled.worker_id == "durable-cancel-test"
    assert cancelled.error["code"] == "cancelled"
    assert "cancelled" in cancelled.result["stderr"].lower()


def test_async_manager_cancels_running_python_job(tmp_path: Path):
    store = LocalJobStore(tmp_path / "jobs.jsonl")
    manager = AsyncJobManager(store)

    queued = manager.enqueue_local_python(
        CodeExecutionRequest(
            language="python",
            entrypoint="main.py",
            files={
                "main.py": (
                    "import time\n"
                    "time.sleep(5)\n"
                    "print('too-late')\n"
                )
            },
            timeout_s=10,
        )
    )
    wait_for_status(store, queued.job_id, {"running"})
    manager.cancel(queued.job_id)
    cancelled = wait_for_status(store, queued.job_id, {"cancelled"})

    assert cancelled.error["code"] == "cancelled"


def test_async_manager_cancels_running_index_rebuild(tmp_path: Path, monkeypatch):
    paper = tmp_path / "papers" / "library" / "paper.pdf"
    paper.parent.mkdir(parents=True)
    paper.write_bytes(b"%PDF-1.4")

    def cancellable_rebuild(_paths, *, cancel_event):
        assert cancel_event is not None
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if cancel_event.is_set():
                raise IngestionCancelled("Index rebuild was cancelled.")
            time.sleep(0.02)
        raise AssertionError("cancel_event was not set")

    monkeypatch.setattr("src.jobs.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("src.jobs.ingestion.discover_pdfs", lambda: [paper])
    monkeypatch.setattr(
        "src.jobs.ingestion.rebuild_vector_store_from_pdfs",
        cancellable_rebuild,
    )

    store = LocalJobStore(tmp_path / "jobs.jsonl")
    manager = AsyncJobManager(store)
    queued = manager.enqueue_index_rebuild(["papers/library/paper.pdf"])
    wait_for_status(store, queued.job_id, {"running"})

    manager.cancel(queued.job_id)
    cancelled = wait_for_status(store, queued.job_id, {"cancelled"})

    assert cancelled.error["code"] == "cancelled"
