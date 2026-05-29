import time
import sqlite3
from pathlib import Path

from src.capabilities import CodeExecutionRequest, ImageGenerationRequest
from src.jobs import AsyncJobManager, LocalJobRunner, LocalJobStore


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


def test_index_rebuild_job_records_selected_pdfs(tmp_path: Path, monkeypatch):
    paper = tmp_path / "papers" / "library" / "paper.pdf"
    paper.parent.mkdir(parents=True)
    paper.write_bytes(b"%PDF-1.4")

    monkeypatch.setattr("src.jobs.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("src.jobs.ingestion.discover_pdfs", lambda: [paper])
    monkeypatch.setattr(
        "src.jobs.ingestion.rebuild_vector_store_from_pdfs",
        lambda paths: (object(), 12),
    )

    store = LocalJobStore(tmp_path / "jobs.jsonl")
    runner = LocalJobRunner(store)
    job = runner.run_index_rebuild(["papers/library/paper.pdf"])

    assert job.status == "succeeded"
    assert job.kind == "index_rebuild"
    assert job.result["paper_count"] == 1
    assert job.result["chunk_count"] == 12


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
    manager = AsyncJobManager(store)

    queued = manager.enqueue_local_python(
        CodeExecutionRequest(
            language="python",
            entrypoint="main.py",
            files={"main.py": "print('async-ok')"},
        )
    )
    finished = wait_for_status(store, queued.job_id, {"succeeded"})

    assert queued.status == "queued"
    assert finished.result["stdout"] == "async-ok\n"


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
