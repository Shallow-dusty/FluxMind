from pathlib import Path

from src.capabilities import CodeExecutionRequest, ImageGenerationRequest
from src.jobs import LocalJobRunner, LocalJobStore


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


def test_local_python_job_persists_execution_result(tmp_path: Path):
    store = LocalJobStore(tmp_path / "jobs.jsonl")
    runner = LocalJobRunner(store)

    job = runner.run_local_python(
        CodeExecutionRequest(
            language="python",
            entrypoint="main.py",
            files={"main.py": "print('job-ok')"},
        )
    )

    assert job.status == "succeeded"
    assert job.result["stdout"] == "job-ok\n"
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
