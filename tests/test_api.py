import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import api
from src.jobs import AsyncJobManager
from src.metadata import PaperRecord
from src.artifacts import artifact_id_for_uri


def test_verify_api_token_allows_when_unconfigured(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")

    api.verify_api_token(None, None)


def test_verify_api_token_accepts_x_api_key(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "secret")

    api.verify_api_token(None, "secret")


def test_verify_api_token_accepts_bearer(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "secret")

    api.verify_api_token("Bearer secret", None)


def test_verify_api_token_rejects_invalid_token(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "secret")

    with pytest.raises(HTTPException) as exc:
        api.verify_api_token("Bearer wrong", None)

    assert exc.value.status_code == 401


def test_query_response_includes_request_id(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")

    def fake_query(question, *, answer_mode):
        return f"{answer_mode}: {question}"

    monkeypatch.setattr(api, "query", fake_query)

    client = TestClient(api.app)
    response = client.post(
        "/query",
        json={"question": "Explain SMC"},
        headers={"X-Request-ID": "req-test"},
    )

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "req-test"
    assert response.json() == {"answer": "explanation: Explain SMC", "request_id": "req-test"}


def test_query_accepts_answer_mode(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")
    seen = {}

    def fake_query(question, *, answer_mode):
        seen["question"] = question
        seen["answer_mode"] = answer_mode
        return "ok"

    monkeypatch.setattr(api, "query", fake_query)

    client = TestClient(api.app)
    response = client.post(
        "/query",
        json={"question": "Derive SMC reaching law", "answer_mode": "derivation"},
    )

    assert response.status_code == 200
    assert seen == {"question": "Derive SMC reaching law", "answer_mode": "derivation"}


def test_query_normalizes_provider_errors(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")

    def fail(_question, *, answer_mode):
        assert answer_mode == "explanation"
        raise TimeoutError("provider timed out")

    monkeypatch.setattr(api, "query", fail)

    client = TestClient(api.app)
    response = client.post(
        "/query",
        json={"question": "Explain SMC"},
        headers={"X-Request-ID": "req-timeout"},
    )

    assert response.status_code == 504
    assert response.json()["detail"] == {
        "code": "provider_timeout",
        "message": "The model provider timed out. Please retry the request.",
        "request_id": "req-timeout",
    }


def test_corpus_papers_endpoint_returns_metadata(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")
    monkeypatch.setattr(
        api,
        "refresh_paper_metadata",
        lambda: [
            PaperRecord(
                paper_id="p1",
                source_path="papers/library/paper.pdf",
                filename="paper.pdf",
                source_kind="library",
                checksum_sha256="a" * 64,
                title="Paper",
                active=True,
                indexed_status="indexed",
                chunk_count=3,
                updated_at="2026-05-30T00:00:00+00:00",
            )
        ],
    )

    client = TestClient(api.app)
    response = client.get("/corpus/papers")

    assert response.status_code == 200
    assert response.json()["papers"][0]["source_path"] == "papers/library/paper.pdf"
    assert response.json()["papers"][0]["indexed_status"] == "indexed"


def test_update_active_corpus_endpoint_persists_selection(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")

    def fake_set_active(source_paths):
        assert source_paths == ["papers/library/paper.pdf"]
        return [
            PaperRecord(
                paper_id="p1",
                source_path="papers/library/paper.pdf",
                filename="paper.pdf",
                source_kind="library",
                checksum_sha256="a" * 64,
                title="Paper",
                active=True,
                indexed_status="active",
                updated_at="2026-05-30T00:00:00+00:00",
            ),
            PaperRecord(
                paper_id="p2",
                source_path="papers/library/off.pdf",
                filename="off.pdf",
                source_kind="library",
                checksum_sha256="b" * 64,
                title="Off",
                active=False,
                indexed_status="available",
                updated_at="2026-05-30T00:00:00+00:00",
            ),
        ]

    monkeypatch.setattr(api, "set_active_paper_source_paths", fake_set_active)

    client = TestClient(api.app)
    response = client.put(
        "/corpus/active",
        json={"source_paths": ["papers/library/paper.pdf"]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["active_source_paths"] == ["papers/library/paper.pdf"]
    assert payload["rebuild_required"] is True
    assert payload["papers"][1]["active"] is False


def test_update_active_corpus_endpoint_rejects_invalid_selection(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")

    def fail(_source_paths):
        raise ValueError("PDF path is not in the selectable corpus: missing.pdf")

    monkeypatch.setattr(api, "set_active_paper_source_paths", fail)

    client = TestClient(api.app)
    response = client.put("/corpus/active", json={"source_paths": ["missing.pdf"]})

    assert response.status_code == 400
    assert "selectable corpus" in response.json()["detail"]


def test_admin_status_endpoint_returns_no_secret_status(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")

    class FakeStatus:
        def to_dict(self):
            return {
                "jobs": {"total": 0},
                "config": {
                    "llm_model": "test-model",
                    "external_providers_enabled": False,
                },
            }

    monkeypatch.setattr(api, "collect_admin_status", lambda: FakeStatus())

    client = TestClient(api.app)
    response = client.get("/admin/status")

    assert response.status_code == 200
    payload = response.json()["status"]
    assert payload["jobs"]["total"] == 0
    assert payload["config"]["llm_model"] == "test-model"
    assert "api_key" not in str(payload).lower()


def test_artifact_list_and_download_endpoints(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")
    artifact_root = tmp_path / "artifacts"
    artifact_path = artifact_root / "code-runs" / "run" / "result.txt"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text("artifact-body", encoding="utf-8")
    uri = artifact_path.resolve().as_uri()
    store = api.LocalJobStore(tmp_path / "jobs.jsonl")
    runner = api.LocalJobRunner(store)
    job = runner.run_local_python(
        api.CodeExecutionRequest(
            language="python",
            entrypoint="main.py",
            files={"main.py": "raise SystemExit(1)"},
        )
    )
    job.status = "succeeded"
    job.artifacts = [
        {
            "kind": "text",
            "uri": uri,
            "mime_type": "text/plain",
            "title": "result.txt",
            "metadata": {"provider": "local"},
        }
    ]
    store.append(job)
    monkeypatch.setattr("src.artifacts.ARTIFACTS_DIR", artifact_root)
    monkeypatch.setattr("src.artifacts.LocalJobStore", lambda: store)

    client = TestClient(api.app)
    listed = client.get("/artifacts")
    artifact_id = artifact_id_for_uri(uri)
    downloaded = client.get(f"/artifacts/{artifact_id}")

    assert listed.status_code == 200
    assert listed.json()["artifacts"][0]["artifact_id"] == artifact_id
    assert downloaded.status_code == 200
    assert downloaded.text == "artifact-body"


def test_mock_image_job_endpoint_returns_persisted_job(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")
    monkeypatch.setattr("src.jobs.JOBS_FILE", tmp_path / "jobs.jsonl")
    monkeypatch.setattr("src.providers.ARTIFACTS_DIR", tmp_path / "artifacts")

    client = TestClient(api.app)
    response = client.post(
        "/jobs/image/mock",
        json={"prompt": "Draw an SMC observer"},
        headers={"X-Request-ID": "req-image"},
    )

    assert response.status_code == 200
    job = response.json()["job"]
    assert job["kind"] == "image_generation"
    assert job["status"] == "succeeded"
    assert job["request_id"] == "req-image"
    assert job["artifacts"][0]["mime_type"] == "image/svg+xml"
    assert job["artifacts"][0]["metadata"]["prompt"] == "Draw an SMC observer"
    assert job["artifacts"][0]["metadata"]["cost_estimate_usd"] == "0"

    loaded = client.get(f"/jobs/{job['job_id']}")
    assert loaded.status_code == 200
    assert loaded.json()["job"]["job_id"] == job["job_id"]


def test_local_python_job_endpoint_returns_execution_result(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")
    monkeypatch.setattr("src.jobs.JOBS_FILE", tmp_path / "jobs.jsonl")

    client = TestClient(api.app)
    response = client.post(
        "/jobs/code/python-local",
        json={
            "entrypoint": "main.py",
            "files": {"main.py": "print('api-job-ok')"},
        },
    )

    assert response.status_code == 200
    job = response.json()["job"]
    assert job["kind"] == "code_execution"
    assert job["status"] == "succeeded"
    assert job["result"]["stdout"] == "api-job-ok\n"


def test_local_octave_job_endpoint_returns_structured_runtime_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")
    monkeypatch.setattr("src.jobs.JOBS_FILE", tmp_path / "jobs.jsonl")
    monkeypatch.setattr("src.providers.shutil.which", lambda _name: None)

    client = TestClient(api.app)
    response = client.post(
        "/jobs/code/octave-local",
        json={
            "entrypoint": "main.m",
            "files": {"main.m": "disp('api-octave-ok');"},
        },
        headers={"X-Request-ID": "req-octave"},
    )

    assert response.status_code == 200
    job = response.json()["job"]
    assert job["kind"] == "code_execution"
    assert job["status"] == "failed"
    assert job["request_id"] == "req-octave"
    assert job["request"]["language"] == "octave"
    assert job["result"]["exit_code"] == 127
    assert job["error"]["code"] == "runtime_unavailable"
    assert "GNU Octave executable not found" in job["error"]["message"]


def test_missing_job_returns_404(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "API_TOKEN", "")
    monkeypatch.setattr("src.jobs.JOBS_FILE", tmp_path / "jobs.jsonl")

    client = TestClient(api.app)
    response = client.get("/jobs/missing")

    assert response.status_code == 404


def test_job_list_cancel_and_retry_endpoints(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")
    monkeypatch.setattr("src.jobs.JOBS_FILE", tmp_path / "jobs.jsonl")

    client = TestClient(api.app)
    created = client.post(
        "/jobs/code/python-local",
        json={
            "entrypoint": "main.py",
            "files": {"main.py": "raise SystemExit(3)"},
        },
    ).json()["job"]

    listed = client.get("/jobs").json()["jobs"]
    assert listed[0]["job_id"] == created["job_id"]

    retried = client.post(f"/jobs/{created['job_id']}/retry").json()["job"]
    assert retried["job_id"] != created["job_id"]
    assert retried["kind"] == "code_execution"

    cancel_response = client.post(f"/jobs/{created['job_id']}/cancel")
    assert cancel_response.status_code == 200


def test_scheduled_retry_endpoint_queues_backoff_job(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")
    store = api.LocalJobStore(tmp_path / "jobs.jsonl")
    manager = AsyncJobManager(store)
    monkeypatch.setattr(api, "get_async_job_manager", lambda: manager)

    client = TestClient(api.app)
    created = api.LocalJobRunner(store).run_local_python(
        api.CodeExecutionRequest(
            language="python",
            entrypoint="main.py",
            files={"main.py": "raise SystemExit(5)"},
        )
    )
    response = client.post(
        f"/jobs/{created.job_id}/retry-scheduled",
        json={"delay_s": 30},
        headers={"X-Request-ID": "req-scheduled"},
    )

    assert response.status_code == 200
    job = response.json()["job"]
    assert job["status"] == "queued"
    assert job["parent_job_id"] == created.job_id
    assert job["request_id"] == "req-scheduled"
    assert job["not_before"] is not None


def test_index_rebuild_job_endpoint(monkeypatch, tmp_path):
    paper = tmp_path / "papers" / "library" / "paper.pdf"
    paper.parent.mkdir(parents=True)
    paper.write_bytes(b"%PDF-1.4")

    monkeypatch.setattr(api, "API_TOKEN", "")
    monkeypatch.setattr("src.jobs.JOBS_FILE", tmp_path / "jobs.jsonl")
    monkeypatch.setattr("src.jobs.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("src.jobs.ingestion.discover_pdfs", lambda: [paper])
    monkeypatch.setattr(
        "src.jobs.ingestion.rebuild_vector_store_from_pdfs",
        lambda paths: (object(), 9),
    )

    client = TestClient(api.app)
    response = client.post(
        "/jobs/index/rebuild",
        json={"source_paths": ["papers/library/paper.pdf"]},
        headers={"X-Request-ID": "req-index"},
    )

    assert response.status_code == 200
    job = response.json()["job"]
    assert job["kind"] == "index_rebuild"
    assert job["status"] == "succeeded"
    assert job["request_id"] == "req-index"
    assert job["result"]["chunk_count"] == 9


def test_async_python_job_endpoint_queues_job(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")
    store = api.LocalJobStore(tmp_path / "jobs.jsonl")
    manager = AsyncJobManager(store)
    monkeypatch.setattr(api, "get_async_job_manager", lambda: manager)

    client = TestClient(api.app)
    response = client.post(
        "/jobs/async/code/python-local",
        json={
            "entrypoint": "main.py",
            "files": {"main.py": "print('async-api-ok')"},
        },
    )

    assert response.status_code == 200
    job = response.json()["job"]
    assert job["status"] == "queued"
    manager._queue.join()
    assert store.get(job["job_id"]).status == "succeeded"


def test_async_octave_job_endpoint_queues_job(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")
    monkeypatch.setattr("src.providers.shutil.which", lambda _name: None)
    store = api.LocalJobStore(tmp_path / "jobs.jsonl")
    manager = AsyncJobManager(store)
    monkeypatch.setattr(api, "get_async_job_manager", lambda: manager)

    client = TestClient(api.app)
    response = client.post(
        "/jobs/async/code/octave-local",
        json={
            "entrypoint": "main.m",
            "files": {"main.m": "disp('async-octave-ok');"},
        },
    )

    assert response.status_code == 200
    job = response.json()["job"]
    assert job["status"] == "queued"
    assert job["request"]["language"] == "octave"
    manager._queue.join()
    assert store.get(job["job_id"]).status == "failed"
