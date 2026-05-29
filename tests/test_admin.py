from src.admin import collect_admin_status
from src.jobs import JobRecord, LocalJobStore
from src.metadata import PaperRecord


def test_collect_admin_status_summarizes_local_runtime(tmp_path, monkeypatch):
    root = tmp_path
    jobs_dir = root / "jobs"
    artifacts_dir = root / "artifacts"
    metadata_dir = root / "metadata"
    index_dir = root / "faiss_index"
    artifact = artifacts_dir / "plot.png"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"plot")
    metadata_dir.mkdir()
    index_dir.mkdir()

    monkeypatch.setattr("src.jobs.JOBS_FILE", jobs_dir / "jobs.jsonl")
    monkeypatch.setattr("src.admin.PROJECT_ROOT", root)
    monkeypatch.setattr("src.admin.JOBS_DIR", jobs_dir)
    monkeypatch.setattr("src.admin.ARTIFACTS_DIR", artifacts_dir)
    monkeypatch.setattr("src.admin.METADATA_DIR", metadata_dir)
    monkeypatch.setattr("src.admin.FAISS_INDEX_DIR", index_dir)
    monkeypatch.setattr("src.admin.LLM_BASE_URL", "https://llm.example.test/v1")
    monkeypatch.setattr(
        "src.admin.refresh_paper_metadata",
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
                updated_at="2026-05-30T00:00:00+00:00",
            )
        ],
    )

    store = LocalJobStore(jobs_dir / "jobs.jsonl")
    store.append(
        JobRecord(
            job_id="job-ok",
            kind="code_execution",
            status="succeeded",
            created_at="2026-05-30T00:00:00+00:00",
            updated_at="2026-05-30T00:00:01+00:00",
            request={},
            artifacts=[
                {
                    "kind": "plot",
                    "uri": artifact.resolve().as_uri(),
                    "mime_type": "image/png",
                    "title": "plot.png",
                    "metadata": {"provider": "local"},
                }
            ],
        )
    )
    store.append(
        JobRecord(
            job_id="job-fail",
            kind="image_generation",
            status="failed",
            created_at="2026-05-30T00:00:00+00:00",
            updated_at="2026-05-30T00:00:02+00:00",
            request={},
            error={"code": "provider_error", "message": "failed"},
        )
    )

    status = collect_admin_status().to_dict()

    assert status["jobs"]["total"] == 2
    assert status["jobs"]["by_status"] == {"failed": 1, "succeeded": 1}
    assert status["jobs"]["by_kind"] == {"code_execution": 1, "image_generation": 1}
    assert status["jobs"]["latest_failed"][0]["job_id"] == "job-fail"
    assert status["artifacts"] == {"total": 1, "bytes": 4}
    assert status["corpus"] == {"papers": 1, "active": 1, "indexed": 1, "failed": 0}
    assert status["config"]["external_providers_enabled"] is False
    assert all("path" in item for item in status["runtime_dirs"])
