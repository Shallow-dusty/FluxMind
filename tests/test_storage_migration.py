import json
from datetime import datetime, timedelta, timezone

import src.storage_migration as storage_migration
from src.jobs import JobRecord, LocalJobStore
from src.storage_migration import (
    collect_platform_migration_rehearsal,
    collect_object_storage_migration_manifest,
    collect_job_store_migration_manifest,
    format_job_store_migration_manifest_markdown,
    format_job_store_migration_verify_markdown,
    format_object_storage_migration_manifest_markdown,
    format_object_storage_migration_verify_markdown,
    format_storage_migration_rehearsal_markdown,
    run_storage_migration_rehearsal,
    verify_job_store_migration_manifest,
    verify_object_storage_migration_manifest,
)


def _seed_runtime(root):
    metadata = root / "metadata"
    metadata.mkdir(parents=True)
    (metadata / "corpus.json").write_text(
        json.dumps({"version": 1, "papers": []}),
        encoding="utf-8",
    )
    (metadata / "runtime_events.jsonl").write_text("", encoding="utf-8")
    jobs = root / "jobs"
    jobs.mkdir()
    LocalJobStore(jobs / "jobs.jsonl")._ensure_sqlite()
    artifacts = root / "artifacts"
    artifacts.mkdir()
    (artifacts / "plot.svg").write_text("<svg />", encoding="utf-8")
    nested = artifacts / "nested"
    nested.mkdir()
    (nested / "plot.json").write_text("{}", encoding="utf-8")
    uploads = root / "papers" / "uploads"
    uploads.mkdir(parents=True)
    (uploads / "paper.pdf").write_bytes(b"%PDF")
    index = root / "faiss_index"
    index.mkdir()
    (index / "active_papers.json").write_text("[]", encoding="utf-8")
    (root / ".env").write_text("SECRET_TOKEN=do-not-copy\n", encoding="utf-8")


def _append_sensitive_job(root, *, status="queued"):
    jobs = root / "jobs"
    jobs.mkdir(exist_ok=True)
    store = LocalJobStore(jobs / "jobs.jsonl")
    record = JobRecord(
        job_id="secret-job-id",
        kind="code_execution",
        status=status,
        created_at="2026-06-16T00:00:00+00:00",
        updated_at="2026-06-16T00:00:00+00:00",
        request={"entrypoint": "main.py", "source": "hunter2-source"},
        result={"stdout": "hunter2-stdout"} if status == "succeeded" else None,
        artifacts=[{"uri": "file:///private/artifact.txt"}] if status == "succeeded" else [],
        error={"message": "hunter2-error"} if status == "failed" else None,
        attempts=1,
        request_id="req-secret",
        worker_id="worker-secret",
        leased_at="2026-06-16T00:00:00+00:00",
        lease_expires_at="2999-06-16T00:00:00+00:00",
        idempotency_key="idempotency-secret",
        max_attempts=2,
        retry_backoff_s=5,
        owner_id="owner-secret",
        owner_label="Owner Secret",
        ownership_source="request",
        logs=[{"message": "hunter2-log"}],
    )
    store.append_new(record)
    return store


def test_storage_migration_rehearsal_copies_and_verifies_runtime_without_secrets(tmp_path):
    source = tmp_path / "source"
    staging = tmp_path / "staging"
    source.mkdir()
    _seed_runtime(source)

    status = run_storage_migration_rehearsal(
        project_root=source,
        staging_root=staging,
        generated_at="2026-06-16T00:00:00+00:00",
    )

    payload = json.dumps(status, ensure_ascii=False, sort_keys=True)
    assert status["rehearsal_ok"] is True
    assert status["content_copied_to_staging"] is True
    assert status["content_exported_in_report"] is False
    assert status["secrets_copied"] is False
    assert status["secrets_exported"] is False
    assert status["summary"]["source_preflight_ok"] is True
    assert status["summary"]["restore_check_ok"] is True
    assert status["summary"]["staged_storage_schema_ok"] is True
    assert status["summary"]["copied_files"] >= 5
    assert (staging / "metadata" / "corpus.json").exists()
    assert (staging / "jobs" / "jobs.sqlite3").exists()
    assert not (staging / ".env").exists()
    assert "do-not-copy" not in payload


def test_platform_migration_rehearsal_public_projection_omits_raw_manifests(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _seed_runtime(source)
    _append_sensitive_job(source)

    status = collect_platform_migration_rehearsal(
        project_root=source,
        generated_at="2026-06-16T00:00:00+00:00",
    )
    payload = json.dumps(status, ensure_ascii=False, sort_keys=True)
    markdown = format_storage_migration_rehearsal_markdown(status)

    assert status["mode"] == "local_runtime_migration_rehearsal"
    assert status["rehearsal_ok"] is True
    assert status["raw_manifests_included"] is False
    assert status["paths_exported"] is False
    assert status["staging_root_retained"] is False
    assert status["summary"]["object_manifest_ready"] is True
    assert status["summary"]["job_store_manifest_ready"] is True
    assert "object_storage_manifest" not in status
    assert "job_store_manifest" not in status
    for secret in (
        "secret-job-id",
        "hunter2-source",
        "hunter2-stdout",
        "owner-secret",
        "idempotency-secret",
        "paper.pdf",
        "corpus.json",
        "do-not-copy",
        "file://",
        str(source),
    ):
        assert secret not in payload
        assert secret not in markdown


def test_object_storage_migration_manifest_uses_opaque_keys_without_paths_or_secrets(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _seed_runtime(source)
    private_upload = source / "papers" / "uploads" / "private-study.pdf"
    private_upload.write_bytes(b"%PDF-private")

    manifest = collect_object_storage_migration_manifest(
        project_root=source,
        key_prefix="s3://hidden-bucket/private-prefix",
        generated_at="2026-06-16T00:00:00+00:00",
    )

    payload = json.dumps(manifest, ensure_ascii=False, sort_keys=True)
    assert manifest["mode"] == "object_storage_migration_manifest"
    assert manifest["content_exported"] is False
    assert manifest["secrets_exported"] is False
    assert manifest["source_paths_exported"] is False
    assert manifest["filenames_exported"] is False
    assert manifest["bucket_exported"] is False
    assert manifest["external_connectivity_checked"] is False
    assert manifest["key_prefix"] == "fluxmind-runtime"
    assert manifest["object_count"] >= 6
    assert manifest["unique_object_count"] >= 1
    assert all(item["object_key"].startswith("fluxmind-runtime/") for item in manifest["objects"])
    assert all(item["sha256"] in item["object_key"] for item in manifest["objects"])
    assert all(item["source_path_token"] for item in manifest["objects"])
    assert "private-study.pdf" not in payload
    assert "paper.pdf" not in payload
    assert "corpus.json" not in payload
    assert "do-not-copy" not in payload
    assert "hidden-bucket" not in payload
    assert "s3://" not in payload


def test_object_storage_migration_manifest_markdown_is_no_secret(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _seed_runtime(source)

    manifest = collect_object_storage_migration_manifest(
        project_root=source,
        generated_at="2026-06-16T00:00:00+00:00",
    )
    markdown = format_object_storage_migration_manifest_markdown(manifest)

    assert "# FluxMind Object Storage Migration Manifest" in markdown
    assert "Content exported: false" in markdown
    assert "Source paths exported: false" in markdown
    assert "paper.pdf" not in markdown
    assert "do-not-copy" not in markdown


def test_object_storage_migration_manifest_verify_accepts_matching_runtime(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _seed_runtime(source)

    manifest = collect_object_storage_migration_manifest(
        project_root=source,
        generated_at="2026-06-16T00:00:00+00:00",
    )
    status = verify_object_storage_migration_manifest(
        manifest,
        project_root=source,
        generated_at="2026-06-16T00:00:00+00:00",
    )
    markdown = format_object_storage_migration_verify_markdown(status)
    payload = json.dumps(status, ensure_ascii=False, sort_keys=True)

    assert status["mode"] == "object_storage_migration_manifest_verify"
    assert status["ok"] is True
    assert status["checked_objects"] == manifest["object_count"]
    assert status["missing_objects"] == 0
    assert status["mismatched_objects"] == 0
    assert status["extra_objects"] == 0
    assert status["manifest_errors"] == []
    assert status["source_paths_exported"] is False
    assert status["filenames_exported"] is False
    assert "Verification OK: true" in markdown
    assert "paper.pdf" not in payload
    assert "corpus.json" not in payload
    assert "do-not-copy" not in payload
    assert "paper.pdf" not in markdown


def test_object_storage_migration_manifest_verify_accepts_rehearsal_output(tmp_path):
    source = tmp_path / "source"
    staging = tmp_path / "staging"
    source.mkdir()
    _seed_runtime(source)
    rehearsal = run_storage_migration_rehearsal(
        project_root=source,
        staging_root=staging,
        include_object_manifest=True,
        generated_at="2026-06-16T00:00:00+00:00",
    )

    status = verify_object_storage_migration_manifest(
        rehearsal,
        project_root=source,
        generated_at="2026-06-16T00:00:00+00:00",
    )

    assert status["ok"] is True
    assert status["checked_objects"] == rehearsal["object_storage_manifest"]["object_count"]


def test_object_storage_migration_manifest_verify_detects_changed_runtime_file(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _seed_runtime(source)
    manifest = collect_object_storage_migration_manifest(
        project_root=source,
        generated_at="2026-06-16T00:00:00+00:00",
    )
    (source / "artifacts" / "plot.svg").write_text("<svg>changed</svg>", encoding="utf-8")

    status = verify_object_storage_migration_manifest(
        manifest,
        project_root=source,
        generated_at="2026-06-16T00:00:00+00:00",
    )

    assert status["ok"] is False
    assert status["mismatched_objects"] == 1
    assert status["missing_objects"] == 0
    assert status["extra_objects"] == 0
    [difference] = status["object_differences"]
    assert difference["group"] == "artifacts"
    assert difference["status"] == "mismatched"
    assert difference["sha256_match"] is False
    assert difference["object_key_match"] is False


def test_object_storage_migration_manifest_verify_rejects_secret_bearing_manifest(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _seed_runtime(source)
    manifest = collect_object_storage_migration_manifest(
        project_root=source,
        generated_at="2026-06-16T00:00:00+00:00",
    )
    manifest["source_paths_exported"] = True
    manifest["bucket"] = "hidden-bucket"
    manifest["groups"][0]["sourcePath"] = "/private/group-source.pdf"
    manifest["groups"][0]["fileName"] = "group-source.pdf"
    manifest["objects"][0]["sourcePath"] = "/private/paper.pdf"
    manifest["objects"][0]["metadata"] = {"credential": "object-credential-secret"}

    status = verify_object_storage_migration_manifest(
        manifest,
        project_root=source,
        generated_at="2026-06-16T00:00:00+00:00",
    )
    payload = json.dumps(status, ensure_ascii=False, sort_keys=True)
    markdown = format_object_storage_migration_verify_markdown(status)

    assert status["ok"] is False
    assert "source_paths_exported_must_be_false" in status["manifest_errors"]
    assert "manifest_contains_unsafe_field" in status["manifest_errors"]
    assert "object_entry_contains_unsafe_field" in status["manifest_errors"]
    assert "hidden-bucket" not in payload
    assert "/private/paper.pdf" not in payload
    assert "object-credential-secret" not in payload
    assert "/private/group-source.pdf" not in payload
    assert "group-source.pdf" not in payload
    assert "hidden-bucket" not in markdown
    assert "/private/paper.pdf" not in markdown
    assert "object-credential-secret" not in markdown
    assert "/private/group-source.pdf" not in markdown
    assert "group-source.pdf" not in markdown


def test_object_storage_migration_manifest_verify_omits_unsafe_group_and_prefix_values(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _seed_runtime(source)
    manifest = collect_object_storage_migration_manifest(
        project_root=source,
        generated_at="2026-06-16T00:00:00+00:00",
    )
    unsafe_object = dict(manifest["objects"][0])
    unsafe_object["group"] = "hunter2-source"
    unsafe_object["object_key"] = (
        f"{manifest['key_prefix']}/hunter2-source/"
        f"{unsafe_object['sha256'][:2]}/{unsafe_object['sha256']}"
    )
    manifest["key_prefix"] = "/private/source-root"
    manifest["groups"].append(
        {
            "name": "/private/group-source.pdf",
            "restore_priority": "operator_state",
            "source_exists": True,
            "source_is_directory": True,
            "object_count": 0,
            "bytes": 0,
        }
    )
    manifest["objects"].append(unsafe_object)
    manifest["group_count"] = len(manifest["groups"])
    manifest["object_count"] = len(manifest["objects"])

    status = verify_object_storage_migration_manifest(
        manifest,
        project_root=source,
        generated_at="2026-06-16T00:00:00+00:00",
    )
    payload = json.dumps(status, ensure_ascii=False, sort_keys=True)
    markdown = format_object_storage_migration_verify_markdown(status)

    assert status["ok"] is False
    assert status["key_prefix"] == "fluxmind-runtime"
    assert "key_prefix_invalid" in status["manifest_errors"]
    assert "group_entry_invalid" in status["manifest_errors"]
    assert "object_entry_group_unknown" in status["manifest_errors"]
    for secret in (
        "/private/source-root",
        "/private/group-source.pdf",
        "hunter2-source",
    ):
        assert secret not in payload
        assert secret not in markdown


def test_job_store_migration_manifest_uses_tokens_without_payload_or_ids(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _seed_runtime(source)
    _append_sensitive_job(source)

    manifest = collect_job_store_migration_manifest(
        project_root=source,
        generated_at="2026-06-16T00:00:00+00:00",
    )
    markdown = format_job_store_migration_manifest_markdown(manifest)
    payload = json.dumps(manifest, ensure_ascii=False, sort_keys=True)

    assert manifest["mode"] == "job_store_migration_manifest"
    assert manifest["ok"] is True
    assert manifest["content_exported"] is False
    assert manifest["secrets_exported"] is False
    assert manifest["payload_exported"] is False
    assert manifest["owner_ids_exported"] is False
    assert manifest["request_ids_exported"] is False
    assert manifest["worker_ids_exported"] is False
    assert manifest["idempotency_keys_exported"] is False
    assert manifest["job_count"] == 1
    assert manifest["idempotency_claim_count"] == 1
    assert manifest["by_status"] == {"queued": 1}
    assert manifest["by_kind"] == {"code_execution": 1}
    assert manifest["by_ownership_source"] == {"request": 1}
    assert manifest["queue_summary"]["queued"] == 1
    assert manifest["queue_summary"]["leased"] == 1
    assert manifest["queue_summary"]["idempotency_claimed_jobs"] == 1
    assert len(manifest["jobs"][0]["job_token"]) == 64
    assert len(manifest["idempotency_claims"][0]["claim_token"]) == 64
    for secret in (
        "secret-job-id",
        "hunter2-source",
        "hunter2-stdout",
        "hunter2-log",
        "req-secret",
        "worker-secret",
        "idempotency-secret",
        "owner-secret",
        "Owner Secret",
        "private/artifact",
    ):
        assert secret not in payload
        assert secret not in markdown


def test_job_store_migration_manifest_verify_accepts_matching_runtime(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _seed_runtime(source)
    _append_sensitive_job(source)
    manifest = collect_job_store_migration_manifest(
        project_root=source,
        generated_at="2026-06-16T00:00:00+00:00",
    )

    status = verify_job_store_migration_manifest(
        manifest,
        project_root=source,
        generated_at="2026-06-16T00:00:00+00:00",
    )
    markdown = format_job_store_migration_verify_markdown(status)
    payload = json.dumps(status, ensure_ascii=False, sort_keys=True)

    assert status["mode"] == "job_store_migration_manifest_verify"
    assert status["ok"] is True
    assert status["expected_jobs"] == 1
    assert status["current_jobs"] == 1
    assert status["missing_jobs"] == 0
    assert status["mismatched_jobs"] == 0
    assert status["extra_jobs"] == 0
    assert status["expected_idempotency_claims"] == 1
    assert status["missing_idempotency_claims"] == 0
    assert "Verification OK: true" in markdown
    assert "hunter2-source" not in payload
    assert "owner-secret" not in payload
    assert "idempotency-secret" not in markdown


def test_job_store_migration_manifest_verify_uses_manifest_time_for_scheduled_jobs(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _seed_runtime(source)
    now = datetime.now(timezone.utc)
    manifest_time = now - timedelta(hours=2)
    not_before = now - timedelta(hours=1)
    _append_sensitive_job(source)
    store = LocalJobStore(source / "jobs" / "jobs.jsonl")
    scheduled = JobRecord(
        job_id="secret-job-id",
        kind="code_execution",
        status="queued",
        created_at=manifest_time.isoformat(),
        updated_at=manifest_time.isoformat(),
        request={"entrypoint": "main.py", "source": "hunter2-source"},
        attempts=1,
        not_before=not_before.isoformat(),
        idempotency_key="idempotency-secret",
        max_attempts=2,
        retry_backoff_s=5,
        owner_id="owner-secret",
        owner_label="Owner Secret",
        ownership_source="request",
    )
    store.append(scheduled)
    manifest = collect_job_store_migration_manifest(
        project_root=source,
        generated_at=manifest_time.isoformat(),
    )

    status = verify_job_store_migration_manifest(manifest, project_root=source)

    assert manifest["jobs"][0]["is_scheduled"] is True
    assert manifest["jobs"][0]["is_due"] is False
    assert status["ok"] is True
    assert status["mismatched_jobs"] == 0


def test_job_store_migration_manifest_verify_detects_changed_job_state(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _seed_runtime(source)
    store = _append_sensitive_job(source)
    manifest = collect_job_store_migration_manifest(
        project_root=source,
        generated_at="2026-06-16T00:00:00+00:00",
    )
    changed = JobRecord(
        job_id="secret-job-id",
        kind="code_execution",
        status="succeeded",
        created_at="2026-06-16T00:00:00+00:00",
        updated_at="2026-06-16T00:01:00+00:00",
        request={"entrypoint": "main.py", "source": "hunter2-source"},
        result={"stdout": "hunter2-stdout"},
        artifacts=[],
        attempts=2,
        idempotency_key="idempotency-secret",
        max_attempts=2,
        retry_backoff_s=5,
        owner_id="owner-secret",
        owner_label="Owner Secret",
        ownership_source="request",
    )
    store.append(changed)

    status = verify_job_store_migration_manifest(
        manifest,
        project_root=source,
        generated_at="2026-06-16T00:00:00+00:00",
    )

    assert status["ok"] is False
    assert status["missing_jobs"] == 0
    assert status["mismatched_jobs"] == 1
    assert status["extra_jobs"] == 0
    [difference] = status["job_differences"]
    assert difference["status"] == "mismatched"
    assert "status" in difference["changed_fields"]
    assert "attempts" in difference["changed_fields"]


def test_job_store_migration_manifest_verify_detects_changed_claim_metadata(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _seed_runtime(source)
    _append_sensitive_job(source)
    manifest = collect_job_store_migration_manifest(
        project_root=source,
        generated_at="2026-06-16T00:00:00+00:00",
    )
    manifest["idempotency_claims"][0]["kind"] = "image_generation"

    status = verify_job_store_migration_manifest(
        manifest,
        project_root=source,
        generated_at="2026-06-16T00:00:00+00:00",
    )
    payload = json.dumps(status, ensure_ascii=False, sort_keys=True)
    markdown = format_job_store_migration_verify_markdown(status)

    assert status["ok"] is False
    assert status["missing_idempotency_claims"] == 0
    assert status["mismatched_idempotency_claims"] == 1
    assert status["extra_idempotency_claims"] == 0
    [difference] = status["idempotency_claim_differences"]
    assert difference["status"] == "mismatched"
    assert difference["changed_fields"] == ["kind"]
    assert "image_generation" not in payload
    assert "image_generation" not in markdown


def test_job_store_migration_manifest_verify_rejects_secret_bearing_manifest(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _seed_runtime(source)
    _append_sensitive_job(source)
    manifest = collect_job_store_migration_manifest(
        project_root=source,
        generated_at="2026-06-16T00:00:00+00:00",
    )
    manifest["owner_id"] = "owner-secret"
    manifest["storage"]["ownerId"] = "storage-owner-secret"
    manifest["timeline"]["requestId"] = "timeline-request-secret"
    manifest["jobs"][0]["payload"] = {"request": "hunter2-source"}
    manifest["jobs"][0]["workerId"] = "worker-secret"
    manifest["idempotency_claims"][0]["idempotency_key"] = "idempotency-secret"
    manifest["idempotency_claims"][0]["idempotencyKey"] = "idempotency-camel-secret"

    status = verify_job_store_migration_manifest(
        manifest,
        project_root=source,
        generated_at="2026-06-16T00:00:00+00:00",
    )
    payload = json.dumps(status, ensure_ascii=False, sort_keys=True)
    markdown = format_job_store_migration_verify_markdown(status)

    assert status["ok"] is False
    assert "manifest_contains_unsafe_field" in status["manifest_errors"]
    assert "job_entry_contains_unsafe_field" in status["manifest_errors"]
    assert "claim_entry_contains_unsafe_field" in status["manifest_errors"]
    assert "owner-secret" not in payload
    assert "storage-owner-secret" not in payload
    assert "timeline-request-secret" not in payload
    assert "hunter2-source" not in payload
    assert "worker-secret" not in payload
    assert "idempotency-secret" not in markdown
    assert "idempotency-camel-secret" not in markdown


def test_job_store_migration_manifest_verify_omits_manifest_kind_and_status_values(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _seed_runtime(source)
    _append_sensitive_job(source)
    manifest = collect_job_store_migration_manifest(
        project_root=source,
        generated_at="2026-06-16T00:00:00+00:00",
    )
    manifest["jobs"][0]["kind"] = "hunter2-kind"
    manifest["jobs"][0]["status"] = "hunter2-status"

    status = verify_job_store_migration_manifest(
        manifest,
        project_root=source,
        generated_at="2026-06-16T00:00:00+00:00",
    )
    payload = json.dumps(status, ensure_ascii=False, sort_keys=True)
    markdown = format_job_store_migration_verify_markdown(status)

    assert status["ok"] is False
    assert status["mismatched_jobs"] == 1
    [difference] = status["job_differences"]
    assert difference == {
        "job_token": manifest["jobs"][0]["job_token"],
        "status": "mismatched",
        "metadata_match": False,
        "changed_fields": ["kind", "status"],
    }
    assert "hunter2-kind" not in payload
    assert "hunter2-status" not in payload
    assert "hunter2-kind" not in markdown
    assert "hunter2-status" not in markdown


def test_storage_migration_rehearsal_can_include_object_manifest(tmp_path):
    source = tmp_path / "source"
    staging = tmp_path / "staging"
    source.mkdir()
    _seed_runtime(source)

    status = run_storage_migration_rehearsal(
        project_root=source,
        staging_root=staging,
        include_object_manifest=True,
        object_key_prefix="lab-runtime",
        generated_at="2026-06-16T00:00:00+00:00",
    )

    payload = json.dumps(status, ensure_ascii=False, sort_keys=True)
    assert status["rehearsal_ok"] is True
    assert status["summary"]["object_manifest_ready"] is True
    assert status["summary"]["object_manifest_objects"] >= 5
    assert status["object_storage_manifest"]["mode"] == "object_storage_migration_manifest"
    assert status["object_storage_manifest"]["key_prefix"] == "lab-runtime"
    assert status["object_storage_manifest"]["source_paths_exported"] is False
    assert "do-not-copy" not in payload
    assert "paper.pdf" not in payload


def test_storage_migration_rehearsal_can_include_job_store_manifest(tmp_path):
    source = tmp_path / "source"
    staging = tmp_path / "staging"
    source.mkdir()
    _seed_runtime(source)
    _append_sensitive_job(source)

    status = run_storage_migration_rehearsal(
        project_root=source,
        staging_root=staging,
        include_job_store_manifest=True,
        generated_at="2026-06-16T00:00:00+00:00",
    )
    payload = json.dumps(status, ensure_ascii=False, sort_keys=True)
    markdown = format_storage_migration_rehearsal_markdown(status)

    assert status["rehearsal_ok"] is True
    assert status["summary"]["job_store_manifest_ready"] is True
    assert status["summary"]["job_store_manifest_jobs"] == 1
    assert status["summary"]["job_store_manifest_claims"] == 1
    assert status["job_store_manifest"]["mode"] == "job_store_migration_manifest"
    assert status["job_store_manifest"]["payload_exported"] is False
    assert "Job-store manifest ready: true" in markdown
    assert "hunter2-source" not in payload
    assert "owner-secret" not in payload
    assert "idempotency-secret" not in markdown


def test_storage_migration_rehearsal_generates_timestamp_when_not_supplied(tmp_path):
    source = tmp_path / "source"
    staging = tmp_path / "staging"
    source.mkdir()
    _seed_runtime(source)

    status = run_storage_migration_rehearsal(project_root=source, staging_root=staging)

    assert status["rehearsal_ok"] is True
    assert status["generated_at"]


def test_storage_migration_rehearsal_rejects_nonempty_staging_without_overwrite(tmp_path):
    source = tmp_path / "source"
    staging = tmp_path / "staging"
    source.mkdir()
    staging.mkdir()
    marker = staging / "marker.txt"
    marker.write_text("keep", encoding="utf-8")
    _seed_runtime(source)

    status = run_storage_migration_rehearsal(
        project_root=source,
        staging_root=staging,
        generated_at="2026-06-16T00:00:00+00:00",
    )

    assert status["rehearsal_ok"] is False
    assert "staging_root_not_empty" in status["blockers"]
    assert marker.read_text(encoding="utf-8") == "keep"


def test_storage_migration_rehearsal_can_overwrite_staging(tmp_path):
    source = tmp_path / "source"
    staging = tmp_path / "staging"
    source.mkdir()
    staging.mkdir()
    (staging / "marker.txt").write_text("remove", encoding="utf-8")
    _seed_runtime(source)

    status = run_storage_migration_rehearsal(
        project_root=source,
        staging_root=staging,
        overwrite_staging=True,
        generated_at="2026-06-16T00:00:00+00:00",
    )

    assert status["rehearsal_ok"] is True
    assert status["staging_root_overwritten"] is True
    assert not (staging / "marker.txt").exists()


def test_storage_migration_rehearsal_rejects_staging_inside_project(tmp_path):
    _seed_runtime(tmp_path)

    status = run_storage_migration_rehearsal(
        project_root=tmp_path,
        staging_root=tmp_path / "staging",
        generated_at="2026-06-16T00:00:00+00:00",
    )

    assert status["rehearsal_ok"] is False
    assert "staging_root_inside_project" in status["blockers"]


def test_storage_migration_rehearsal_rejects_staging_parent_without_deleting_source(tmp_path):
    staging = tmp_path / "parent"
    source = staging / "source"
    source.mkdir(parents=True)
    _seed_runtime(source)
    source_marker = source / "artifacts" / "source.txt"
    source_marker.write_text("keep-source", encoding="utf-8")
    parent_marker = staging / "parent.txt"
    parent_marker.write_text("keep-parent", encoding="utf-8")

    status = run_storage_migration_rehearsal(
        project_root=source,
        staging_root=staging,
        overwrite_staging=True,
        generated_at="2026-06-16T00:00:00+00:00",
    )

    assert status["rehearsal_ok"] is False
    assert status["content_copied_to_staging"] is False
    assert status["staging_root_overwritten"] is False
    assert "staging_root_contains_project" in status["blockers"]
    assert source_marker.read_text(encoding="utf-8") == "keep-source"
    assert parent_marker.read_text(encoding="utf-8") == "keep-parent"


def test_storage_migration_rehearsal_skips_symlinks(tmp_path):
    source = tmp_path / "source"
    staging = tmp_path / "staging"
    source.mkdir()
    _seed_runtime(source)
    (source / "artifacts" / "leak").symlink_to(source / ".env")

    status = run_storage_migration_rehearsal(
        project_root=source,
        staging_root=staging,
        generated_at="2026-06-16T00:00:00+00:00",
    )

    assert status["rehearsal_ok"] is True
    assert status["summary"]["skipped_symlinks"] == 1
    assert not (staging / "artifacts" / "leak").exists()


def test_storage_migration_rehearsal_can_include_runtime_dependencies(tmp_path):
    source = tmp_path / "source"
    staging = tmp_path / "staging"
    source.mkdir()
    _seed_runtime(source)
    models = source / "models"
    models.mkdir()
    (models / "model.bin").write_bytes(b"model")

    status = run_storage_migration_rehearsal(
        project_root=source,
        staging_root=staging,
        include_runtime_dependencies=True,
        generated_at="2026-06-16T00:00:00+00:00",
    )

    groups = {group["name"]: group for group in status["copy"]["groups"]}
    assert status["rehearsal_ok"] is True
    assert groups["models"]["status"] == "copied"
    assert (staging / "models" / "model.bin").exists()


def test_storage_migration_copy_group_reports_runtime_dependency_skip(tmp_path):
    source = tmp_path / "models"
    source.mkdir()

    result = storage_migration._copy_runtime_group(
        name="models",
        source_path=source,
        target_path=tmp_path / "staging" / "models",
        restore_priority="runtime_dependency",
        include_runtime_dependencies=False,
    )

    assert result["status"] == "skipped_runtime_dependency"
    assert result["source_exists"] is True


def test_storage_migration_copy_group_rejects_file_source(tmp_path):
    source = tmp_path / "metadata"
    source.write_text("not a directory", encoding="utf-8")

    result = storage_migration._copy_runtime_group(
        name="metadata",
        source_path=source,
        target_path=tmp_path / "staging" / "metadata",
        restore_priority="required",
        include_runtime_dependencies=False,
    )

    assert result["status"] == "source_not_directory"
    assert result["errors"] == ["source_not_directory"]


def test_storage_migration_copy_group_reports_copy_failure(tmp_path, monkeypatch):
    source = tmp_path / "metadata"
    source.mkdir()
    (source / "corpus.json").write_text("{}", encoding="utf-8")

    def fail_copy(*_args, **_kwargs):
        raise OSError("copy failed")

    monkeypatch.setattr(storage_migration.shutil, "copy2", fail_copy)
    result = storage_migration._copy_runtime_group(
        name="metadata",
        source_path=source,
        target_path=tmp_path / "staging" / "metadata",
        restore_priority="required",
        include_runtime_dependencies=False,
    )

    assert result["status"] == "copy_failed"
    assert result["errors"] == ["copy_failed"]


def test_storage_migration_rehearsal_reports_source_preflight_failure(tmp_path, monkeypatch):
    source = tmp_path / "source"
    staging = tmp_path / "staging"
    source.mkdir()
    _seed_runtime(source)
    monkeypatch.setattr(
        storage_migration,
        "collect_platform_migration_preflight",
        lambda **_kwargs: {"preflight_ok": False, "activation_ready": False, "blockers": {}},
    )

    status = run_storage_migration_rehearsal(
        project_root=source,
        staging_root=staging,
        generated_at="2026-06-16T00:00:00+00:00",
    )

    assert status["rehearsal_ok"] is False
    assert "source_preflight_failed" in status["blockers"]


def test_storage_migration_rehearsal_reports_staged_restore_failure(tmp_path, monkeypatch):
    source = tmp_path / "source"
    staging = tmp_path / "staging"
    source.mkdir()
    _seed_runtime(source)
    monkeypatch.setattr(
        storage_migration,
        "collect_runtime_restore_check",
        lambda *_args, **_kwargs: {"ok": False, "manifest_errors": ["bad_manifest"]},
    )

    status = run_storage_migration_rehearsal(
        project_root=source,
        staging_root=staging,
        generated_at="2026-06-16T00:00:00+00:00",
    )

    assert status["rehearsal_ok"] is False
    assert "staged_restore_check_failed" in status["blockers"]


def test_storage_migration_rehearsal_reports_staged_schema_drift(tmp_path, monkeypatch):
    source = tmp_path / "source"
    staging = tmp_path / "staging"
    source.mkdir()
    _seed_runtime(source)
    monkeypatch.setattr(
        storage_migration,
        "storage_schema_status_for_root",
        lambda _root: {"ok": False, "problem_count": 1, "stores": []},
    )

    status = run_storage_migration_rehearsal(
        project_root=source,
        staging_root=staging,
        generated_at="2026-06-16T00:00:00+00:00",
    )

    assert status["rehearsal_ok"] is False
    assert "staged_storage_schema_drift" in status["blockers"]


def test_format_storage_migration_rehearsal_markdown_is_no_secret(tmp_path):
    source = tmp_path / "source"
    staging = tmp_path / "staging"
    source.mkdir()
    _seed_runtime(source)

    status = run_storage_migration_rehearsal(
        project_root=source,
        staging_root=staging,
        generated_at="2026-06-16T00:00:00+00:00",
    )
    markdown = format_storage_migration_rehearsal_markdown(status)

    assert "# FluxMind Runtime Migration Rehearsal" in markdown
    assert "Rehearsal OK: true" in markdown
    assert "Content exported in report: false" in markdown
    assert "Secrets copied: false" in markdown
    assert "do-not-copy" not in markdown
