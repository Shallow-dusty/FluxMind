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
import time
import urllib.error
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SSH_COMMAND_TIMEOUT_FLOOR_S = 180.0


def check(condition: bool, label: str, failures: list[str]) -> None:
    status = "ok" if condition else "fail"
    print(f"{status:4} {label}")
    if not condition:
        failures.append(label)


def http_status(url: str, timeout: float, retries: int) -> int | None:
    request = urllib.request.Request(url, headers={"User-Agent": "FluxMindHealth/1.0"})
    last_status: int | None = None
    attempts = max(1, retries)
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.status
        except urllib.error.HTTPError as exc:
            last_status = exc.code
            if exc.code not in {429, 502, 503, 504}:
                return exc.code
        except OSError:
            last_status = None
        if attempt + 1 < attempts:
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
        "src/execution_policy.py",
        "src/providers.py",
        "src/jobs.py",
        "src/artifacts.py",
        "src/admin.py",
        "src/runtime.py",
        "src/api_keys.py",
        "src/metadata.py",
        "src/storage_manifest.py",
        "src/storage_schema.py",
        "src/platform_migration.py",
        "src/storage_migration.py",
        "src/product_readiness.py",
        "src/product_registry.py",
        "src/provider_readiness.py",
        "src/quality_readiness.py",
        "src/evaluation.py",
        "src/execution_templates.py",
        "eval/rag_baseline.json",
        "scripts/evaluate_rag.py",
        "scripts/runtime_manifest.py",
        "scripts/storage_schema.py",
        "scripts/platform_migration_preflight.py",
        "scripts/platform_migration_rehearsal.py",
        "scripts/product_readiness.py",
        "scripts/product_registry.py",
        "scripts/provider_readiness.py",
        "scripts/quality_readiness.py",
        "scripts/api_key_registry.py",
        "scripts/deploy_sync.py",
        "scripts/run_job_worker.py",
        "deploy/systemd/fluxmind-worker.service",
        "docs/DEPLOYMENT_STATUS.md",
        "docs/ARCHITECTURE.md",
        "docs/BACKLOG.md",
        "docs/PLATFORM_AUDIT_AND_ROADMAP.md",
        "docs/QUALITY_ROADMAP.md",
        "docs/PRODUCTION_GAP_AND_MARKET_RESEARCH.md",
        "docs/FEATURE_AUDIT.md",
        "papers/library/manifest.json",
    ]
    for relative in required:
        check((PROJECT_ROOT / relative).exists(), f"required file: {relative}", failures)

    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    check("11.FluxMind" in readme, "README records formal workspace index", failures)
    check("Previous temporary index `80` has been retired" in readme, "README records 80 retirement", failures)
    repo_status = (PROJECT_ROOT / "docs" / "REPO_STATUS.md").read_text(encoding="utf-8")
    check(
        "Source/eval quality baseline   9b1cbc5 test: expand FluxMind community quality eval" in repo_status
        and "Current implementation commit  c130778 feat: add local product quota guard" in repo_status
        and "Current docs/health sync       efe2143 docs: document product quota guard" in repo_status
        and "Last deployed source/eval baseline 9b1cbc5 test: expand FluxMind community quality eval" in repo_status
        and "Live verification follow-up    30-paper corpus and 107/107 live retrieval refreshed on 2026-06-17 00:37 CST"
        in repo_status
        and "Latest deploy follow-up        c130778/efe2143 synced with restart and live-checked on 2026-06-17 00:39 CST"
        in repo_status,
        "repo status records current source and deployed baselines",
        failures,
    )
    check(
        "HEAD          a51a060" not in repo_status and "origin/main   a51a060" not in repo_status,
        "repo status does not contain stale pre-follow-up git table",
        failures,
    )
    roadmap = (PROJECT_ROOT / "docs" / "PLATFORM_AUDIT_AND_ROADMAP.md").read_text(encoding="utf-8")
    check(
        "Decide whether to push the current 36 local commits" not in roadmap,
        "roadmap does not contain stale pre-push near-term plan",
        failures,
    )
    market_research = (
        PROJECT_ROOT / "docs" / "PRODUCTION_GAP_AND_MARKET_RESEARCH.md"
    ).read_text(encoding="utf-8")
    check(
        "Production Gap Matrix" in market_research,
        "market research records production gap matrix",
        failures,
    )
    check(
        "Community Demand Signals" in market_research,
        "market research records community demand signals",
        failures,
    )
    feature_audit = (PROJECT_ROOT / "docs" / "FEATURE_AUDIT.md").read_text(encoding="utf-8")
    check("API Route Coverage" in feature_audit, "feature audit route coverage installed", failures)
    check("POST   /query/retrieve" in feature_audit, "feature audit records retrieval diagnostics route", failures)
    check("GET    /admin/runtime-manifest" in feature_audit, "feature audit records runtime manifest route", failures)
    check("POST   /admin/runtime-manifest/restore-check" in feature_audit, "feature audit records runtime restore-check route", failures)
    check("Product platform layer        incomplete" in feature_audit, "feature audit records platform gap", failures)
    platform_migration_source = (PROJECT_ROOT / "src" / "platform_migration.py").read_text(
        encoding="utf-8"
    )
    check(
        "collect_platform_migration_preflight" in platform_migration_source
        and "activation_ready" in platform_migration_source,
        "platform migration preflight collector installed",
        failures,
    )
    check(
        "format_platform_migration_preflight_markdown" in platform_migration_source
        and "content_exported" in platform_migration_source
        and "secrets_exported" in platform_migration_source,
        "platform migration preflight no-secret markdown installed",
        failures,
    )
    platform_migration_cli = (
        PROJECT_ROOT / "scripts" / "platform_migration_preflight.py"
    ).read_text(encoding="utf-8")
    check(
        "--require-activation" in platform_migration_cli
        and "preflight_ok" in platform_migration_cli,
        "platform migration preflight CLI installed",
        failures,
    )
    storage_migration_source = (PROJECT_ROOT / "src" / "storage_migration.py").read_text(
        encoding="utf-8"
    )
    check(
        "run_storage_migration_rehearsal" in storage_migration_source
        and "staged_restore_check" in storage_migration_source
        and "staged_storage_schema" in storage_migration_source,
        "storage migration rehearsal collector installed",
        failures,
    )
    check(
        "secrets_copied" in storage_migration_source
        and "content_exported_in_report" in storage_migration_source
        and "staging_root_inside_project" in storage_migration_source,
        "storage migration rehearsal no-secret guards installed",
        failures,
    )
    storage_migration_cli = (
        PROJECT_ROOT / "scripts" / "platform_migration_rehearsal.py"
    ).read_text(encoding="utf-8")
    check(
        "--staging-root" in storage_migration_cli
        and "--overwrite-staging" in storage_migration_cli
        and "TemporaryDirectory" in storage_migration_cli,
        "platform migration rehearsal CLI installed",
        failures,
    )
    product_readiness_source = (PROJECT_ROOT / "src" / "product_readiness.py").read_text(
        encoding="utf-8"
    )
    check(
        "collect_product_readiness" in product_readiness_source
        and "multi_user_identity_not_configured" in product_readiness_source
        and "billing_provider_not_configured" in product_readiness_source,
        "product readiness collector installed",
        failures,
    )
    check(
        "format_product_readiness_markdown" in product_readiness_source
        and "secrets_exported" in product_readiness_source
        and "content_exported" in product_readiness_source,
        "product readiness no-secret markdown installed",
        failures,
    )
    product_readiness_cli = (PROJECT_ROOT / "scripts" / "product_readiness.py").read_text(
        encoding="utf-8"
    )
    check(
        "--require-activation" in product_readiness_cli
        and "local_foundation_ready" in product_readiness_cli,
        "product readiness CLI installed",
        failures,
    )
    api_keys_source = (PROJECT_ROOT / "src" / "api_keys.py").read_text(encoding="utf-8")
    check(
        "LocalApiKeyRegistry" in api_keys_source
        and "token_hash" in api_keys_source
        and "revoke_key" in api_keys_source,
        "local API key registry installed",
        failures,
    )
    check(
        "secrets_exported" in api_keys_source
        and "api_key_registry_backend_status" in api_keys_source,
        "local API key registry no-secret status installed",
        failures,
    )
    api_key_registry_cli = (PROJECT_ROOT / "scripts" / "api_key_registry.py").read_text(
        encoding="utf-8"
    )
    check(
        "create" in api_key_registry_cli
        and "revoke" in api_key_registry_cli
        and "verify" in api_key_registry_cli
        and "list" in api_key_registry_cli,
        "local API key registry CLI installed",
        failures,
    )
    product_registry_source = (PROJECT_ROOT / "src" / "product_registry.py").read_text(
        encoding="utf-8"
    )
    check(
        "LocalProductRegistry" in product_registry_source
        and "product_users" in product_registry_source
        and "quota_limits" in product_registry_source
        and "billing_accounts" in product_registry_source,
        "local product registry installed",
        failures,
    )
    check(
        "secrets_exported" in product_registry_source
        and "product_registry_backend_status" in product_registry_source,
        "local product registry no-secret status installed",
        failures,
    )
    check(
        "workspace_for_user" in product_registry_source
        and "quota_decision" in product_registry_source,
        "local product quota decision installed",
        failures,
    )
    product_registry_cli = (PROJECT_ROOT / "scripts" / "product_registry.py").read_text(
        encoding="utf-8"
    )
    check(
        "bootstrap-local" in product_registry_cli
        and "record-usage" in product_registry_cli
        and "set-quota" in product_registry_cli,
        "local product registry CLI installed",
        failures,
    )
    provider_readiness_source = (PROJECT_ROOT / "src" / "provider_readiness.py").read_text(
        encoding="utf-8"
    )
    check(
        "collect_provider_readiness" in provider_readiness_source
        and "external_image_provider_not_configured" in provider_readiness_source
        and "matlab_backend_not_configured" in provider_readiness_source,
        "provider readiness collector installed",
        failures,
    )
    check(
        "format_provider_readiness_markdown" in provider_readiness_source
        and "secrets_exported" in provider_readiness_source
        and "content_exported" in provider_readiness_source,
        "provider readiness no-secret markdown installed",
        failures,
    )
    provider_readiness_cli = (PROJECT_ROOT / "scripts" / "provider_readiness.py").read_text(
        encoding="utf-8"
    )
    check(
        "--require-activation" in provider_readiness_cli
        and "local_foundation_ready" in provider_readiness_cli,
        "provider readiness CLI installed",
        failures,
    )
    quality_readiness_source = (PROJECT_ROOT / "src" / "quality_readiness.py").read_text(
        encoding="utf-8"
    )
    check(
        "collect_quality_readiness" in quality_readiness_source
        and "quality_maturity_targets" in quality_readiness_source
        and "live_report_unreadable" in quality_readiness_source,
        "quality readiness collector installed",
        failures,
    )
    check(
        "format_quality_readiness_markdown" in quality_readiness_source
        and "secrets_exported" in quality_readiness_source
        and "paths_exported" in quality_readiness_source,
        "quality readiness no-secret markdown installed",
        failures,
    )
    quality_readiness_cli = (PROJECT_ROOT / "scripts" / "quality_readiness.py").read_text(
        encoding="utf-8"
    )
    check(
        "--require-target" in quality_readiness_cli
        and "--live-report" in quality_readiness_cli
        and "local_foundation_ready" in quality_readiness_cli,
        "quality readiness CLI installed",
        failures,
    )

    app_source = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")
    check("st.write_stream" not in app_source, "chat stream avoids st.write_stream", failures)
    check("notranslate" in app_source and 'translate", "no"' in app_source, "translation guard installed", failures)
    check("get_async_job_manager" in app_source, "Streamlit async job panel installed", failures)
    check("job_search" in app_source and "job_status_filter" in app_source, "Streamlit job filters installed", failures)
    check("worker_leases" in app_source, "Streamlit worker lease status installed", failures)
    check("mock_image_template" in app_source, "Streamlit diagram template selector installed", failures)
    check("PYTHON_EXECUTION_TEMPLATES" in app_source and "python_execution_template" in app_source, "Streamlit Python execution templates installed", failures)
    check("OCTAVE_EXECUTION_TEMPLATES" in app_source and "octave_execution_template" in app_source, "Streamlit Octave execution templates installed", failures)
    check("render_admin_status" in app_source, "Streamlit admin status panel installed", failures)
    check("render_retention_preview" in app_source and "collect_retention_preview" in app_source, "Streamlit retention preview panel installed", failures)
    check("retention_delete" in app_source and "apply_retention_delete" in app_source, "Streamlit retention delete guard installed", failures)
    check("render_runtime_events" in app_source and "event_kind_filter" in app_source, "Streamlit runtime events panel installed", failures)
    check("status_provider_failures" in app_source, "Streamlit provider failure status panel installed", failures)
    check("status_query_usage" in app_source, "Streamlit query usage status panel installed", failures)
    check("status_retrieval_traces" in app_source and "retrieval_trace" in app_source, "Streamlit retrieval trace status panel installed", failures)
    check("status_cost_pricing" in app_source, "Streamlit query cost pricing panel installed", failures)
    check("status_code_execution" in app_source and "alert_thresholds" in app_source, "Streamlit code execution alert status installed", failures)
    check("status_api_access" in app_source and '"api_access"' in app_source, "Streamlit API access audit status installed", failures)
    check("rate_limited_recent" in app_source and '"rate_limit"' in app_source, "Streamlit API rate-limit status installed", failures)
    check("download_admin_metrics" in app_source and "format_admin_metrics" in app_source, "Streamlit metrics download installed", failures)
    check("status_execution_policy" in app_source and "code_execution_allowed_imports" in app_source, "Streamlit execution policy status installed", failures)
    check("status_storage" in app_source and "storage_readiness" in app_source, "Streamlit storage readiness panel installed", failures)
    check("distributed_job_store" in app_source, "Streamlit distributed job-store readiness panel installed", failures)
    check("status_storage_inventory" in app_source, "Streamlit storage inventory panel installed", failures)
    check("status_storage_schemas" in app_source and "storage_schemas" in app_source, "Streamlit storage schema panel installed", failures)
    check("status_platform_readiness" in app_source and "platform_readiness" in app_source, "Streamlit platform readiness panel installed", failures)
    check("status_product_readiness" in app_source and "product_readiness" in app_source, "Streamlit product readiness panel installed", failures)
    check("status_provider_readiness" in app_source and "provider_readiness" in app_source, "Streamlit provider readiness panel installed", failures)
    check("status_runtime_manifest" in app_source and "download_runtime_manifest" in app_source, "Streamlit runtime manifest panel installed", failures)
    check("runtime_restore_manifest_upload" in app_source and "format_runtime_restore_check_markdown" in app_source, "Streamlit runtime restore-check panel installed", failures)
    check("artifact_id" in app_source and "artifact_metadata" in app_source and "artifact_search" in app_source, "Streamlit artifact reference metadata installed", failures)
    check("octave_job" in app_source and "enqueue_local_octave" in app_source, "Streamlit Octave job panel installed", failures)
    check(
        "CorpusProfileStore" in app_source
        and "activate_profile_rebuild" in app_source
        and "collect_corpus_profile_status" in app_source
        and "format_corpus_profile_status_report" in app_source
        and "corpus_profile_report_download" in app_source,
        "Streamlit corpus profile panel installed",
        failures,
    )
    api_source = (PROJECT_ROOT / "api.py").read_text(encoding="utf-8")
    check("verify_configured_api_key_token" in api_source and "api_key_registry_configured" in api_source, "API auth supports local key registry", failures)
    check("enforce_product_quota" in api_source and "product_quota_guard" in api_source, "API query paths support local product quota guard", failures)
    check("/artifacts" in api_source, "artifact export route installed", failures)
    check("job_kind: str | None" in api_source and "kind: str | None" in api_source, "artifact metadata filters installed", failures)
    check("/admin/status" in api_source, "admin status route installed", failures)
    check("/admin/runtime-manifest" in api_source and "collect_runtime_backup_manifest" in api_source, "admin runtime manifest route installed", failures)
    storage_manifest_source = (PROJECT_ROOT / "src" / "storage_manifest.py").read_text(encoding="utf-8")
    check(
        "API_KEY_REGISTRY_FILE" in storage_manifest_source
        and "api_key_registry_sqlite" in storage_manifest_source,
        "runtime manifest includes API key registry state",
        failures,
    )
    check(
        "PRODUCT_REGISTRY_FILE" in storage_manifest_source
        and "product_registry_sqlite" in storage_manifest_source,
        "runtime manifest includes product registry state",
        failures,
    )
    check("/admin/runtime-manifest/restore-check" in api_source and "collect_runtime_restore_check" in api_source, "admin runtime restore-check route installed", failures)
    check("/admin/retention" in api_source and "collect_retention_preview" in api_source, "admin retention preview route installed", failures)
    check("/admin/retention/delete" in api_source and "apply_retention_delete" in api_source, "admin retention delete route installed", failures)
    check("/admin/events" in api_source and "list_runtime_events" in api_source, "admin runtime events route installed", failures)
    check("/corpus/papers" in api_source, "corpus metadata route installed", failures)
    check("filter_paper_records" in api_source and "indexed_status" in api_source, "corpus paper metadata filters installed", failures)
    check("/corpus/chunks" in api_source and "page: int | None" in api_source and "q=q" in api_source, "corpus chunk metadata route installed", failures)
    check("/corpus/structure" in api_source and "extract_pdf_structure_markers" in api_source and "q: str | None" in api_source, "corpus PDF structure route installed", failures)
    check("/corpus/structure/report" in api_source and "format_corpus_structure_report" in api_source, "corpus PDF structure report route installed", failures)
    check("/corpus/status" in api_source and "collect_corpus_status" in api_source, "corpus lifecycle status route installed", failures)
    check("/corpus/active" in api_source, "active corpus selection route installed", failures)
    check(
        "/corpus/profiles" in api_source
        and "CorpusProfileStore" in api_source
        and "corpus_profile_status" in api_source,
        "corpus profile routes installed",
        failures,
    )
    check("/corpus/profiles/{profile_id}/report" in api_source and "format_corpus_profile_status_report" in api_source, "corpus profile report route installed", failures)
    check("/corpus/profiles/{profile_id}/rebuild" in api_source, "corpus profile rebuild route installed", failures)
    check("/query/inspect" in api_source and "query_with_metadata" in api_source, "query citation inspection route installed", failures)
    check("/query/retrieve" in api_source and "retrieve_with_metadata" in api_source, "query retrieval diagnostics route installed", failures)
    check("/query/report" in api_source and "format_query_report" in api_source, "query Markdown report route installed", failures)
    check("Paper-to-Code Handoff" in api_source and "extract_markdown_code_blocks" in api_source, "paper-to-code report handoff installed", failures)
    check("/admin/status/report" in api_source and "format_admin_status_report" in api_source, "admin status report route installed", failures)
    check("/jobs/index/rebuild" in api_source, "index rebuild job route installed", failures)
    check("/jobs/async/index/rebuild" in api_source, "async index rebuild job route installed", failures)
    check("status: str | None" in api_source and "kind: str | None" in api_source and "q=q" in api_source, "job metadata filters installed", failures)
    check("idempotency_key" in api_source and "existing_idempotent_job" in api_source, "job idempotency API installed", failures)
    check("owner_id: str | None" in api_source and "request_ownership" in api_source, "API ownership metadata fields installed", failures)
    check("/jobs/code/octave-local" in api_source and "/jobs/async/code/octave-local" in api_source, "Octave-compatible job routes installed", failures)
    check("/jobs/{job_id}/retry" in api_source, "job retry route installed", failures)
    check("/jobs/{job_id}/retry-scheduled" in api_source, "scheduled retry route installed", failures)
    check('"logs": record.logs' in api_source, "job transition logs exposed by API", failures)
    check("append_runtime_event" in api_source, "query provider failures are recorded", failures)
    check("api_access_audit_middleware" in api_source and "kind=\"api_access\"" in api_source, "API access audit events are recorded", failures)
    check("api_rate_limit_decision" in api_source and "API rate limit exceeded" in api_source, "API rate-limit guard installed", failures)
    check("record_query_usage" in api_source and "query_usage" in api_source, "query usage estimates are recorded", failures)
    check("record_retrieval_trace" in api_source and 'kind="retrieval_trace"' in api_source, "query retrieval trace events are recorded", failures)
    check("provider_usage" in api_source and "provider_total_tokens" in api_source, "provider query usage passthrough installed", failures)
    check("/admin/metrics" in api_source and "format_admin_metrics" in api_source, "admin metrics route installed", failures)
    check("warm_existing_vector_store" in api_source and "build_vector_store" not in api_source, "API startup avoids synchronous index rebuild", failures)
    check("start_background_vector_store_warmup" in api_source and "@app.get(\"/ready\")" in api_source, "API startup warmup readiness route installed", failures)
    check("faiss.loader" in api_source and "setLevel(logging.ERROR)" in api_source, "FAISS optional fallback log noise reduced", failures)
    storage_manifest_source = (PROJECT_ROOT / "src" / "storage_manifest.py").read_text(encoding="utf-8")
    check("collect_runtime_backup_manifest" in storage_manifest_source and "secrets_exported" in storage_manifest_source, "runtime backup manifest installed", failures)
    check("collect_runtime_restore_check" in storage_manifest_source and "content_restored" in storage_manifest_source, "runtime restore dry-run check installed", failures)
    check("manifest_errors" in storage_manifest_source and "hash_algorithm must be sha256" in storage_manifest_source, "runtime restore manifest validation installed", failures)
    runtime_manifest_cli = (PROJECT_ROOT / "scripts" / "runtime_manifest.py").read_text(encoding="utf-8")
    check("format_runtime_backup_manifest_markdown" in runtime_manifest_cli and "--format" in runtime_manifest_cli, "runtime manifest CLI installed", failures)
    check("--restore-check" in runtime_manifest_cli and "format_runtime_restore_check_markdown" in runtime_manifest_cli, "runtime restore-check CLI installed", failures)
    jobs_source = (PROJECT_ROOT / "src" / "jobs.py").read_text(encoding="utf-8")
    check("sqlite3" in jobs_source and "CREATE TABLE IF NOT EXISTS jobs" in jobs_source, "SQLite job state mirror installed", failures)
    check("logs:" in jobs_source and "append_job_log" in jobs_source, "job transition logs installed", failures)
    check("not_before" in jobs_source and "schedule_retry" in jobs_source, "scheduled retry/backoff installed", failures)
    check("recover_queued_jobs" in jobs_source and "queue_health" in jobs_source, "durable queued job recovery installed", failures)
    check("worker_lease_health" in jobs_source and "active_worker_ids" in jobs_source, "worker lease health summary installed", failures)
    check("execution_timeout" in jobs_source, "local execution timeout error code installed", failures)
    check("_code_execution_provider" in jobs_source and "CODE_EXECUTION_BACKEND" in jobs_source, "job execution backend selector installed", failures)
    check("deadline_at" in jobs_source and "job_deadline_exceeded" in jobs_source, "queued job deadline policy installed", failures)
    check("find_by_idempotency_key" in jobs_source and "job_idempotency" in jobs_source and "append_new" in jobs_source, "durable job idempotency lookup installed", failures)
    check("dead_lettered" in jobs_source and "max_attempts" in jobs_source and "retry_backoff_s" in jobs_source, "durable job retry/dead-letter policy installed", failures)
    check("normalize_ownership" in jobs_source and "owner_id TEXT" in jobs_source, "durable job ownership metadata installed", failures)
    check("kind=\"code_execution\"" in jobs_source and "_record_code_execution_event" in jobs_source, "code execution runtime events installed", failures)
    check("claim_next_due_job" in jobs_source and "lease_expires_at" in jobs_source, "durable worker lease foundation installed", failures)
    check("LocalDurableJobWorker" in jobs_source and "run_once" in jobs_source, "explicit durable worker loop installed", failures)
    check("_monitor_cancellation" in jobs_source and "cancel_poll_interval_s" in jobs_source, "durable worker cancellation polling installed", failures)
    check("IngestionCancelled" in jobs_source and "run_index_rebuild" in jobs_source, "index rebuild cancellation handling installed", failures)
    worker_source = (PROJECT_ROOT / "scripts" / "run_job_worker.py").read_text(encoding="utf-8")
    check("LocalDurableJobWorker" in worker_source and "--worker-id" in worker_source, "durable worker CLI installed", failures)
    check("--forever" in worker_source and "run_polling" in worker_source, "durable worker long-running CLI mode installed", failures)
    worker_unit = (PROJECT_ROOT / "deploy" / "systemd" / "fluxmind-worker.service").read_text(encoding="utf-8")
    check("scripts/run_job_worker.py --forever" in worker_unit and "NoNewPrivileges=true" in worker_unit, "durable worker systemd unit installed", failures)
    execution_template_source = (PROJECT_ROOT / "src" / "execution_templates.py").read_text(encoding="utf-8")
    check("smc_reaching_law" in execution_template_source and "pmsm_current_decay" in execution_template_source, "local execution templates installed", failures)
    ingestion_source = (PROJECT_ROOT / "src" / "ingestion.py").read_text(encoding="utf-8")
    check("IngestionCancelled" in ingestion_source and "_raise_if_cancelled" in ingestion_source, "cancellable ingestion checkpoints installed", failures)
    check("extract_pdf_bibliographic_metadata" in ingestion_source and "paper_metadata_entries" in ingestion_source, "uploaded PDF metadata extraction installed", failures)
    check("_candidate_authors_from_first_page" in ingestion_source and "_candidate_topic_tags_from_first_page" in ingestion_source, "first-page author/keyword extraction installed", failures)
    check("_find_existing_pdf_by_checksum" in ingestion_source and "_sha256_bytes" in ingestion_source, "uploaded PDF checksum dedup installed", failures)
    check("scan_uploaded_pdf" in ingestion_source and "active_content_markers" in ingestion_source, "uploaded PDF pre-write scan installed", failures)
    providers_source = (PROJECT_ROOT / "src" / "providers.py").read_text(encoding="utf-8")
    execution_policy_source = (PROJECT_ROOT / "src" / "execution_policy.py").read_text(encoding="utf-8")
    check("ExecutionPolicyResult" in execution_policy_source and "POLICY_VIOLATION_EXIT_CODE" in execution_policy_source, "execution policy module installed", failures)
    check("python_import_not_allowed" in execution_policy_source and "octave_shell_call" in execution_policy_source, "execution policy abuse guards installed", failures)
    check("LocalOctaveExecutionProvider" in providers_source and "gnu-octave-local" in providers_source, "local Octave provider installed", failures)
    check("sliding-mode-observer" in providers_source and "paper-figure-redraft" in providers_source, "local diagram templates installed", failures)
    check("docker_execution_status" in providers_source and "docker_permission_denied" in providers_source, "docker execution readiness status installed", failures)
    check("DockerExecutionProvider" in providers_source and "docker_container_bind_mount" in providers_source, "Docker execution provider installed", failures)
    check('"--network"' in providers_source and '"none"' in providers_source and "no-new-privileges" in providers_source, "Docker execution sandbox flags installed", failures)
    check("evaluate_request_policy" in providers_source and "execution_policy_failure_result" in providers_source, "execution policy preflight installed", failures)
    check("execution_limit_preexec" in providers_source and "cpu_limit_enforced" in providers_source, "local execution CPU/memory resource metadata installed", failures)
    check("provider_runtime" in providers_source and "python_version" in providers_source, "local execution environment metadata installed", failures)
    check("filesystem_isolation" in providers_source and "network_policy_enforced" in providers_source, "local execution policy metadata installed", failures)
    check("BoundedStreamReader" in providers_source and "stdout_truncated" in providers_source, "local execution output limits installed", failures)
    check(
        "CODE_EXECUTION_MAX_ARTIFACTS" in providers_source
        and "artifact_collection_truncated" in providers_source,
        "local execution artifact limits installed",
        failures,
    )
    check("_resolve_workdir_path" in providers_source and "_is_collectable_output" in providers_source, "local execution path containment installed", failures)
    check("MAX_EXECUTION_FILES" in providers_source and "MAX_EXECUTION_TOTAL_BYTES" in providers_source, "local execution input size limits installed", failures)
    check("checksum_sha256" in providers_source and "byte_count" in providers_source, "local artifact checksum metadata installed", failures)
    chain_source = (PROJECT_ROOT / "src" / "chain.py").read_text(encoding="utf-8")
    check("hybrid_retrieve" in chain_source and "keyword_search_documents" in chain_source, "hybrid retrieval installed", failures)
    check("rerank_documents" in chain_source and "bm25_relevance_scores" in chain_source, "local BM25-lite reranker installed", failures)
    check("learned_rerank_documents" in chain_source and "RERANKER_MODEL" in chain_source, "optional local learned reranker installed", failures)
    check("seen_sources" in chain_source and "source diversity" in chain_source, "source-diverse reranking installed", failures)
    check("citation_instruction" in chain_source and "Valid numbered source refs" in chain_source, "numbered citation prompt guard installed", failures)
    check("neutralize_invalid_numbered_citations" in chain_source, "invalid numbered citation neutralizer installed", failures)
    check("RetrievalDiagnostics" in chain_source and "retrieve_with_metadata" in chain_source, "no-LLM retrieval diagnostics installed", failures)
    check("provider_usage_from_response" in chain_source and "usage_metadata" in chain_source, "provider token usage extraction installed", failures)
    check("Generated Artifact References" in chain_source and "generated_artifact_context" in chain_source, "artifact references in RAG context installed", failures)
    artifacts_source = (PROJECT_ROOT / "src" / "artifacts.py").read_text(encoding="utf-8")
    check("format_artifact_references" in artifacts_source, "artifact reference formatter installed", failures)
    check("CREATE TABLE IF NOT EXISTS artifacts" in artifacts_source and "storage_status" in artifacts_source, "artifact SQLite metadata mirror installed", failures)
    check("ownership_from_record" in artifacts_source and "idx_artifacts_owner_id" in artifacts_source, "artifact ownership metadata installed", failures)
    check("integrity_status" in artifacts_source and "checksum_mismatch" in artifacts_source, "artifact integrity status installed", failures)
    metadata_source = (PROJECT_ROOT / "src" / "metadata.py").read_text(encoding="utf-8")
    check("CREATE TABLE IF NOT EXISTS papers" in metadata_source and "storage_status" in metadata_source, "corpus SQLite metadata mirror installed", failures)
    check("CorpusProfileStore" in metadata_source and "CORPUS_PROFILES_FILE" in metadata_source, "corpus profile store installed", failures)
    check("atomic_write_json" in metadata_source and "NamedTemporaryFile" in metadata_source, "atomic corpus JSON writes installed", failures)
    check("doi" in metadata_source and "arxiv_id" in metadata_source and "topic_tags" in metadata_source, "paper bibliographic enrichment fields installed", failures)
    check("CREATE TABLE IF NOT EXISTS chunks" in metadata_source and "ChunkMetadataStore" in metadata_source, "chunk SQLite metadata mirror installed", failures)
    check("def source_paths" in metadata_source, "chunk metadata source path listing installed", failures)
    check("page: int | None" in metadata_source and "preview LIKE" in metadata_source, "chunk metadata filters installed", failures)
    admin_source = (PROJECT_ROOT / "src" / "admin.py").read_text(encoding="utf-8")
    check("provider_failures" in admin_source and "list_runtime_events" in admin_source, "admin provider failure history installed", failures)
    check("api_access" in admin_source and "by_token_status" in admin_source, "admin API access audit summary installed", failures)
    check("rate_limited_recent" in admin_source and "api_rate_limit_enabled" in admin_source, "admin API rate-limit summary installed", failures)
    check("upload_scans" in admin_source and "upload_scan_events" in admin_source, "admin upload scan summary installed", failures)
    check("summarize_provider_failure_alerts" in admin_source and "provider_failure_rate_high" in admin_source, "admin provider failure alerts installed", failures)
    check("collect_retention_preview" in admin_source and "delete_enabled" in admin_source, "no-delete retention preview installed", failures)
    check("apply_retention_delete" in admin_source and "RETENTION_DELETE_ENABLED" in admin_source, "guarded retention delete installed", failures)
    check("query_usage" in admin_source and "estimated_total_tokens" in admin_source, "admin query usage estimates installed", failures)
    check("query_usage_duration_ms" in admin_source and '"duration_ms"' in admin_source, "admin query latency summary installed", failures)
    check("summarize_query_usage_alerts" in admin_source and "query_duration_average_high" in admin_source, "admin query latency alerts installed", failures)
    check("retrieval_traces" in admin_source and "retrieval_source_page_incomplete" in admin_source, "admin retrieval trace summary installed", failures)
    check("summarize_retrieval_trace_alerts" in admin_source and "retrieval_empty_rate_high" in admin_source, "admin retrieval trace alerts installed", failures)
    check("fluxmind_retrieval_traces_recent_total" in admin_source, "admin retrieval trace metrics installed", failures)
    check("by_owner_id" in admin_source and "job_owner_counts" in admin_source, "admin ownership summaries installed", failures)
    check("summarize_job_alerts" in admin_source and "job_failures_recent" in admin_source, "admin job health alerts installed", failures)
    check("worker_leases" in admin_source and "worker_lease_health" in admin_source, "admin worker lease status installed", failures)
    check("provider_total_tokens" in admin_source and "provider_usage_events" in admin_source, "admin provider token usage summary installed", failures)
    check("summarize_query_cost" in admin_source and "cost_source" in admin_source, "admin query cost estimates installed", failures)
    check("docker_execution" in admin_source and "code_execution_backend" in admin_source, "admin execution sandbox readiness installed", failures)
    check("code_execution_policy" in admin_source and "code_execution_allowed_imports" in admin_source, "admin execution policy status installed", failures)
    check("code_execution_max_stdout_bytes" in admin_source and "code_execution_max_stderr_bytes" in admin_source, "admin execution output limits installed", failures)
    check(
        "code_execution_max_artifacts" in admin_source
        and "artifact_collection_truncations" in admin_source,
        "admin execution artifact limits installed",
        failures,
    )
    check(
        "summarize_code_execution_alerts" in admin_source
        and "code_execution_failure_rate_high" in admin_source,
        "admin code execution alerts installed",
        failures,
    )
    check("code_execution_events" in admin_source and "policy_violations" in admin_source, "admin code execution event summary installed", failures)
    check("storage_readiness_status" in admin_source and "external_storage_configured" in admin_source, "admin durable storage readiness installed", failures)
    check("storage_inventory_status" in admin_source and "content_scanned" in admin_source, "admin storage inventory installed", failures)
    check("API_KEY_REGISTRY_FILE" in admin_source and "api_key_registry_sqlite" in admin_source, "admin storage inventory includes API key registry state", failures)
    check("PRODUCT_REGISTRY_FILE" in admin_source and "product_registry_sqlite" in admin_source, "admin storage inventory includes product registry state", failures)
    check("storage_schema_status" in admin_source and "storage_schemas" in admin_source, "admin storage schema inventory installed", failures)
    check("distributed_job_store_status" in admin_source and "external_job_store_configured" in admin_source, "admin distributed job-store readiness installed", failures)
    check("platform_readiness_status" in admin_source and "distributed_worker_acceptance" in admin_source, "admin platform readiness installed", failures)
    check("collect_product_readiness" in admin_source and "product_readiness" in admin_source, "admin product readiness installed", failures)
    check("fluxmind_product_local_foundation_ready" in admin_source and "fluxmind_product_activation_ready" in admin_source, "admin product readiness metrics installed", failures)
    check("fluxmind_product_quota_guard_enabled" in admin_source, "admin product quota guard metric installed", failures)
    check("collect_provider_readiness" in admin_source and "provider_readiness" in admin_source, "admin provider readiness installed", failures)
    check("fluxmind_provider_local_foundation_ready" in admin_source and "fluxmind_provider_activation_ready" in admin_source, "admin provider readiness metrics installed", failures)
    storage_schema_source = (PROJECT_ROOT / "src" / "storage_schema.py").read_text(encoding="utf-8")
    storage_schema_cli = (PROJECT_ROOT / "scripts" / "storage_schema.py").read_text(encoding="utf-8")
    check("STORAGE_SCHEMA_VERSION" in storage_schema_source and "missing_required_columns" in storage_schema_source, "storage schema drift checks installed", failures)
    check("API_KEY_COLUMNS" in storage_schema_source and "api_key_registry_sqlite" in storage_schema_source, "API key registry storage schema installed", failures)
    check("PRODUCT_USER_COLUMNS" in storage_schema_source and "product_registry_sqlite" in storage_schema_source, "product registry storage schema installed", failures)
    check("storage_schema_status_for_root" in storage_schema_cli and "format_storage_schema_markdown" in storage_schema_cli, "storage schema CLI installed", failures)
    check("reranker_model_configured" in admin_source and "reranker_model_available" in admin_source, "admin reranker config status installed", failures)
    check('"storage": metadata_store.storage_status()' in admin_source, "admin corpus storage status installed", failures)
    check("corpus_index_status" in admin_source, "admin corpus index freshness installed", failures)
    check("corpus_status_from_state" in admin_source and "index_jobs" in admin_source, "corpus lifecycle status installed", failures)
    check("format_admin_status_report" in admin_source, "admin status Markdown report installed", failures)
    check("format_admin_metrics" in admin_source and "fluxmind_jobs_total" in admin_source, "admin metrics text export installed", failures)

    manifest = json.loads((PROJECT_ROOT / "papers/library/manifest.json").read_text(encoding="utf-8"))
    check(len(manifest) >= 30, "seed paper manifest has at least 30 entries", failures)
    check(
        any(item.get("doi") or item.get("arxiv_id") for item in manifest.values()),
        "seed paper manifest has DOI/arXiv enrichment",
        failures,
    )
    eval_config = json.loads((PROJECT_ROOT / "eval" / "rag_baseline.json").read_text(encoding="utf-8"))
    retrieval_eval_cases = eval_config.get("cases", []) + eval_config.get("retrieval_only_cases", [])
    check(
        any(case.get("recorded_answer") for case in eval_config.get("cases", [])),
        "recorded answer eval fixtures installed",
        failures,
    )
    check(
        eval_config.get("quality_gates", {}).get("minimum_case_count", 0) >= 40
        and len(eval_config.get("quality_gates", {}).get("required_answer_modes", [])) >= 5,
        "RAG eval aggregate quality gates installed",
        failures,
    )
    maturity_targets = {
        str(target.get("id"))
        for target in eval_config.get("quality_maturity_targets", [])
        if isinstance(target, dict)
    }
    check(
        {"self_use", "small_group", "community"}.issubset(maturity_targets),
        "RAG eval quality maturity targets installed",
        failures,
    )
    check(
        eval_config.get("quality_gates", {}).get("minimum_topic_tag_count", 0) >= 80
        and "zero speed" in eval_config.get("quality_gates", {}).get("required_topic_tags", []),
        "RAG eval topic coverage gates installed",
        failures,
    )
    check(
        eval_config.get("quality_gates", {}).get("minimum_recorded_answer_count", 0) >= 40
        and sum(1 for case in eval_config.get("cases", []) if case.get("recorded_answer")) >= 40,
        "RAG eval 40-case recorded-answer gate installed",
        failures,
    )
    check(
        eval_config.get("quality_gates", {}).get("minimum_retrieval_only_case_count", 0) >= 60
        and len(eval_config.get("retrieval_only_cases", [])) >= 60,
        "RAG eval retrieval-only case gate installed",
        failures,
    )
    check(
        eval_config.get("quality_gates", {}).get("minimum_retrieval_eval_question_count", 0) >= 100
        and len(eval_config.get("cases", [])) + len(eval_config.get("retrieval_only_cases", [])) >= 100,
        "RAG eval 100-question retrieval gate installed",
        failures,
    )
    check(
        eval_config.get("quality_gates", {}).get("minimum_code_output_case_count", 0) >= 8
        and len(eval_config.get("code_output_cases", [])) >= 8,
        "RAG eval code-output case gate installed",
        failures,
    )
    check(
        eval_config.get("quality_gates", {}).get("minimum_code_output_pass_rate", 0) >= 1.0
        and "python" in eval_config.get("quality_gates", {}).get("required_code_output_languages", []),
        "RAG eval code-output language/pass gates installed",
        failures,
    )
    check(
        "smc_reaching_law" in eval_config.get("quality_gates", {}).get("required_code_output_template_ids", [])
        and any(case.get("template_id") == "smc_reaching_law" for case in eval_config.get("code_output_cases", [])),
        "RAG eval code-output template gate installed",
        failures,
    )
    check(
        "local_job" in eval_config.get("quality_gates", {}).get("required_code_output_execution_modes", [])
        and any(case.get("execution_mode") == "local_job" for case in eval_config.get("code_output_cases", [])),
        "RAG eval job-backed code-output gate installed",
        failures,
    )
    required_pdf_kinds = set(eval_config.get("quality_gates", {}).get("required_pdf_structure_kinds", []))
    check(
        eval_config.get("quality_gates", {}).get("minimum_pdf_structure_case_count", 0) >= 15
        and {"equation", "table", "figure", "algorithm"}.issubset(required_pdf_kinds)
        and len(eval_config.get("pdf_structure_cases", [])) >= 15,
        "RAG eval PDF structure gate installed",
        failures,
    )
    check(
        "paper_to_code" in eval_config.get("quality_gates", {}).get("required_eval_lanes", [])
        and all(case.get("topic_tags") and case.get("eval_lanes") for case in retrieval_eval_cases),
        "RAG eval lane metadata installed",
        failures,
    )
    check(
        "observer_estimation" in eval_config.get("quality_gates", {}).get("required_topic_groups", [])
        and "topic_groups" in eval_config.get("domain_ontology", {}),
        "RAG eval domain ontology gates installed",
        failures,
    )
    check(
        all(
            all(ref.get("source_path") and ref.get("snippet") for ref in case.get("expected_refs", []))
            for case in eval_config.get("cases", [])
        ),
        "source/page eval references installed",
        failures,
    )
    evaluation_source = (PROJECT_ROOT / "src" / "evaluation.py").read_text(encoding="utf-8")
    check("verify_source_reference" in evaluation_source, "source/page eval verification installed", failures)
    check("evaluate_live_config" in evaluation_source and "query_inspect_payload" in evaluation_source, "live RAG eval scoring installed", failures)
    check("evaluate_live_retrieval_config" in evaluation_source and "query_retrieve_payload" in evaluation_source, "live retrieval eval scoring installed", failures)
    check(
        "evaluate_code_output_case" in evaluation_source
        and "CodeOutputCaseResult" in evaluation_source
        and "PYTHON_EXECUTION_TEMPLATES" in evaluation_source,
        "code-output eval scoring installed",
        failures,
    )
    check("evaluate_pdf_structure_case" in evaluation_source and "PdfStructureCaseResult" in evaluation_source, "PDF structure eval scoring installed", failures)
    check("evaluate_regression_gates" in evaluation_source and "RegressionGateResult" in evaluation_source, "aggregate RAG regression gates installed", failures)
    check("build_evaluation_report" in evaluation_source and "schema_version" in evaluation_source, "RAG eval JSON report builder installed", failures)
    check(
        "evaluate_quality_maturity_targets" in evaluation_source
        and "quality_maturity" in evaluation_source,
        "RAG eval quality maturity report installed",
        failures,
    )
    evaluate_rag_source = (PROJECT_ROOT / "scripts" / "evaluate_rag.py").read_text(encoding="utf-8")
    check("--live-url" in evaluate_rag_source and "evaluate_live_config" in evaluate_rag_source, "live RAG eval CLI installed", failures)
    check("--retrieval-url" in evaluate_rag_source and "evaluate_live_retrieval_config" in evaluate_rag_source, "live retrieval eval CLI installed", failures)
    check("code-output case" in evaluate_rag_source, "code-output eval CLI installed", failures)
    check("PDF structure case" in evaluate_rag_source, "PDF structure eval CLI installed", failures)
    check("regression gate" in evaluate_rag_source and "evaluate_regression_gates" in evaluate_rag_source, "RAG aggregate regression gate CLI installed", failures)
    check("--json-report" in evaluate_rag_source and "build_evaluation_report" in evaluate_rag_source, "RAG eval JSON report CLI installed", failures)
    deploy_sync_source = (PROJECT_ROOT / "scripts" / "deploy_sync.py").read_text(encoding="utf-8")
    check("--dry-run" in deploy_sync_source and "--apply" in deploy_sync_source, "safe deploy sync dry-run/apply guard installed", failures)
    check("REQUIRED_RUNTIME_EXCLUDES" in deploy_sync_source and "models/" in deploy_sync_source and "venv/" in deploy_sync_source, "safe deploy sync runtime excludes installed", failures)
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
            "systemctl is-active cloudflared-fluxmind-smy.service fluxmind-ui.service fluxmind-api.service fluxmind-worker.service docker.service; "
            "ss -ltnp | egrep '18501|18502'; "
            "curl -sS --max-time 10 http://127.0.0.1:18502/health; "
            "test -f /opt/fluxmind/app.py; "
            "grep -q 'render_streaming_response' /opt/fluxmind/app.py; "
            "grep -q 'get_async_job_manager' /opt/fluxmind/app.py; "
            "grep -q 'job_search' /opt/fluxmind/app.py; "
            "grep -q 'worker_leases' /opt/fluxmind/app.py; "
            "grep -q 'mock_image_template' /opt/fluxmind/app.py; "
            "grep -q 'PYTHON_EXECUTION_TEMPLATES' /opt/fluxmind/app.py; "
            "grep -q 'OCTAVE_EXECUTION_TEMPLATES' /opt/fluxmind/app.py; "
            "grep -q 'smc_reaching_law' /opt/fluxmind/src/execution_templates.py; "
            "grep -q 'pmsm_current_decay' /opt/fluxmind/src/execution_templates.py; "
            "grep -q 'render_retention_preview' /opt/fluxmind/app.py; "
            "grep -q 'render_runtime_events' /opt/fluxmind/app.py; "
            "grep -q 'status_provider_failures' /opt/fluxmind/app.py; "
            "grep -q 'status_query_usage' /opt/fluxmind/app.py; "
            "grep -q 'status_cost_pricing' /opt/fluxmind/app.py; "
            "grep -q 'download_admin_metrics' /opt/fluxmind/app.py; "
            "grep -q 'status_execution_policy' /opt/fluxmind/app.py; "
            "grep -q 'status_storage' /opt/fluxmind/app.py; "
            "grep -q 'status_storage_inventory' /opt/fluxmind/app.py; "
            "grep -q 'storage_readiness' /opt/fluxmind/app.py; "
            "grep -q 'distributed_job_store' /opt/fluxmind/app.py; "
            "grep -q 'artifact_metadata' /opt/fluxmind/app.py; "
            "grep -q 'artifact_search' /opt/fluxmind/app.py; "
            "grep -q 'enqueue_local_octave' /opt/fluxmind/app.py; "
            "grep -q 'CorpusProfileStore' /opt/fluxmind/app.py; "
            "grep -q 'collect_corpus_profile_status' /opt/fluxmind/app.py; "
            "grep -q 'format_corpus_profile_status_report' /opt/fluxmind/app.py; "
            "grep -q 'corpus_profile_report_download' /opt/fluxmind/app.py; "
            "grep -q '/artifacts' /opt/fluxmind/api.py; "
            "grep -q 'job_kind: str | None' /opt/fluxmind/api.py; "
            "grep -q '/admin/status' /opt/fluxmind/api.py; "
            "grep -q '/admin/retention' /opt/fluxmind/api.py; "
            "grep -q '/admin/events' /opt/fluxmind/api.py; "
            "grep -q '/corpus/papers' /opt/fluxmind/api.py; "
            "grep -q 'filter_paper_records' /opt/fluxmind/api.py; "
            "grep -q '/corpus/chunks' /opt/fluxmind/api.py; "
            "grep -q 'page: int | None' /opt/fluxmind/api.py; "
            "grep -q '/corpus/status' /opt/fluxmind/api.py; "
            "grep -q '/corpus/active' /opt/fluxmind/api.py; "
            "grep -q '/corpus/profiles' /opt/fluxmind/api.py; "
            "grep -q 'corpus_profile_status' /opt/fluxmind/api.py; "
            "grep -q '/corpus/profiles/{profile_id}/report' /opt/fluxmind/api.py; "
            "grep -q 'format_corpus_profile_status_report' /opt/fluxmind/api.py; "
            "grep -q '/corpus/profiles/{profile_id}/rebuild' /opt/fluxmind/api.py; "
            "grep -q 'activate_profile_rebuild' /opt/fluxmind/app.py; "
            "grep -q '/query/inspect' /opt/fluxmind/api.py; "
            "grep -q '/query/retrieve' /opt/fluxmind/api.py; "
            "grep -q '/query/report' /opt/fluxmind/api.py; "
            "grep -q 'query_with_metadata' /opt/fluxmind/api.py; "
            "grep -q 'retrieve_with_metadata' /opt/fluxmind/api.py; "
            "grep -q '/jobs/async/index/rebuild' /opt/fluxmind/api.py; "
            "grep -q 'q=q' /opt/fluxmind/api.py; "
            "grep -q '/jobs/code/octave-local' /opt/fluxmind/api.py; "
            "grep -q '/jobs/async/code/octave-local' /opt/fluxmind/api.py; "
            "grep -q '/jobs/{job_id}/retry-scheduled' /opt/fluxmind/api.py; "
            "grep -q '\"logs\": record.logs' /opt/fluxmind/api.py; "
            "grep -q '/admin/status/report' /opt/fluxmind/api.py; "
            "grep -q '/admin/metrics' /opt/fluxmind/api.py; "
            "grep -q 'append_runtime_event' /opt/fluxmind/api.py; "
            "grep -q 'record_query_usage' /opt/fluxmind/api.py; "
            "grep -q 'provider_total_tokens' /opt/fluxmind/api.py; "
            "grep -q 'warm_existing_vector_store' /opt/fluxmind/api.py; "
            "grep -q 'start_background_vector_store_warmup' /opt/fluxmind/api.py; "
            "grep -q '@app.get(\"/ready\")' /opt/fluxmind/api.py; "
            "grep -q 'faiss.loader' /opt/fluxmind/api.py; "
            "! grep -q 'build_vector_store' /opt/fluxmind/api.py; "
            "test -f /opt/fluxmind/src/capabilities.py; "
            "test -f /opt/fluxmind/src/execution_policy.py; "
            "test -f /opt/fluxmind/src/admin.py; "
            "test -f /opt/fluxmind/src/runtime.py; "
            "test -f /opt/fluxmind/src/api_keys.py; "
            "test -f /opt/fluxmind/src/product_registry.py; "
            "grep -q 'LocalApiKeyRegistry' /opt/fluxmind/src/api_keys.py; "
            "grep -q 'token_hash' /opt/fluxmind/src/api_keys.py; "
            "grep -q 'api_key_registry_backend_status' /opt/fluxmind/src/api_keys.py; "
            "grep -q 'LocalProductRegistry' /opt/fluxmind/src/product_registry.py; "
            "grep -q 'product_registry_backend_status' /opt/fluxmind/src/product_registry.py; "
            "grep -q 'quota_decision' /opt/fluxmind/src/product_registry.py; "
            "grep -q 'verify_configured_api_key_token' /opt/fluxmind/api.py; "
            "grep -q 'api_key_registry_configured' /opt/fluxmind/api.py; "
            "grep -q 'enforce_product_quota' /opt/fluxmind/api.py; "
            "grep -q 'api_key_registry_sqlite' /opt/fluxmind/src/admin.py; "
            "grep -q 'product_registry_sqlite' /opt/fluxmind/src/admin.py; "
            "grep -q 'fluxmind_product_quota_guard_enabled' /opt/fluxmind/src/admin.py; "
            "grep -q 'api_key_registry_sqlite' /opt/fluxmind/src/storage_schema.py; "
            "grep -q 'product_registry_sqlite' /opt/fluxmind/src/storage_schema.py; "
            "grep -q 'api_key_registry_sqlite' /opt/fluxmind/src/storage_manifest.py; "
            "grep -q 'product_registry_sqlite' /opt/fluxmind/src/storage_manifest.py; "
            "grep -q 'create' /opt/fluxmind/scripts/api_key_registry.py; "
            "grep -q 'revoke' /opt/fluxmind/scripts/api_key_registry.py; "
            "grep -q 'bootstrap-local' /opt/fluxmind/scripts/product_registry.py; "
            "grep -q 'CREATE TABLE IF NOT EXISTS papers' /opt/fluxmind/src/metadata.py; "
            "grep -q 'CorpusProfileStore' /opt/fluxmind/src/metadata.py; "
            "grep -q 'atomic_write_json' /opt/fluxmind/src/metadata.py; "
            "grep -q 'NamedTemporaryFile' /opt/fluxmind/src/metadata.py; "
            "grep -q 'arxiv_id' /opt/fluxmind/src/metadata.py; "
            "grep -q 'topic_tags' /opt/fluxmind/src/metadata.py; "
            "grep -q 'CREATE TABLE IF NOT EXISTS chunks' /opt/fluxmind/src/metadata.py; "
            "grep -q 'def source_paths' /opt/fluxmind/src/metadata.py; "
            "grep -q 'preview LIKE' /opt/fluxmind/src/metadata.py; "
            "grep -q 'CREATE TABLE IF NOT EXISTS jobs' /opt/fluxmind/src/jobs.py; "
            "grep -q 'append_job_log' /opt/fluxmind/src/jobs.py; "
            "grep -q 'schedule_retry' /opt/fluxmind/src/jobs.py; "
            "grep -q 'recover_queued_jobs' /opt/fluxmind/src/jobs.py; "
            "grep -q 'worker_lease_health' /opt/fluxmind/src/jobs.py; "
            "grep -q 'execution_timeout' /opt/fluxmind/src/jobs.py; "
            "grep -q 'job_deadline_exceeded' /opt/fluxmind/src/jobs.py; "
            "grep -q 'claim_next_due_job' /opt/fluxmind/src/jobs.py; "
            "grep -q 'lease_expires_at' /opt/fluxmind/src/jobs.py; "
            "grep -q 'LocalDurableJobWorker' /opt/fluxmind/src/jobs.py; "
            "grep -q '_monitor_cancellation' /opt/fluxmind/src/jobs.py; "
            "grep -q '_code_execution_provider' /opt/fluxmind/src/jobs.py; "
            "grep -q 'CODE_EXECUTION_BACKEND' /opt/fluxmind/src/jobs.py; "
            "grep -q '_record_code_execution_event' /opt/fluxmind/src/jobs.py; "
            "grep -q 'kind=\"code_execution\"' /opt/fluxmind/src/jobs.py; "
            "grep -q 'LocalDurableJobWorker' /opt/fluxmind/scripts/run_job_worker.py; "
            "grep -q -- '--forever' /opt/fluxmind/scripts/run_job_worker.py; "
            "grep -q 'scripts/run_job_worker.py --forever' /etc/systemd/system/fluxmind-worker.service; "
            "grep -q 'IngestionCancelled' /opt/fluxmind/src/jobs.py; "
            "grep -q '_raise_if_cancelled' /opt/fluxmind/src/ingestion.py; "
            "grep -q 'extract_pdf_bibliographic_metadata' /opt/fluxmind/src/ingestion.py; "
            "grep -q '_candidate_authors_from_first_page' /opt/fluxmind/src/ingestion.py; "
            "grep -q '_candidate_topic_tags_from_first_page' /opt/fluxmind/src/ingestion.py; "
            "grep -q '_find_existing_pdf_by_checksum' /opt/fluxmind/src/ingestion.py; "
            "grep -q 'queue_health' /opt/fluxmind/src/admin.py; "
            "grep -q 'worker_leases' /opt/fluxmind/src/admin.py; "
            "grep -q 'LocalOctaveExecutionProvider' /opt/fluxmind/src/providers.py; "
            "grep -q 'evaluate_request_policy' /opt/fluxmind/src/providers.py; "
            "grep -q 'execution_policy_failure_result' /opt/fluxmind/src/providers.py; "
            "grep -q 'ExecutionPolicyResult' /opt/fluxmind/src/execution_policy.py; "
            "grep -q 'python_import_not_allowed' /opt/fluxmind/src/execution_policy.py; "
            "grep -q 'octave_shell_call' /opt/fluxmind/src/execution_policy.py; "
            "grep -q 'sliding-mode-observer' /opt/fluxmind/src/providers.py; "
            "grep -q 'paper-figure-redraft' /opt/fluxmind/src/providers.py; "
            "grep -q 'docker_execution_status' /opt/fluxmind/src/providers.py; "
            "grep -q 'DockerExecutionProvider' /opt/fluxmind/src/providers.py; "
            "grep -q 'docker_container_bind_mount' /opt/fluxmind/src/providers.py; "
            "grep -q 'no-new-privileges' /opt/fluxmind/src/providers.py; "
            "grep -q 'execution_limit_preexec' /opt/fluxmind/src/providers.py; "
            "grep -q 'cpu_limit_enforced' /opt/fluxmind/src/providers.py; "
            "grep -q 'provider_runtime' /opt/fluxmind/src/providers.py; "
            "grep -q 'python_version' /opt/fluxmind/src/providers.py; "
            "grep -q 'filesystem_isolation' /opt/fluxmind/src/providers.py; "
            "grep -q 'network_policy_enforced' /opt/fluxmind/src/providers.py; "
            "grep -q 'BoundedStreamReader' /opt/fluxmind/src/providers.py; "
            "grep -q 'stdout_truncated' /opt/fluxmind/src/providers.py; "
            "grep -q '_resolve_workdir_path' /opt/fluxmind/src/providers.py; "
            "grep -q 'MAX_EXECUTION_TOTAL_BYTES' /opt/fluxmind/src/providers.py; "
            "grep -q 'checksum_sha256' /opt/fluxmind/src/providers.py; "
            "grep -q 'byte_count' /opt/fluxmind/src/providers.py; "
            "grep -q 'hybrid_retrieve' /opt/fluxmind/src/chain.py; "
            "grep -q 'bm25_relevance_scores' /opt/fluxmind/src/chain.py; "
            "grep -q 'learned_rerank_documents' /opt/fluxmind/src/chain.py; "
            "grep -q 'RERANKER_MODEL' /opt/fluxmind/src/chain.py; "
            "grep -q 'seen_sources' /opt/fluxmind/src/chain.py; "
            "grep -q 'citation_instruction' /opt/fluxmind/src/chain.py; "
            "grep -q 'neutralize_invalid_numbered_citations' /opt/fluxmind/src/chain.py; "
            "grep -q 'RetrievalDiagnostics' /opt/fluxmind/src/chain.py; "
            "grep -q 'retrieve_with_metadata' /opt/fluxmind/src/chain.py; "
            "grep -q 'provider_usage_from_response' /opt/fluxmind/src/chain.py; "
            "grep -q 'QueryResult' /opt/fluxmind/src/chain.py; "
            "grep -q 'missing_source_page_refs' /opt/fluxmind/src/chain.py; "
            "grep -q 'Generated Artifact References' /opt/fluxmind/src/chain.py; "
            "grep -q 'evaluate_live_config' /opt/fluxmind/src/evaluation.py; "
            "grep -q 'evaluate_live_retrieval_config' /opt/fluxmind/src/evaluation.py; "
            "grep -q 'evaluate_regression_gates' /opt/fluxmind/src/evaluation.py; "
            "grep -q 'build_evaluation_report' /opt/fluxmind/src/evaluation.py; "
            "grep -q -- '--live-url' /opt/fluxmind/scripts/evaluate_rag.py; "
            "grep -q -- '--retrieval-url' /opt/fluxmind/scripts/evaluate_rag.py; "
            "grep -q 'regression gate' /opt/fluxmind/scripts/evaluate_rag.py; "
            "grep -q -- '--json-report' /opt/fluxmind/scripts/evaluate_rag.py; "
            "grep -q 'collect_quality_readiness' /opt/fluxmind/src/quality_readiness.py; "
            "grep -q 'format_quality_readiness_markdown' /opt/fluxmind/src/quality_readiness.py; "
            "grep -q -- '--require-target' /opt/fluxmind/scripts/quality_readiness.py; "
            "grep -q -- '--live-report' /opt/fluxmind/scripts/quality_readiness.py; "
            "grep -q 'REQUIRED_RUNTIME_EXCLUDES' /opt/fluxmind/scripts/deploy_sync.py; "
            "grep -q -- '--dry-run' /opt/fluxmind/scripts/deploy_sync.py; "
            "grep -q -- '--apply' /opt/fluxmind/scripts/deploy_sync.py; "
            "grep -q 'models/' /opt/fluxmind/scripts/deploy_sync.py; "
            "grep -q 'venv/' /opt/fluxmind/scripts/deploy_sync.py; "
            "grep -q 'format_artifact_references' /opt/fluxmind/src/artifacts.py; "
            "grep -q 'CREATE TABLE IF NOT EXISTS artifacts' /opt/fluxmind/src/artifacts.py; "
            "grep -q 'integrity_status' /opt/fluxmind/src/artifacts.py; "
            "grep -q 'provider_failures' /opt/fluxmind/src/admin.py; "
            "grep -q 'collect_retention_preview' /opt/fluxmind/src/admin.py; "
            "grep -q 'query_usage' /opt/fluxmind/src/admin.py; "
            "grep -q 'provider_total_tokens' /opt/fluxmind/src/admin.py; "
            "grep -q 'summarize_query_cost' /opt/fluxmind/src/admin.py; "
            "grep -q 'docker_execution' /opt/fluxmind/src/admin.py; "
            "grep -q 'code_execution_policy' /opt/fluxmind/src/admin.py; "
            "grep -q 'code_execution_allowed_imports' /opt/fluxmind/src/admin.py; "
            "grep -q 'code_execution_max_stdout_bytes' /opt/fluxmind/src/admin.py; "
            "grep -q 'code_execution_max_stderr_bytes' /opt/fluxmind/src/admin.py; "
            "grep -q 'code_execution_events' /opt/fluxmind/src/admin.py; "
            "grep -q 'policy_violations' /opt/fluxmind/src/admin.py; "
            "grep -q 'storage_readiness_status' /opt/fluxmind/src/admin.py; "
            "grep -q 'distributed_job_store_status' /opt/fluxmind/src/admin.py; "
            "grep -q 'storage_inventory_status' /opt/fluxmind/src/admin.py; "
            "grep -q 'content_scanned' /opt/fluxmind/src/admin.py; "
            "grep -q 'external_storage_configured' /opt/fluxmind/src/admin.py; "
            "grep -q 'reranker_model_configured' /opt/fluxmind/src/admin.py; "
            "grep -q 'reranker_model_available' /opt/fluxmind/src/admin.py; "
            "grep -q 'corpus_index_status' /opt/fluxmind/src/admin.py; "
            "grep -q 'corpus_status_from_state' /opt/fluxmind/src/admin.py; "
            "grep -q 'format_admin_status_report' /opt/fluxmind/src/admin.py; "
            "grep -q 'format_admin_metrics' /opt/fluxmind/src/admin.py; "
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
            "chunks = root / 'metadata' / 'chunks.sqlite3'\n"
            "rows = 0\n"
            "chunk_sources = []\n"
            "if chunks.exists():\n"
            "    import sqlite3\n"
            "    with sqlite3.connect(chunks) as conn:\n"
            "        rows = conn.execute('SELECT COUNT(*) FROM chunks').fetchone()[0]\n"
            "        chunk_sources = [row[0] for row in conn.execute('SELECT DISTINCT source_path FROM chunks ORDER BY source_path')]\n"
            "print(f'chunk_metadata_rows={rows}')\n"
            "print(f'chunk_metadata_sources={len(chunk_sources)}')\n"
            "if papers and rows <= 0:\n"
            "    raise SystemExit('chunk metadata rows missing for active corpus')\n"
            "missing = sorted(set(papers) - set(chunk_sources))\n"
            "extra = sorted(set(chunk_sources) - set(papers))\n"
            "print(f'index_fresh={not missing and not extra}')\n"
            "if missing or extra:\n"
            "    raise SystemExit(f'chunk metadata source mismatch missing={missing} extra={extra}')\n"
            "from urllib import parse, request\n"
            "token = ''\n"
            "env = root / '.env'\n"
            "if env.exists():\n"
            "    for line in env.read_text(encoding='utf-8').splitlines():\n"
            "        if line.startswith('FLUXMIND_API_TOKEN='):\n"
            "            token = line.split('=', 1)[1].strip()\n"
            "            break\n"
            "headers = {'X-API-Key': token} if token else {}\n"
            "def api_get(path):\n"
            "    req = request.Request('http://127.0.0.1:18502' + path, headers=headers)\n"
            "    with request.urlopen(req, timeout=10) as resp:\n"
            "        return json.loads(resp.read().decode('utf-8'))\n"
            "def api_get_text(path):\n"
            "    req = request.Request('http://127.0.0.1:18502' + path, headers=headers)\n"
            "    with request.urlopen(req, timeout=10) as resp:\n"
            "        return resp.read().decode('utf-8')\n"
            "def api_post(path, payload):\n"
            "    post_headers = dict(headers)\n"
            "    post_headers['Content-Type'] = 'application/json'\n"
            "    data = json.dumps(payload).encode('utf-8')\n"
            "    req = request.Request('http://127.0.0.1:18502' + path, data=data, headers=post_headers, method='POST')\n"
            "    with request.urlopen(req, timeout=20) as resp:\n"
            "        return json.loads(resp.read().decode('utf-8'))\n"
            "admin_status = api_get('/admin/status').get('status', {})\n"
            "metrics = api_get_text('/admin/metrics')\n"
            "if 'fluxmind_jobs_total' not in metrics or 'fluxmind_api_access_recent_total' not in metrics:\n"
            "    raise SystemExit('admin metrics smoke returned unexpected metrics text')\n"
            "metrics_forbidden_scan = metrics.lower().replace('api_key_registry_sqlite', '')\n"
            "if 'api_key' in metrics_forbidden_scan or 'owner_id' in metrics_forbidden_scan:\n"
            "    raise SystemExit('admin metrics smoke exposed forbidden metadata')\n"
            "print('admin_metrics_smoke=ok')\n"
            "docker_execution = admin_status.get('config', {}).get('docker_execution', {})\n"
            "if 'configured' not in docker_execution or 'available' not in docker_execution:\n"
            "    raise SystemExit('admin status missing docker execution readiness')\n"
            "print(f'docker_execution_readiness=configured={docker_execution.get(\"configured\")} available={docker_execution.get(\"available\")} reason={docker_execution.get(\"reason\")}')\n"
            "storage_readiness = admin_status.get('config', {}).get('storage_readiness', {})\n"
            "metadata_storage = storage_readiness.get('metadata', {})\n"
            "object_storage = storage_readiness.get('object_storage', {})\n"
            "if 'available' not in metadata_storage or 'available' not in object_storage:\n"
            "    raise SystemExit('admin status missing storage readiness')\n"
            "print(f'storage_readiness=metadata_backend={metadata_storage.get(\"backend\")} metadata_available={metadata_storage.get(\"available\")} object_backend={object_storage.get(\"backend\")} object_available={object_storage.get(\"available\")} external_configured={storage_readiness.get(\"external_storage_configured\")}')\n"
            "distributed_job_store = admin_status.get('config', {}).get('distributed_job_store', {})\n"
            "if 'available' not in distributed_job_store or 'external_job_store_configured' not in distributed_job_store:\n"
            "    raise SystemExit('admin status missing distributed job-store readiness')\n"
            "print(f'distributed_job_store=backend={distributed_job_store.get(\"backend\")} available={distributed_job_store.get(\"available\")} external_configured={distributed_job_store.get(\"external_job_store_configured\")}')\n"
            "sample_chunks = api_get('/corpus/chunks?limit=1').get('chunks', [])\n"
            "if sample_chunks:\n"
            "    sample = sample_chunks[0]\n"
            "    q = (sample.get('preview') or '').split()[0]\n"
            "    params = {'source_path': sample['source_path'], 'limit': 10}\n"
            "    if sample.get('page') is not None:\n"
            "        params['page'] = sample['page']\n"
            "    if q:\n"
            "        params['q'] = q\n"
            "    filtered = api_get('/corpus/chunks?' + parse.urlencode(params)).get('chunks', [])\n"
            "    missing = api_get('/corpus/chunks?' + parse.urlencode({'source_path': sample['source_path'], 'q': 'definitely-no-such-chunk-token'})).get('chunks', [])\n"
            "    if not any(chunk.get('chunk_id') == sample.get('chunk_id') for chunk in filtered):\n"
            "        raise SystemExit('chunk filter smoke did not return the sampled chunk')\n"
            "    if missing:\n"
            "        raise SystemExit('chunk filter smoke returned rows for an impossible query')\n"
            "    print(f'chunk_filter_smoke={sample.get(\"chunk_id\")} filtered_count={len(filtered)} missing_filter_count={len(missing)}')\n"
            "else:\n"
            "    print('chunk_filter_smoke=skipped no chunk samples')\n"
            "retrieval = api_post('/query/retrieve', {'question': 'sliding mode control observer', 'answer_mode': 'literature_review'}).get('retrieval', {})\n"
            "if int(retrieval.get('context_count') or 0) <= 0:\n"
            "    raise SystemExit('retrieval diagnostics returned no context refs')\n"
            "if retrieval.get('missing_source_page_refs'):\n"
            "    raise SystemExit(f'retrieval diagnostics missing source/page refs: {retrieval.get(\"missing_source_page_refs\")}')\n"
            "print(f'retrieval_smoke=context_count={retrieval.get(\"context_count\")} ok={retrieval.get(\"ok\")}')\n"
            "profiles = api_get('/corpus/profiles').get('profiles', [])\n"
            "if profiles:\n"
            "    profile_id = profiles[0]['profile_id']\n"
            "    report = api_get_text('/corpus/profiles/' + parse.quote(profile_id, safe='') + '/report')\n"
            "    if '# FluxMind Corpus Profile Status' not in report or 'Profile ID:' not in report:\n"
            "        raise SystemExit('corpus profile report smoke returned unexpected markdown')\n"
            "    print(f'corpus_profile_report_smoke={profile_id}')\n"
            "else:\n"
            "    print('corpus_profile_report_smoke=skipped no profiles')\n"
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
