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
        "src/share_links.py",
        "src/metadata.py",
        "src/activation_suite.py",
        "src/collaboration_readiness.py",
        "src/openapi_contract.py",
        "src/storage_manifest.py",
        "src/storage_schema.py",
        "src/platform_migration.py",
        "src/storage_migration.py",
        "src/product_readiness.py",
        "src/product_activation_rehearsal.py",
        "src/product_registry.py",
        "src/provider_guard.py",
        "src/provider_readiness.py",
        "src/provider_runtime_rehearsal.py",
        "src/quality_readiness.py",
        "src/evaluation.py",
        "src/execution_templates.py",
        "eval/rag_baseline.json",
        "scripts/_safe_cli.py",
        "scripts/evaluate_rag.py",
        "scripts/activation_suite.py",
        "scripts/collaboration_readiness.py",
        "scripts/openapi_contract.py",
        "scripts/runtime_manifest.py",
        "scripts/storage_schema.py",
        "scripts/platform_migration_preflight.py",
        "scripts/platform_migration_rehearsal.py",
        "scripts/product_readiness.py",
        "scripts/product_activation_rehearsal.py",
        "scripts/product_registry.py",
        "scripts/provider_readiness.py",
        "scripts/provider_runtime_rehearsal.py",
        "scripts/quality_readiness.py",
        "scripts/api_key_registry.py",
        "scripts/share_link_registry.py",
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
        and "Current implementation commit  125664d fix: sanitize Streamlit job failure messages" in repo_status
        and "Current docs/health sync       docs: record Streamlit job result audit status (this commit)" in repo_status
        and "by the fifty-three local commits below" in repo_status
        and "125664d fix: sanitize Streamlit job failure messages" in repo_status
        and "1f9e8d2 docs: refresh git and documentation drift status" in repo_status
        and "848eef4 docs: record job request id audit status" in repo_status
        and "73be318 fix: redact unsafe job request ids" in repo_status
        and "a1a047e docs: record job owner metadata audit status" in repo_status
        and "c9d1f38 fix: redact job owner metadata" in repo_status
        and "887f53f docs: record job idempotency key audit status" in repo_status
        and "d65a8de fix: redact job idempotency keys" in repo_status
        and "2458fd8 docs: record job detail projection audit status" in repo_status
        and "de19eda test: isolate code execution event assertion" in repo_status
        and "ddd17b1 fix: redact job detail code outputs" in repo_status
        and "51230f8 docs: record Streamlit validation error audit status" in repo_status
        and "673cd2f fix: sanitize Streamlit validation error output" in repo_status
        and "384665a docs: record API request validation audit status" in repo_status
        and "4bf9775 fix: sanitize API request validation errors" in repo_status
        and "9e7c4e2 docs: record index rebuild job projection audit status" in repo_status
        and "19de06f fix: redact index rebuild job source paths" in repo_status
        and "1115b02 docs: refresh FluxMind git and documentation drift status" in repo_status
        and "Streamlit job result failure-message follow-up on 2026-06-20 23:42 CST" in repo_status
        and "main...origin/main [ahead 52]" in repo_status
        and "pass, 632 tests" in repo_status
        and "safe_streamlit_status_message" in repo_status
        and "raw `job.error[\"message\"]`" in repo_status
        and "Git/documentation drift refresh on 2026-06-20 23:31 CST" in repo_status
        and "main...origin/main [ahead 50]" in repo_status
        and "after the job request-ID docs sync" in repo_status
        and "no docs/health anchor drift" in repo_status
        and "Job request ID projection follow-up on 2026-06-20 23:21 CST" in repo_status
        and "main...origin/main [ahead 49]" in repo_status
        and "pass, 631 tests" in repo_status
        and "Bearer secret-request-token" in repo_status
        and "unsafe request_id exported=false" in repo_status
        and "Job owner metadata projection follow-up on 2026-06-20 23:12 CST" in repo_status
        and "main...origin/main [ahead 47]" in repo_status
        and "pass, 630 tests" in repo_status
        and "sk-secret-owner-id-12345678" in repo_status
        and "raw owner_id/owner_label exported=false" in repo_status
        and "Job idempotency key projection follow-up on 2026-06-20 23:01 CST" in repo_status
        and "main...origin/main [ahead 45]" in repo_status
        and "pass, 629 tests" in repo_status
        and "sk-secret-idempotency-key-verify-12345678" in repo_status
        and "raw idempotency_key exported=false" in repo_status
        and "Job detail API projection follow-up on 2026-06-20 22:43 CST" in repo_status
        and "main...origin/main [ahead 43]" in repo_status
        and "pass, 628 tests" in repo_status
        and "sk-secret-job-output-verify" in repo_status
        and "secret-main.py, stdout, stderr" in repo_status
        and "Streamlit validation error-output follow-up on 2026-06-20 22:30 CST" in repo_status
        and "main...origin/main [ahead 40]" in repo_status
        and "pass, 627 tests" in repo_status
        and "safe_streamlit_error_text" in repo_status
        and "rg '.format(error=exc)' app.py                              no matches" in repo_status
        and "API request validation projection follow-up on 2026-06-20 22:21 CST" in repo_status
        and "main...origin/main [ahead 38]" in repo_status
        and "pass, 626 tests" in repo_status
        and "RequestValidationError" in repo_status
        and "Invalid request field." in repo_status
        and "input field exported=false" in repo_status
        and "Index rebuild job projection follow-up on 2026-06-20 22:10 CST" in repo_status
        and "main...origin/main [ahead 36]" in repo_status
        and "source_path_count" in repo_status
        and "leaked=false" in repo_status
        and "pass, 624 tests" in repo_status
        and "index rebuild job API projection" in repo_status
        and "b82c6c6 docs: record API validation error audit status" in repo_status
        and "Git/documentation drift refresh on 2026-06-20 22:01 CST" in repo_status
        and "main...origin/main [ahead 34]" in repo_status
        and "after the API validation error-output docs sync" in repo_status
        and "API validation error-output follow-up on 2026-06-20 21:50 CST" in repo_status
        and "pass, 622 tests, 2 known warnings" in repo_status
        and "API-validation-error-sanitizer anchor" in repo_status
        and "rg 'detail=str(exc)' api.py                                no matches" in repo_status
        and "invalid_corpus_source_path" in repo_status
        and "artifact_export_denied" in repo_status
        and "Git/documentation status refresh on 2026-06-20 21:38 CST" in repo_status
        and "main...origin/main [ahead 31]" in repo_status
        and "docs-only refresh" in repo_status
        and "routes=69, operations=76" in repo_status
        and "no docs/health anchor drift" in repo_status
        and "Streamlit admin UI error-output follow-up on 2026-06-20 20:44 CST" in repo_status
        and "pass, 619 tests, 2 known warnings" in repo_status
        and "Artifact download failures" in repo_status
        and "admin on-demand buttons" in repo_status
        and "no longer contains direct `st.error(str(exc))` or" in repo_status
        and "Product registry Streamlit error-output follow-up on 2026-06-20 20:32 CST" in repo_status
        and "pass, 618 tests, 2 known warnings" in repo_status
        and "sanitizes Streamlit product-registry management exception output" in repo_status
        and "workspace list/create, member, quota, and permission" in repo_status
        and "Share-link Streamlit error-output follow-up on 2026-06-20 20:20 CST" in repo_status
        and "pass, 617 tests, 2 known warnings" in repo_status
        and "sanitizes Streamlit share-link management exception output" in repo_status
        and "uses the sanitized error helper" in repo_status
        and "c7b6d9d docs: refresh git and drift status" in repo_status
        and "6066547 docs: record runtime event redaction audit" in repo_status
        and "Last deployed source/eval baseline 9b1cbc5 test: expand FluxMind community quality eval" in repo_status
        and "Live verification follow-up    30-paper corpus and 107/107 live retrieval refreshed on 2026-06-17 02:37 CST"
        in repo_status
        and "Latest deploy follow-up        95f1760/e4da2e9 synced without restart and live-checked on 2026-06-17 02:59 CST"
        in repo_status
        and "Octave-aware code-output fallback" in repo_status
        and "Git/documentation drift refresh on 2026-06-20 20:10 CST" in repo_status
        and "no OpenAPI no-secret snapshot drift" in repo_status
        and "share tokens/URLs exported=false" in repo_status
        and "API-key public metadata projection follow-up on 2026-06-20 16:16 CST" in repo_status
        and "create/list/verify/revoke now removes raw owner IDs" in repo_status
        and "Git/docs drift refresh on 2026-06-20 16:08 CST" in repo_status
        and "pass, 17 docs/feature-audit/" in repo_status
        and "share-link admin runtime-event workspace-present" in repo_status
        and "Runtime event metadata-value redaction follow-up on 2026-06-20 16:01 CST" in repo_status
        and "pass, 616 tests, 2 known warnings" in repo_status
        and "runtime event metadata-value redaction" in repo_status
        and "Execution input materialization follow-up on 2026-06-20 15:50 CST" in repo_status
        and "pass, 614 tests, 2 known warnings" in repo_status
        and "execution input materialization conflict handling" in repo_status
        and "regular-file entrypoint guards" in repo_status
        and "Product registry referential-integrity follow-up on 2026-06-20 15:39 CST" in repo_status
        and "pass, 610 tests, 2 known warnings" in repo_status
        and "unsafe member user IDs are not echoed by the CLI" in repo_status
        and "Artifact path-resolution follow-up on 2026-06-20 15:26 CST" in repo_status
        and "pass, 608 tests, 2 known warnings" in repo_status
        and "rejects nonlocal file artifact URIs" in repo_status
        and "Corpus same-name metadata follow-up on 2026-06-20 15:18 CST" in repo_status
        and "pass, 606 tests, 2 known warnings" in repo_status
        and "source-path-specific entries" in repo_status
        and "Job lease-release follow-up on 2026-06-20 15:05 CST" in repo_status
        and "pass, 605 tests, 2 known warnings" in repo_status
        and "fingerprint=15bdfa2ae5ec34f1d0045c38b7137cf2b31a27857b1571a035a8efc12d61d18c" in repo_status
        and "pass, ok=true, diff_count=0 against the" in repo_status
        and "no storage-schema drift" in repo_status
        and "Share-link event-evidence follow-up on 2026-06-20 15:00 CST" in repo_status
        and "live answer count/pass-rate/term-coverage gates" in repo_status
        and "live_retrieval_pass_rate=1.0" in repo_status,
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
        "Decide whether to push the current 36 local commits" not in roadmap
        and "pass, 632 tests, 2 known warnings" in roadmap
        and "Streamlit job result error audit" in roadmap
        and "Streamlit-job-result-error-sanitizer" in roadmap
        and "Git/documentation drift refresh" in roadmap
        and "Job request ID projection audit" in roadmap
        and "job-request-id-projection" in roadmap
        and "Job owner metadata projection audit" in roadmap
        and "job-owner-metadata-projection" in roadmap
        and "Job idempotency key projection audit" in roadmap
        and "job-idempotency-key-projection" in roadmap
        and "Job detail API projection audit" in roadmap
        and "job-detail-code-output-projection" in roadmap
        and "Streamlit validation error-output audit" in roadmap
        and "Streamlit-validation-error-sanitizer" in roadmap
        and "API request validation projection audit" in roadmap
        and "request-validation-error-projection" in roadmap
        and "Index rebuild job projection audit" in roadmap
        and "index-rebuild-job-projection" in roadmap
        and "execution-input-materialization" in roadmap
        and "runtime-event-metadata-value-redaction" in roadmap
        and "admin-on-demand-error-sanitizer" in roadmap
        and "artifact-gallery-error-sanitizer" in roadmap
        and "API-validation-error-sanitizer" in roadmap
        and "product-registry-error-sanitizer" in roadmap
        and "share-link-error-sanitizer" in roadmap,
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
    check(
        "collect_platform_migration_rehearsal" in storage_migration_source
        and "storage_migration_rehearsal_public_status" in storage_migration_source
        and "raw_manifests_included" in storage_migration_source
        and "paths_exported" in storage_migration_source,
        "storage migration rehearsal public projection installed",
        failures,
    )
    check(
        "collect_object_storage_migration_manifest" in storage_migration_source
        and "source_paths_exported" in storage_migration_source
        and "object_key_strategy" in storage_migration_source,
        "object storage migration manifest installed",
        failures,
    )
    check(
        "verify_object_storage_migration_manifest" in storage_migration_source
        and "object_storage_migration_manifest_verify" in storage_migration_source
        and "object_differences" in storage_migration_source,
        "object storage migration manifest verifier installed",
        failures,
    )
    check(
        "collect_job_store_migration_manifest" in storage_migration_source
        and "job_store_migration_manifest" in storage_migration_source
        and "payload_exported" in storage_migration_source
        and "owner_ids_exported" in storage_migration_source,
        "job-store migration manifest installed",
        failures,
    )
    check(
        "verify_job_store_migration_manifest" in storage_migration_source
        and "job_store_migration_manifest_verify" in storage_migration_source
        and "job_differences" in storage_migration_source,
        "job-store migration manifest verifier installed",
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
    check(
        "--include-object-manifest" in storage_migration_cli
        and "--object-key-prefix" in storage_migration_cli
        and "--verify-object-manifest" in storage_migration_cli,
        "platform migration rehearsal object manifest CLI installed",
        failures,
    )
    check(
        "--include-job-store-manifest" in storage_migration_cli
        and "--verify-job-store-manifest" in storage_migration_cli
        and "verify_job_store_migration_manifest" in storage_migration_cli,
        "platform migration rehearsal job-store manifest CLI installed",
        failures,
    )
    api_source = (PROJECT_ROOT / "api.py").read_text(encoding="utf-8")
    check(
        "collect_platform_migration_rehearsal" in api_source
        and '"/admin/platform-migration-rehearsal"' in api_source
        and '"/admin/platform-migration-rehearsal/report"' in api_source
        and "fluxmind-platform-migration-rehearsal.md" in api_source,
        "platform migration rehearsal admin API installed",
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
    product_rehearsal_source = (
        PROJECT_ROOT / "src" / "product_activation_rehearsal.py"
    ).read_text(encoding="utf-8")
    check(
        "collect_product_activation_rehearsal" in product_rehearsal_source
        and "quota_exceeded" in product_rehearsal_source
        and "product_role_forbidden" in product_rehearsal_source
        and "token_exported" in product_rehearsal_source,
        "product activation rehearsal collector installed",
        failures,
    )
    check(
        "workspace_isolation" in product_rehearsal_source
        and "viewer_cross_workspace_query_denied" in product_rehearsal_source
        and "owner_cross_workspace_admin_denied" in product_rehearsal_source
        and "share_links_enabled" in product_rehearsal_source
        and "private_corpora_enabled" in product_rehearsal_source,
        "product activation rehearsal exercises workspace isolation",
        failures,
    )
    check(
        "format_product_activation_rehearsal_markdown" in product_rehearsal_source
        and "secrets_exported" in product_rehearsal_source
        and "paths_exported" in product_rehearsal_source,
        "product activation rehearsal no-secret markdown installed",
        failures,
    )
    check(
        "collect_product_activation_rehearsal" in api_source
        and '"/admin/product-activation-rehearsal"' in api_source
        and '"/admin/product-activation-rehearsal/report"' in api_source
        and "fluxmind-product-activation-rehearsal.md" in api_source,
        "product activation rehearsal admin API installed",
        failures,
    )
    product_rehearsal_cli = (
        PROJECT_ROOT / "scripts" / "product_activation_rehearsal.py"
    ).read_text(encoding="utf-8")
    check(
        "--require-activation" in product_rehearsal_cli
        and "collect_product_activation_rehearsal" in product_rehearsal_cli,
        "product activation rehearsal CLI installed",
        failures,
    )
    collaboration_source = (
        PROJECT_ROOT / "src" / "collaboration_readiness.py"
    ).read_text(encoding="utf-8")
    check(
        "collect_collaboration_readiness" in collaboration_source
        and "private_corpora_disabled" in collaboration_source
        and "share_links_disabled" in collaboration_source
        and "share_link_registry_backend_status" in collaboration_source
        and "collaboration_guard_not_ready" in collaboration_source,
        "collaboration readiness collector installed",
        failures,
    )
    check(
        "format_collaboration_readiness_markdown" in collaboration_source
        and "share_tokens_exported" in collaboration_source
        and "share_urls_exported" in collaboration_source
        and "identifiers_exported" in collaboration_source,
        "collaboration readiness no-secret markdown installed",
        failures,
    )
    collaboration_cli = (
        PROJECT_ROOT / "scripts" / "collaboration_readiness.py"
    ).read_text(encoding="utf-8")
    check(
        "--require-activation" in collaboration_cli
        and "collect_collaboration_readiness" in collaboration_cli,
        "collaboration readiness CLI installed",
        failures,
    )
    check(
        "collect_collaboration_readiness" in api_source
        and '"/admin/collaboration-readiness"' in api_source
        and '"/admin/collaboration-readiness/report"' in api_source
        and "fluxmind-collaboration-readiness.md" in api_source,
        "collaboration readiness admin API installed",
        failures,
    )
    api_keys_source = (PROJECT_ROOT / "src" / "api_keys.py").read_text(encoding="utf-8")
    api_key_registry_cli = (PROJECT_ROOT / "scripts" / "api_key_registry.py").read_text(
        encoding="utf-8"
    )
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
    check(
        "owner_id_fingerprint" in api_keys_source
        and "description_fingerprint" in api_keys_source
        and "owner_exported" in api_keys_source
        and "description_exported" in api_keys_source
        and "owner_fingerprint=" in api_key_registry_cli,
        "local API key registry public metadata projection installed",
        failures,
    )
    check(
        "create" in api_key_registry_cli
        and "revoke" in api_key_registry_cli
        and "verify" in api_key_registry_cli
        and "list" in api_key_registry_cli,
        "local API key registry CLI installed",
        failures,
    )
    check(
        'args.command == "create" and args.format != "json"' in api_key_registry_cli
        and "requires --format json" in api_key_registry_cli,
        "local API key registry create requires JSON one-time token output",
        failures,
    )
    share_link_source = (PROJECT_ROOT / "src" / "share_links.py").read_text(
        encoding="utf-8"
    )
    check(
        "LocalShareLinkRegistry" in share_link_source
        and "token_hash" in share_link_source
        and "revoke_link" in share_link_source
        and "resolve_token" in share_link_source,
        "local share-link registry installed",
        failures,
    )
    check(
        "share_link_registry_backend_status" in share_link_source
        and "share_tokens_exported" in share_link_source
        and "share_urls_exported" in share_link_source
        and "resource_ref_fingerprint" in share_link_source,
        "local share-link registry no-secret status installed",
        failures,
    )
    share_link_registry_cli = (PROJECT_ROOT / "scripts" / "share_link_registry.py").read_text(
        encoding="utf-8"
    )
    check(
        "create" in share_link_registry_cli
        and "revoke" in share_link_registry_cli
        and "resolve" in share_link_registry_cli
        and "list" in share_link_registry_cli,
        "local share-link registry CLI installed",
        failures,
    )
    check(
        'args.command == "create" and args.format != "json"' in share_link_registry_cli
        and "one-time share token" in share_link_registry_cli,
        "local share-link registry create requires JSON one-time token output",
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
    check(
        "permission_decision" in product_registry_source
        and "PRODUCT_RBAC_ACTION_ROLES" in product_registry_source,
        "local product RBAC decision installed",
        failures,
    )
    check(
        "workspace_detail" in product_registry_source
        and "list_workspace_summaries" in product_registry_source,
        "local product registry admin summaries installed",
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
    check(
        "add-member" in product_registry_cli
        and "check-permission" in product_registry_cli,
        "local product RBAC CLI installed",
        failures,
    )
    provider_readiness_source = (PROJECT_ROOT / "src" / "provider_readiness.py").read_text(
        encoding="utf-8"
    )
    check(
        "collect_provider_readiness" in provider_readiness_source
        and "external_image_provider_not_configured" in provider_readiness_source
        and "matlab_backend_not_configured" in provider_readiness_source
        and "provider_quota_guard_invalid_limit" in provider_readiness_source,
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
    provider_guard_source = (PROJECT_ROOT / "src" / "provider_guard.py").read_text(
        encoding="utf-8"
    )
    check(
        "provider_quota_guard_decision" in provider_guard_source
        and "provider_prompt_token_limit_exceeded" in provider_guard_source
        and "provider_cost_limit_exceeded" in provider_guard_source,
        "provider quota/cost guard decision installed",
        failures,
    )
    chain_source = (PROJECT_ROOT / "src" / "chain.py").read_text(encoding="utf-8")
    runtime_source = (PROJECT_ROOT / "src" / "runtime.py").read_text(encoding="utf-8")
    check(
        "ProviderQuotaGuardError" in runtime_source
        and "ProviderQuotaGuardError" in chain_source,
        "provider quota/cost guard denial error installed",
        failures,
    )
    check(
        "sanitize_runtime_event_message" in runtime_source
        and "sk-[A-Za-z0-9_-]{8,}" in runtime_source
        and "SAFE_RUNTIME_EVENT_MESSAGE_REDACTION" in runtime_source,
        "runtime event messages redact bare secret-like tokens",
        failures,
    )
    check(
        "SENSITIVE_RUNTIME_EVENT_METADATA_VALUE_PATTERNS" in runtime_source
        and "SAFE_RUNTIME_EVENT_METADATA_VALUE_REDACTION" in runtime_source
        and "file://\\S+" in runtime_source,
        "runtime event metadata values redact secret-like strings",
        failures,
    )
    provider_rehearsal_source = (
        PROJECT_ROOT / "src" / "provider_runtime_rehearsal.py"
    ).read_text(encoding="utf-8")
    check(
        "collect_provider_runtime_rehearsal" in provider_rehearsal_source
        and "MockImageGenerationProvider" in provider_rehearsal_source
        and "LocalPythonExecutionProvider" in provider_rehearsal_source
        and "LocalOctaveExecutionProvider" in provider_rehearsal_source,
        "provider runtime rehearsal collector installed",
        failures,
    )
    check(
        "format_provider_runtime_rehearsal_markdown" in provider_rehearsal_source
        and "external_activation_ready" in provider_rehearsal_source
        and "paths_exported" in provider_rehearsal_source,
        "provider runtime rehearsal no-secret markdown installed",
        failures,
    )
    check(
        "provider_quota_guard_decision" in provider_rehearsal_source
        and "provider_prompt_token_limit_exceeded" in provider_rehearsal_source,
        "provider runtime rehearsal exercises quota guard",
        failures,
    )
    check(
        "_policy_violation_summary" in provider_rehearsal_source
        and "python_import_not_allowed" in provider_rehearsal_source
        and "python_call_not_allowed" in provider_rehearsal_source
        and "octave_shell_call" in provider_rehearsal_source
        and "Execution Abuse Policy" in provider_rehearsal_source,
        "provider runtime rehearsal exercises execution abuse policy",
        failures,
    )
    provider_rehearsal_cli = (
        PROJECT_ROOT / "scripts" / "provider_runtime_rehearsal.py"
    ).read_text(encoding="utf-8")
    check(
        "--require-local-foundation" in provider_rehearsal_cli
        and "collect_provider_runtime_rehearsal" in provider_rehearsal_cli,
        "provider runtime rehearsal CLI installed",
        failures,
    )
    check(
        "collect_provider_runtime_rehearsal" in api_source
        and '"/admin/provider-runtime-rehearsal"' in api_source
        and '"/admin/provider-runtime-rehearsal/report"' in api_source
        and "fluxmind-provider-runtime-rehearsal.md" in api_source,
        "provider runtime rehearsal admin API installed",
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
        "live_answer_pass_rate" in quality_readiness_source
        and "average_live_answer_term_coverage" in quality_readiness_source
        and "minimum_live_retrieval_pass_rate" in quality_readiness_source,
        "quality readiness live quality thresholds installed",
        failures,
    )
    check(
        "format_quality_readiness_markdown" in quality_readiness_source
        and "secrets_exported" in quality_readiness_source
        and "paths_exported" in quality_readiness_source
        and "Target Gap Summary" in quality_readiness_source,
        "quality readiness no-secret markdown installed",
        failures,
    )
    check(
        "_target_gap_summary" in quality_readiness_source
        and "count_gaps" in quality_readiness_source
        and "quality_gaps" in quality_readiness_source,
        "quality readiness target gap summary installed",
        failures,
    )
    check(
        "_evidence_requests" in quality_readiness_source
        and "next_evidence_request" in quality_readiness_source
        and "## Evidence Requests" in quality_readiness_source,
        "quality readiness evidence request summary installed",
        failures,
    )
    check(
        "_evidence_collection_plan" in quality_readiness_source
        and "community_evidence_plan" in quality_readiness_source
        and "next_evidence_plan" in quality_readiness_source
        and "## Evidence Collection Plan" in quality_readiness_source,
        "quality readiness evidence collection plan installed",
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
    check(
        "collect_quality_readiness" in api_source
        and '"/admin/quality-readiness"' in api_source
        and '"/admin/quality-readiness/report"' in api_source
        and "fluxmind-quality-readiness.md" in api_source,
        "quality readiness admin API installed",
        failures,
    )
    check(
        "admin_quality_readiness_with_report" in api_source
        and "QualityReadinessRequest" in api_source
        and "live_reports=live_reports" in api_source
        and "QualityReadinessResponse" in api_source,
        "quality readiness admin API live-report input installed",
        failures,
    )
    activation_suite_source = (PROJECT_ROOT / "src" / "activation_suite.py").read_text(
        encoding="utf-8"
    )
    check(
        "collect_activation_suite" in activation_suite_source
        and "collect_product_readiness" in activation_suite_source
        and "collect_product_activation_rehearsal" in activation_suite_source
        and "collect_collaboration_readiness" in activation_suite_source
        and "collect_provider_runtime_rehearsal" in activation_suite_source
        and "run_storage_migration_rehearsal" in activation_suite_source
        and "collect_quality_readiness" in activation_suite_source,
        "activation suite collector installed",
        failures,
    )
    check(
        "collect_openapi_contract" in activation_suite_source
        and "openapi_schema" in activation_suite_source
        and '"openapi_contract.ok"' in activation_suite_source,
        "activation suite OpenAPI contract gate installed",
        failures,
    )
    check(
        "format_activation_suite_markdown" in activation_suite_source
        and "raw_reports_included" in activation_suite_source
        and "paths_exported" in activation_suite_source
        and "full_activation_ready" in activation_suite_source,
        "activation suite no-secret markdown installed",
        failures,
    )
    check(
        "_next_quality_evidence" in activation_suite_source
        and "next_evidence_target" in activation_suite_source
        and "## Next Quality Evidence" in activation_suite_source,
        "activation suite next quality evidence summary installed",
        failures,
    )
    check(
        "_activation_action_plan" in activation_suite_source
        and "activation_action_plan" in activation_suite_source
        and 'area="product_readiness"' in activation_suite_source
        and 'area="collaboration_readiness"' in activation_suite_source
        and 'area="openapi_contract"' in activation_suite_source
        and "collaboration_activation_not_ready" in activation_suite_source
        and "product_readiness_activation_not_ready" in activation_suite_source
        and "## Activation Action Plan" in activation_suite_source,
        "activation suite full activation action plan installed",
        failures,
    )
    check(
        "collect_activation_suite" in api_source
        and '"/admin/activation-suite"' in api_source
        and '"/admin/activation-suite/report"' in api_source
        and "fluxmind-activation-suite.md" in api_source,
        "activation suite admin API installed",
        failures,
    )
    check(
        "admin_activation_suite" in api_source
        and "admin_activation_suite_with_report" in api_source
        and "ActivationSuiteRequest" in api_source
        and "live_reports=live_reports" in api_source
        and "openapi_schema=app.openapi()" in api_source
        and "verify_api_token" in api_source
        and "ActivationSuiteResponse" in api_source,
        "activation suite admin API auth and live-report input installed",
        failures,
    )
    activation_suite_cli = (PROJECT_ROOT / "scripts" / "activation_suite.py").read_text(
        encoding="utf-8"
    )
    check(
        "--require-target" in activation_suite_cli
        and "full_activation" in activation_suite_cli
        and "collect_activation_suite" in activation_suite_cli,
        "activation suite CLI installed",
        failures,
    )
    check(
        "_load_openapi_schema" in activation_suite_cli
        and "api.app.openapi()" in activation_suite_cli
        and "openapi_schema=_load_openapi_schema()" in activation_suite_cli,
        "activation suite CLI OpenAPI contract input installed",
        failures,
    )
    openapi_contract_source = (PROJECT_ROOT / "src" / "openapi_contract.py").read_text(
        encoding="utf-8"
    )
    check(
        "collect_openapi_contract" in openapi_contract_source
        and "REQUIRED_PATH_METHODS" in openapi_contract_source
        and "protected_auth_headers_missing" in openapi_contract_source
        and "operation_fingerprint" in openapi_contract_source,
        "OpenAPI contract collector installed",
        failures,
    )
    check(
        "verify_openapi_contract_snapshot" in openapi_contract_source
        and "format_openapi_contract_snapshot_verify_markdown" in openapi_contract_source
        and "snapshot_contract_drift" in openapi_contract_source,
        "OpenAPI contract snapshot verifier installed",
        failures,
    )
    check(
        "snapshot_contract_shape_invalid" in openapi_contract_source
        and "snapshot_raw_schema_included" in openapi_contract_source
        and "snapshot_valid" in openapi_contract_source,
        "OpenAPI contract snapshot verifier no-secret projection installed",
        failures,
    )
    check(
        "format_openapi_contract_markdown" in openapi_contract_source
        and "raw_schema_exported" in openapi_contract_source
        and "paths_exported" in openapi_contract_source
        and "secrets_exported" in openapi_contract_source,
        "OpenAPI contract no-secret markdown installed",
        failures,
    )
    openapi_contract_cli = (PROJECT_ROOT / "scripts" / "openapi_contract.py").read_text(
        encoding="utf-8"
    )
    check(
        "--require-local-contract" in openapi_contract_cli
        and "--verify-snapshot" in openapi_contract_cli
        and "--require-no-drift" in openapi_contract_cli
        and "collect_openapi_contract" in openapi_contract_cli
        and "api.app.openapi()" in openapi_contract_cli,
        "OpenAPI contract CLI installed",
        failures,
    )
    safe_cli_source = (PROJECT_ROOT / "scripts" / "_safe_cli.py").read_text(
        encoding="utf-8"
    )
    no_secret_cli_sources = [
        (PROJECT_ROOT / "scripts" / path).read_text(encoding="utf-8")
        for path in (
            "product_readiness.py",
            "provider_readiness.py",
            "quality_readiness.py",
            "product_activation_rehearsal.py",
            "collaboration_readiness.py",
            "provider_runtime_rehearsal.py",
            "activation_suite.py",
            "openapi_contract.py",
            "platform_migration_preflight.py",
            "platform_migration_rehearsal.py",
            "storage_schema.py",
        )
    ]
    check(
        "def format_os_error" in safe_cli_source
        and "sanitize_cli_error_message" in safe_cli_source
        and "SENSITIVE_CLI_ERROR_PATTERNS" in safe_cli_source
        and "CLI_ERROR_REDACTION" in safe_cli_source
        and all("format_os_error" in source for source in no_secret_cli_sources)
        and all("error: {format_os_error(exc)}" in source for source in no_secret_cli_sources),
        "no-secret readiness CLI OS errors omit raw paths",
        failures,
    )
    check(
        "collect_openapi_contract" in api_source
        and '"/admin/openapi-contract"' in api_source
        and '"/admin/openapi-contract/report"' in api_source
        and '"/admin/openapi-contract/verify"' in api_source
        and '"/admin/openapi-contract/verify/report"' in api_source
        and "fluxmind-openapi-contract.md" in api_source,
        "OpenAPI contract admin API installed",
        failures,
    )
    check(
        "record_admin_check_event" in api_source
        and 'kind="admin_check"' in api_source
        and "_record_openapi_contract_check" in api_source
        and "_record_collaboration_readiness_check" in api_source
        and "_record_activation_suite_check" in api_source,
        "admin readiness check runtime events installed",
        failures,
    )
    admin_source = (PROJECT_ROOT / "src" / "admin.py").read_text(encoding="utf-8")
    check(
        "admin_checks" in admin_source
        and "fluxmind_admin_checks_recent_total" in admin_source
        and "_admin_check_event_admin_dict" in admin_source,
        "admin readiness check status summary installed",
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
    check(
        "collect_product_activation_rehearsal" in app_source
        and "run_product_activation_rehearsal" in app_source
        and "product_activation_rehearsal_status" in app_source
        and "format_product_activation_rehearsal_markdown" in app_source,
        "Streamlit product activation rehearsal on-demand panel installed",
        failures,
    )
    check(
        "collect_collaboration_readiness" in app_source
        and "run_collaboration_readiness" in app_source
        and "collaboration_readiness_status" in app_source
        and "format_collaboration_readiness_markdown" in app_source,
        "Streamlit collaboration readiness on-demand panel installed",
        failures,
    )
    check(
        "collect_provider_runtime_rehearsal" in app_source
        and "run_provider_runtime_rehearsal" in app_source
        and "provider_runtime_rehearsal_status" in app_source
        and "format_provider_runtime_rehearsal_markdown" in app_source,
        "Streamlit provider runtime rehearsal on-demand panel installed",
        failures,
    )
    check(
        "collect_quality_readiness" in app_source
        and "run_quality_readiness" in app_source
        and "quality_readiness_status" in app_source,
        "Streamlit quality readiness on-demand panel installed",
        failures,
    )
    check(
        "quality_readiness_live_report" in app_source
        and "json.loads(quality_readiness_live_report" in app_source
        and "format_quality_readiness_markdown" in app_source,
        "Streamlit quality readiness live-report upload installed",
        failures,
    )
    check(
        "collect_activation_suite" in app_source
        and "run_activation_suite" in app_source
        and "activation_suite_status" in app_source,
        "Streamlit activation suite on-demand panel installed",
        failures,
    )
    check(
        "activation_suite_live_report" in app_source
        and "json.loads(activation_suite_live_report" in app_source
        and "live_reports=live_reports" in app_source,
        "Streamlit activation suite live-report upload installed",
        failures,
    )
    check(
        "local_foundation_requires" in app_source
        and "OpenAPI contract" in app_source
        and "openapi_schema=api.app.openapi()" in app_source,
        "Streamlit activation suite OpenAPI contract input installed",
        failures,
    )
    check(
        "collect_openapi_contract" in app_source
        and "run_openapi_contract" in app_source
        and "openapi_contract_status" in app_source
        and "run_openapi_contract_verify" in app_source
        and "openapi_contract_verify_status" in app_source
        and "format_openapi_contract_markdown" in app_source,
        "Streamlit OpenAPI contract on-demand panel installed",
        failures,
    )
    check("render_retention_preview" in app_source and "collect_retention_preview" in app_source, "Streamlit retention preview panel installed", failures)
    check("retention_delete" in app_source and "apply_retention_delete" in app_source, "Streamlit retention delete guard installed", failures)
    check(
        "render_runtime_events" in app_source
        and "event_kind_filter" in app_source
        and "runtime_event_to_safe_dict(event, include_request_id=False)" in app_source
        and "event.__dict__" not in app_source,
        "Streamlit runtime events panel installed",
        failures,
    )
    check(
        "RUNTIME_EVENT_KIND_FILTER_OPTIONS" in app_source
        and "provider_quota_guard" in app_source
        and "product_quota" in app_source
        and "product_rbac" in app_source
        and "product_registry_admin" in app_source
        and "share_link_admin" in app_source
        and "admin_check" in app_source,
        "Streamlit runtime event guard filters installed",
        failures,
    )
    check("status_provider_failures" in app_source, "Streamlit provider failure status panel installed", failures)
    check("status_query_usage" in app_source, "Streamlit query usage status panel installed", failures)
    check("status_retrieval_traces" in app_source and "retrieval_trace" in app_source, "Streamlit retrieval trace status panel installed", failures)
    check("status_cost_pricing" in app_source, "Streamlit query cost pricing panel installed", failures)
    check("status_code_execution" in app_source and "alert_thresholds" in app_source, "Streamlit code execution alert status installed", failures)
    check("status_api_access" in app_source and '"api_access"' in app_source, "Streamlit API access audit status installed", failures)
    check("status_admin_checks" in app_source and '"admin_checks"' in app_source, "Streamlit admin check status installed", failures)
    check("rate_limited_recent" in app_source and '"rate_limit"' in app_source, "Streamlit API rate-limit status installed", failures)
    check("download_admin_metrics" in app_source and "format_admin_metrics" in app_source, "Streamlit metrics download installed", failures)
    check("status_execution_policy" in app_source and "code_execution_allowed_imports" in app_source, "Streamlit execution policy status installed", failures)
    check("status_storage" in app_source and "storage_readiness" in app_source, "Streamlit storage readiness panel installed", failures)
    check("distributed_job_store" in app_source, "Streamlit distributed job-store readiness panel installed", failures)
    check("status_storage_inventory" in app_source, "Streamlit storage inventory panel installed", failures)
    check("status_storage_schemas" in app_source and "storage_schemas" in app_source, "Streamlit storage schema panel installed", failures)
    check("status_platform_readiness" in app_source and "platform_readiness" in app_source, "Streamlit platform readiness panel installed", failures)
    check(
        "collect_platform_migration_rehearsal" in app_source
        and "run_platform_migration_rehearsal" in app_source
        and "platform_migration_rehearsal_status" in app_source
        and "format_storage_migration_rehearsal_markdown" in app_source,
        "Streamlit platform migration rehearsal on-demand panel installed",
        failures,
    )
    check("status_product_readiness" in app_source and "product_readiness" in app_source, "Streamlit product readiness panel installed", failures)
    check(
        "render_product_registry_management" in app_source
        and "product_registry_management" in app_source,
        "Streamlit product registry management panel installed",
        failures,
    )
    check(
        "STREAMLIT_PRODUCT_REGISTRY_MANAGEMENT_ENABLED" in app_source
        and '"management_enabled": STREAMLIT_PRODUCT_REGISTRY_MANAGEMENT_ENABLED' in app_source
        and "if not STREAMLIT_PRODUCT_REGISTRY_MANAGEMENT_ENABLED:" in app_source
        and app_source.find("if not STREAMLIT_PRODUCT_REGISTRY_MANAGEMENT_ENABLED:")
        < app_source.find("registry = LocalProductRegistry()"),
        "Streamlit product registry management requires explicit enable flag",
        failures,
    )
    product_registry_management_block = ""
    if (
        "def render_product_registry_management()" in app_source
        and "def render_share_link_registry_management()" in app_source
    ):
        product_registry_management_block = app_source.split(
            "def render_product_registry_management()", 1
        )[1].split("def render_share_link_registry_management()", 1)[0]
    check(
        "safe_streamlit_error_message" in app_source
        and "st.error(str(exc))" not in product_registry_management_block
        and product_registry_management_block.count(
            "st.error(safe_streamlit_error_message(exc))"
        )
        >= 5,
        "Streamlit product registry management sanitizes error output",
        failures,
    )
    check(
        "STREAMLIT_SHARE_LINK_MANAGEMENT_ENABLED" in app_source
        and '"management_enabled": STREAMLIT_SHARE_LINK_MANAGEMENT_ENABLED' in app_source
        and "if not STREAMLIT_SHARE_LINK_MANAGEMENT_ENABLED:" in app_source
        and "type=\"password\"" in app_source
        and app_source.find("if not STREAMLIT_SHARE_LINK_MANAGEMENT_ENABLED:")
        < app_source.find("registry = LocalShareLinkRegistry()"),
        "Streamlit share-link management requires explicit enable flag",
        failures,
    )
    share_link_management_block = ""
    if (
        "def render_share_link_registry_management()" in app_source
        and "def render_admin_status()" in app_source
    ):
        share_link_management_block = app_source.split(
            "def render_share_link_registry_management()", 1
        )[1].split("def render_admin_status()", 1)[0]
    check(
        "safe_streamlit_error_message" in app_source
        and "format_os_error" in app_source
        and "sanitize_cli_error_message" in app_source
        and "st.error(str(exc))" not in share_link_management_block
        and share_link_management_block.count("st.error(safe_streamlit_error_message(exc))") >= 5,
        "Streamlit share-link management sanitizes error output",
        failures,
    )
    admin_status_block = ""
    if "def render_admin_status()" in app_source and "def render_retention_preview()" in app_source:
        admin_status_block = app_source.split("def render_admin_status()", 1)[1].split(
            "def render_retention_preview()", 1
        )[0]
    check(
        "safe_streamlit_error_message" in app_source
        and "st.error(str(exc))" not in admin_status_block
        and admin_status_block.count("st.error(safe_streamlit_error_message(exc))") >= 8,
        "Streamlit admin on-demand panels sanitize OS error output",
        failures,
    )
    check(
        "def safe_streamlit_error_text" in app_source
        and ".format(error=exc)" not in app_source
        and 'safe_streamlit_error_text(text["quality_readiness_report_invalid"], exc)'
        in admin_status_block
        and 'safe_streamlit_error_text(text["activation_suite_report_invalid"], exc)'
        in admin_status_block
        and 'safe_streamlit_error_text(text["openapi_contract_snapshot_invalid"], exc)'
        in admin_status_block
        and 'safe_streamlit_error_text(text["runtime_restore_invalid_manifest"], exc)'
        in admin_status_block
        and 'safe_streamlit_error_text(text["upload_failed"], exc)' in app_source,
        "Streamlit user-upload/admin validation errors sanitize exception output",
        failures,
    )
    check("status_provider_readiness" in app_source and "provider_readiness" in app_source, "Streamlit provider readiness panel installed", failures)
    check("status_runtime_manifest" in app_source and "download_runtime_manifest" in app_source, "Streamlit runtime manifest panel installed", failures)
    check("runtime_restore_manifest_upload" in app_source and "format_runtime_restore_check_markdown" in app_source, "Streamlit runtime restore-check panel installed", failures)
    check("artifact_id" in app_source and "artifact_metadata" in app_source and "artifact_search" in app_source, "Streamlit artifact reference metadata installed", failures)
    artifact_gallery_block = ""
    if (
        "def render_latest_artifacts()" in app_source
        and "def render_product_registry_management()" in app_source
    ):
        artifact_gallery_block = app_source.split("def render_latest_artifacts()", 1)[
            1
        ].split("def render_product_registry_management()", 1)[0]
    check(
        "st.caption(str(exc))" not in artifact_gallery_block
        and "st.caption(safe_streamlit_error_message(exc))" in artifact_gallery_block,
        "Streamlit artifact gallery sanitizes download error output",
        failures,
    )
    job_result_block = ""
    if "def render_job_result(job)" in app_source and "def job_sidebar_summary(job)" in app_source:
        job_result_block = app_source.split("def render_job_result(job)", 1)[1].split(
            "def job_sidebar_summary(job)", 1
        )[0]
    check(
        "safe_streamlit_status_message(error_message, fallback=job.status)" in job_result_block
        and '(job.error or {}).get("message", job.status)' not in job_result_block,
        "Streamlit job result sanitizes failure message output",
        failures,
    )
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
    profile_report_block = ""
    if (
        "selected_profile = st.selectbox(" in app_source
        and 'if st.button(text["activate_profile"]' in app_source
    ):
        profile_report_block = app_source.split("selected_profile = st.selectbox(", 1)[
            1
        ].split('if st.button(text["activate_profile"]', 1)[0]
    check(
        "safe_streamlit_error_message(exc)" in profile_report_block
        and "error = normalize_exception(exc)" not in profile_report_block
        and "error.message" not in profile_report_block,
        "Streamlit corpus profile report sanitizes failure message output",
        failures,
    )
    api_source = (PROJECT_ROOT / "api.py").read_text(encoding="utf-8")
    check("verify_configured_api_key_token" in api_source and "api_key_registry_configured" in api_source, "API auth supports local key registry", failures)
    check("enforce_product_quota" in api_source and "product_quota_guard" in api_source, "API query paths support local product quota guard", failures)
    check("enforce_product_rbac" in api_source and "product_rbac_guard" in api_source, "API write paths support local product RBAC guard", failures)
    check(
        "/admin/product-registry/workspaces" in api_source
        and "admin_product_registry_create_workspace" in api_source
        and "admin_product_registry_check_permission" in api_source,
        "API product registry management routes installed",
        failures,
    )
    check(
        "/admin/share-links/status" in api_source
        and "/admin/share-links/{link_id}/revoke" in api_source
        and "admin_share_link_create" in api_source
        and "admin_share_link_resolve" in api_source,
        "API share-link registry management routes installed",
        failures,
    )
    check(
        "record_share_link_admin_event" in api_source
        and 'kind="share_link_admin"' in api_source
        and "share_tokens_exported" in api_source
        and "share_urls_exported" in api_source,
        "API share-link admin events are no-secret",
        failures,
    )
    check(
        "def enforce_product_registry_admin_read" in api_source
        and api_source.count("enforce_product_registry_admin_read(") >= 4
        and 'endpoint="/admin/product-registry/workspaces"' in api_source
        and 'endpoint="/admin/product-registry/workspaces/{workspace_id}"' in api_source
        and 'endpoint="/admin/product-registry/permissions/check"' in api_source,
        "API product registry read routes use local product RBAC guard",
        failures,
    )
    check("/artifacts" in api_source, "artifact export route installed", failures)
    check("job_kind: str | None" in api_source and "kind: str | None" in api_source, "artifact metadata filters installed", failures)
    check(
        "def public_error_detail" in api_source
        and "detail=str(exc)" not in api_source
        and 'public_error_detail("artifact_export_denied")' in api_source
        and api_source.count('public_error_detail("invalid_corpus_source_path")') >= 5,
        "API validation errors sanitize exception text",
        failures,
    )
    check(
        "RequestValidationError" in api_source
        and "def public_request_validation_errors" in api_source
        and '"input"' not in api_source.split("def public_request_validation_errors", 1)[1].split(
            "def api_auth_context", 1
        )[0],
        "API request validation errors omit submitted input values",
        failures,
    )
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
    check(
        "SHARE_LINK_TOKEN_STORE_FILE" in storage_manifest_source
        and "share_link_registry_sqlite" in storage_manifest_source,
        "runtime manifest includes share-link registry state",
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
    test_api_source = (PROJECT_ROOT / "tests" / "test_api.py").read_text(encoding="utf-8")
    check(
        "def public_job_request" in api_source
        and "def public_job_result" in api_source
        and '"source_path_count"' in api_source,
        "index rebuild job API projection redacts source paths",
        failures,
    )
    check(
        "def public_job_error" in api_source
        and "def public_job_logs" in api_source
        and "def public_code_runtime_metadata" in api_source
        and '"stdout" not in job["result"]' in test_api_source
        and '"files" not in job["request"]' in test_api_source,
        "job detail API projection redacts code input and output content",
        failures,
    )
    check("status: str | None" in api_source and "kind: str | None" in api_source and "q=q" in api_source, "job metadata filters installed", failures)
    job_summary_start = api_source.find("def job_summary_to_dict")
    job_summary_end = api_source.find("\ndef existing_idempotent_job", job_summary_start)
    job_summary_source = api_source[job_summary_start:job_summary_end]
    job_detail_start = api_source.find("def job_to_dict")
    job_detail_end = api_source.find("\ndef _source_path_count", job_detail_start)
    job_detail_source = api_source[job_detail_start:job_detail_end]
    check(
        "owner_label_present" in job_summary_source
        and '"owner_label":' not in job_summary_source,
        "job list summaries avoid raw owner labels",
        failures,
    )
    check("idempotency_key" in api_source and "existing_idempotent_job" in api_source, "job idempotency API installed", failures)
    check(
        "idempotency_key_fingerprint" in api_source
        and "idempotency_key_exported" in api_source
        and '"idempotency_key": record.idempotency_key' not in api_source
        and "test_job_response_redacts_secret_like_idempotency_key" in test_api_source,
        "job detail API projection redacts raw idempotency keys",
        failures,
    )
    check(
        "owner_id_fingerprint" in job_detail_source
        and "owner_label_fingerprint" in job_detail_source
        and "owner_exported" in job_detail_source
        and '"owner_id":' not in job_detail_source
        and '"owner_label":' not in job_detail_source
        and "test_job_response_redacts_owner_metadata" in test_api_source,
        "job detail API projection redacts raw owner metadata",
        failures,
    )
    check(
        "def public_request_id" in api_source
        and "sanitize_runtime_event_request_id" in api_source
        and "**request_id" in job_detail_source
        and '"request_id": record.request_id' not in api_source
        and "test_job_response_redacts_unsafe_request_id" in test_api_source,
        "job detail API projection redacts unsafe request IDs",
        failures,
    )
    check("owner_id: str | None" in api_source and "request_ownership" in api_source, "API ownership metadata fields installed", failures)
    check("/jobs/code/octave-local" in api_source and "/jobs/async/code/octave-local" in api_source, "Octave-compatible job routes installed", failures)
    check("/jobs/{job_id}/retry" in api_source, "job retry route installed", failures)
    check("/jobs/{job_id}/retry-scheduled" in api_source, "scheduled retry route installed", failures)
    check('"logs": public_job_logs(record)' in api_source, "job transition logs exposed by public API projection", failures)
    check("append_runtime_event" in api_source, "query provider failures are recorded", failures)
    check(
        "runtime_ownership_metadata" in api_source
        and "metadata.update(runtime_ownership_metadata" in api_source
        and "**ownership" not in api_source,
        "query runtime events avoid raw owner metadata",
        failures,
    )
    check(
        "record_query_exception_event" in api_source
        and "provider_quota_guard" in api_source
        and "ProviderQuotaGuardError" in api_source,
        "query provider quota guard denials are recorded separately",
        failures,
    )
    check(
        "product_workspace_present" in api_source
        and "product_workspace_id" not in api_source,
        "product runtime events avoid raw workspace IDs",
        failures,
    )
    check("api_access_audit_middleware" in api_source and "kind=\"api_access\"" in api_source, "API access audit events are recorded", failures)
    check(
        "_api_access_route_metadata" in api_source
        and '"route_present"' in api_source
        and '"route_fingerprint"' in api_source
        and '"path": request.url.path' not in api_source
        and "api_access.event_log_failed path=" not in api_source,
        "API access audit events avoid raw request paths",
        failures,
    )
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
    check(
        "runtime_ownership_metadata" in jobs_source
        and "**runtime_ownership_metadata" in jobs_source,
        "code execution runtime events avoid raw owner metadata",
        failures,
    )
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
    check(
        "could not be materialized" in providers_source and "current_name" in providers_source,
        "local execution input materialization conflict guard installed",
        failures,
    )
    check("entrypoint.is_file()" in providers_source, "local execution entrypoint regular-file guard installed", failures)
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
    check(
        "owner_count" in admin_source
        and "by_ownership_source" in admin_source
        and "job_ownership_source_counts" in admin_source,
        "admin ownership-source summaries installed",
        failures,
    )
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
    check("SHARE_LINK_TOKEN_STORE_FILE" in admin_source and "share_link_registry_sqlite" in admin_source, "admin storage inventory includes share-link registry state", failures)
    check("storage_schema_status" in admin_source and "storage_schemas" in admin_source, "admin storage schema inventory installed", failures)
    check("distributed_job_store_status" in admin_source and "external_job_store_configured" in admin_source, "admin distributed job-store readiness installed", failures)
    check("platform_readiness_status" in admin_source and "distributed_worker_acceptance" in admin_source, "admin platform readiness installed", failures)
    check("collect_product_readiness" in admin_source and "product_readiness" in admin_source, "admin product readiness installed", failures)
    check("fluxmind_product_local_foundation_ready" in admin_source and "fluxmind_product_activation_ready" in admin_source, "admin product readiness metrics installed", failures)
    check("fluxmind_product_quota_guard_enabled" in admin_source, "admin product quota guard metric installed", failures)
    check("fluxmind_product_rbac_guard_enabled" in admin_source, "admin product RBAC guard metric installed", failures)
    check("collect_provider_readiness" in admin_source and "provider_readiness" in admin_source, "admin provider readiness installed", failures)
    check("fluxmind_provider_local_foundation_ready" in admin_source and "fluxmind_provider_activation_ready" in admin_source, "admin provider readiness metrics installed", failures)
    storage_schema_source = (PROJECT_ROOT / "src" / "storage_schema.py").read_text(encoding="utf-8")
    storage_schema_cli = (PROJECT_ROOT / "scripts" / "storage_schema.py").read_text(encoding="utf-8")
    check("STORAGE_SCHEMA_VERSION" in storage_schema_source and "missing_required_columns" in storage_schema_source, "storage schema drift checks installed", failures)
    check("API_KEY_COLUMNS" in storage_schema_source and "api_key_registry_sqlite" in storage_schema_source, "API key registry storage schema installed", failures)
    check("PRODUCT_USER_COLUMNS" in storage_schema_source and "product_registry_sqlite" in storage_schema_source, "product registry storage schema installed", failures)
    check("SHARE_LINK_COLUMNS" in storage_schema_source and "share_link_registry_sqlite" in storage_schema_source, "share-link registry storage schema installed", failures)
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
        eval_config.get("quality_gates", {}).get("minimum_code_output_case_count", 0) >= 13
        and len(eval_config.get("code_output_cases", [])) >= 13,
        "RAG eval code-output case gate installed",
        failures,
    )
    required_code_languages = eval_config.get("quality_gates", {}).get(
        "required_code_output_languages", []
    )
    check(
        eval_config.get("quality_gates", {}).get("minimum_code_output_pass_rate", 0) >= 1.0
        and "python" in required_code_languages
        and "octave" in required_code_languages,
        "RAG eval code-output language/pass gates installed",
        failures,
    )
    required_code_templates = eval_config.get("quality_gates", {}).get(
        "required_code_output_template_ids", []
    )
    code_output_cases = eval_config.get("code_output_cases", [])
    has_python_template_case = any(
        case.get("template_id") == "smc_reaching_law"
        for case in code_output_cases
    )
    has_octave_template_case = any(
        case.get("template_id") == "pmsm_current_decay"
        and case.get("language") == "octave"
        and case.get("expected_runtime_unavailable")
        for case in code_output_cases
    )
    check(
        "smc_reaching_law" in required_code_templates
        and "pmsm_current_decay" in required_code_templates
        and has_python_template_case
        and has_octave_template_case,
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
        eval_config.get("quality_gates", {}).get("minimum_pdf_structure_case_count", 0) >= 30
        and {"equation", "table", "figure", "algorithm"}.issubset(required_pdf_kinds)
        and len(eval_config.get("pdf_structure_cases", [])) >= 30,
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
        "_request_id_evidence" in evaluation_source
        and "request_id_present" in evaluation_source
        and "request_id_redacted" in evaluation_source
        and "request_id: str" not in evaluation_source,
        "live eval JSON report request IDs are redacted",
        failures,
    )
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
    check(
        "request_id_present=" in evaluate_rag_source
        and "request_id_redacted=" in evaluate_rag_source
        and "request_id={result.request_id}" not in evaluate_rag_source,
        "live eval CLI request IDs are redacted",
        failures,
    )
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
            "grep -q 'STREAMLIT_PRODUCT_REGISTRY_MANAGEMENT_ENABLED' /opt/fluxmind/app.py; "
            "grep -q 'product_registry_management_disabled' /opt/fluxmind/app.py; "
            "grep -q 'STREAMLIT_SHARE_LINK_MANAGEMENT_ENABLED' /opt/fluxmind/app.py; "
            "grep -q 'share_link_management_disabled' /opt/fluxmind/app.py; "
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
            "grep -q '\"logs\": public_job_logs(record)' /opt/fluxmind/api.py; "
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
            "test -f /opt/fluxmind/src/share_links.py; "
            "grep -q 'LocalApiKeyRegistry' /opt/fluxmind/src/api_keys.py; "
            "grep -q 'token_hash' /opt/fluxmind/src/api_keys.py; "
            "grep -q 'api_key_registry_backend_status' /opt/fluxmind/src/api_keys.py; "
            "grep -q 'LocalProductRegistry' /opt/fluxmind/src/product_registry.py; "
            "grep -q 'product_registry_backend_status' /opt/fluxmind/src/product_registry.py; "
            "grep -q 'quota_decision' /opt/fluxmind/src/product_registry.py; "
            "grep -q 'permission_decision' /opt/fluxmind/src/product_registry.py; "
            "grep -q 'LocalShareLinkRegistry' /opt/fluxmind/src/share_links.py; "
            "grep -q 'share_link_registry_backend_status' /opt/fluxmind/src/share_links.py; "
            "grep -q 'verify_configured_api_key_token' /opt/fluxmind/api.py; "
            "grep -q 'api_key_registry_configured' /opt/fluxmind/api.py; "
            "grep -q 'enforce_product_quota' /opt/fluxmind/api.py; "
            "grep -q 'enforce_product_rbac' /opt/fluxmind/api.py; "
            "grep -q 'enforce_product_registry_admin_read' /opt/fluxmind/api.py; "
            "grep -q 'endpoint=\"/admin/product-registry/workspaces\"' /opt/fluxmind/api.py; "
            "grep -q 'endpoint=\"/admin/product-registry/permissions/check\"' /opt/fluxmind/api.py; "
            "grep -q '/admin/share-links/status' /opt/fluxmind/api.py; "
            "grep -q 'record_share_link_admin_event' /opt/fluxmind/api.py; "
            "grep -q 'api_key_registry_sqlite' /opt/fluxmind/src/admin.py; "
            "grep -q 'product_registry_sqlite' /opt/fluxmind/src/admin.py; "
            "grep -q 'share_link_registry_sqlite' /opt/fluxmind/src/admin.py; "
            "grep -q 'fluxmind_product_quota_guard_enabled' /opt/fluxmind/src/admin.py; "
            "grep -q 'fluxmind_product_rbac_guard_enabled' /opt/fluxmind/src/admin.py; "
            "grep -q 'api_key_registry_sqlite' /opt/fluxmind/src/storage_schema.py; "
            "grep -q 'product_registry_sqlite' /opt/fluxmind/src/storage_schema.py; "
            "grep -q 'share_link_registry_sqlite' /opt/fluxmind/src/storage_schema.py; "
            "grep -q 'api_key_registry_sqlite' /opt/fluxmind/src/storage_manifest.py; "
            "grep -q 'product_registry_sqlite' /opt/fluxmind/src/storage_manifest.py; "
            "grep -q 'share_link_registry_sqlite' /opt/fluxmind/src/storage_manifest.py; "
            "grep -q 'create' /opt/fluxmind/scripts/api_key_registry.py; "
            "grep -q 'requires --format json' /opt/fluxmind/scripts/api_key_registry.py; "
            "grep -q 'revoke' /opt/fluxmind/scripts/api_key_registry.py; "
            "test -f /opt/fluxmind/scripts/share_link_registry.py; "
            "grep -q 'one-time share token' /opt/fluxmind/scripts/share_link_registry.py; "
            "grep -q 'bootstrap-local' /opt/fluxmind/scripts/product_registry.py; "
            "grep -q 'check-permission' /opt/fluxmind/scripts/product_registry.py; "
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
            "grep -q 'provider_quota_guard_invalid_limit' /opt/fluxmind/src/provider_readiness.py; "
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
            "grep -q 'live_answer_pass_rate' /opt/fluxmind/src/quality_readiness.py; "
            "grep -q 'average_live_answer_term_coverage' /opt/fluxmind/src/quality_readiness.py; "
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
