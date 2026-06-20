import json

from src.activation_suite import (
    ACTIVATION_SUITE_STATE_DIR,
    _quality_summary,
    collect_activation_suite,
    format_activation_suite_markdown,
)


def _api_openapi_schema():
    import api

    return api.app.openapi()


def test_activation_suite_collects_no_secret_local_foundation(tmp_path):
    root = tmp_path / "suite"

    status = collect_activation_suite(
        root=root,
        generated_at="2026-06-19T00:00:00+00:00",
        openapi_schema=_api_openapi_schema(),
    )

    payload = json.dumps(status, ensure_ascii=False, sort_keys=True)
    assert status["mode"] == "activation_suite"
    assert status["ok"] is True
    assert status["local_foundation_ready"] is True
    assert status["full_activation_ready"] is False
    assert status["check_count"] == 7
    assert status["failed_check_count"] == 0
    assert status["full_activation_blocker_count"] == 5
    assert status["activation_step_count"] == 5
    assert status["local_only"] is True
    assert status["raw_reports_included"] is False
    assert status["content_exported"] is False
    assert status["secrets_exported"] is False
    assert status["paths_exported"] is False
    assert status["connectivity_checked"] is False
    assert status["checks"]["product_readiness"]["ok"] is True
    assert status["checks"]["product_readiness"]["activation_ready"] is False
    assert status["checks"]["product_readiness"]["activation_blockers"] == [
        "multi_user_identity_not_configured",
        "api_key_lifecycle_not_configured",
        "identity_quota_store_not_configured",
        "billing_provider_not_configured",
        "billing_attribution_not_enabled",
    ]
    assert status["checks"]["product_activation"]["ok"] is True
    assert status["checks"]["product_activation"]["activation_ready"] is True
    assert status["checks"]["product_activation"]["workspace_isolation_ok"] is True
    assert status["checks"]["product_activation"]["workspace_count"] == 2
    assert status["checks"]["collaboration_readiness"]["ok"] is True
    assert status["checks"]["collaboration_readiness"]["safe_default_ready"] is True
    assert status["checks"]["collaboration_readiness"]["activation_ready"] is False
    assert status["checks"]["collaboration_readiness"]["policy_scenario_count"] == 13
    assert status["checks"]["openapi_contract"]["ok"] is True
    assert status["checks"]["openapi_contract"]["local_contract_ready"] is True
    assert status["checks"]["openapi_contract"]["raw_schema_exported"] is False
    assert "openapi_contract.ok" in status["required_gates"]["local_foundation"]
    assert status["checks"]["provider_runtime"]["ok"] is True
    assert status["checks"]["provider_runtime"]["provider_activation_ready"] is False
    assert status["checks"]["storage_migration"]["ok"] is True
    assert status["checks"]["storage_migration"]["job_store_manifest_ready"] is True
    assert status["checks"]["quality_readiness"]["local_foundation_ready"] is True
    next_evidence = status["checks"]["quality_readiness"]["next_evidence"]
    assert next_evidence["target"] == "small_group"
    assert next_evidence["gap_count"] == 1
    assert next_evidence["evidence_sources"] == ["live_eval_report"]
    assert next_evidence["gaps"][0]["metric"] == "live_retrieval_result_count"
    assert (
        "product_readiness_activation_not_ready"
        in status["blockers"]["full_activation"]
    )
    assert "provider_activation_not_ready" in status["blockers"]["full_activation"]
    assert "collaboration_activation_not_ready" in status["blockers"]["full_activation"]
    assert "quality_community_not_ready" in status["blockers"]["full_activation"]
    action_plan = status["activation_action_plan"]
    assert action_plan["target"] == "full_activation"
    assert action_plan["ready"] is False
    assert action_plan["step_count"] == 5
    steps_by_area = {step["area"]: step for step in action_plan["steps"]}
    assert set(steps_by_area) == {
        "product_readiness",
        "collaboration_readiness",
        "provider_activation",
        "platform_migration",
        "community_quality",
    }
    assert "product_readiness.py" in steps_by_area["product_readiness"][
        "verification_command"
    ]
    assert "multi_user_identity_not_configured" in steps_by_area["product_readiness"][
        "blockers"
    ]
    assert "collaboration_readiness.py" in steps_by_area["collaboration_readiness"][
        "verification_command"
    ]
    assert "share_links_disabled" in steps_by_area["collaboration_readiness"][
        "blockers"
    ]
    assert "provider_readiness.py" in steps_by_area["provider_activation"][
        "verification_command"
    ]
    assert "platform_migration_preflight.py" in steps_by_area["platform_migration"][
        "verification_command"
    ]
    community_step = steps_by_area["community_quality"]
    assert "--require-target community" in community_step["verification_command"]
    assert {
        "corpus_manifest",
        "eval_baseline",
        "live_eval_report",
    }.issubset({step["evidence_source"] for step in community_step["substeps"]})
    assert "<api-base-url>" in payload
    assert "<report.json>" in payload
    assert str(tmp_path) not in payload
    for sensitive in (
        "fmk_",
        "hunter2",
        "file://",
        "api_keys.sqlite3",
        "product_registry.sqlite3",
        "share-token",
        "workspace-",
        "corpus-",
        "Provider rehearsal SMC observer diagram",
        "provider-runtime-rehearsal-ok",
        "summary.txt",
        "main.py",
        "main.m",
    ):
        assert sensitive not in payload


def test_activation_suite_markdown_is_no_secret(tmp_path):
    root = tmp_path / "suite"
    status = collect_activation_suite(
        root=root,
        generated_at="2026-06-19T00:00:00+00:00",
        openapi_schema=_api_openapi_schema(),
    )

    markdown = format_activation_suite_markdown(status)

    assert "# FluxMind Activation Suite" in markdown
    assert "Local foundation ready: true" in markdown
    assert "Full activation ready: false" in markdown
    assert "Checks: 7" in markdown
    assert "Failed checks: 0" in markdown
    assert "Full activation blockers: 5" in markdown
    assert "Activation steps: 5" in markdown
    assert "product_readiness: ok=true" in markdown
    assert "product_activation: ok=true" in markdown
    assert "collaboration_readiness: ok=true" in markdown
    assert "openapi_contract: ok=true" in markdown
    assert "local_contract_ready=true" in markdown
    assert "provider_runtime: ok=true" in markdown
    assert "storage_migration: ok=true" in markdown
    assert "quality_readiness: ok=true" in markdown
    assert "## Next Quality Evidence" in markdown
    assert "Target: small_group" in markdown
    assert "Sources: live_eval_report" in markdown
    assert "count live_retrieval_result_count" in markdown
    assert "source=live_eval_report" in markdown
    assert "## Activation Action Plan" in markdown
    assert "product_readiness: ready=false" in markdown
    assert "collaboration_readiness: ready=false" in markdown
    assert "provider_activation: ready=false" in markdown
    assert "platform_migration: ready=false" in markdown
    assert "community_quality: ready=false" in markdown
    assert "--require-target community" in markdown
    assert "product_readiness_activation_not_ready" in markdown
    assert "<api-base-url>" in markdown
    assert "<report.json>" in markdown
    assert "provider_activation_not_ready" in markdown
    assert "collaboration_activation_not_ready" in markdown
    assert "share_links_disabled" in markdown
    assert str(tmp_path) not in markdown
    for sensitive in (
        "fmk_",
        "hunter2",
        "file://",
        "api_keys.sqlite3",
        "product_registry.sqlite3",
        "share-token",
        "workspace-",
        "corpus-",
        "Provider rehearsal SMC observer diagram",
        "provider-runtime-rehearsal-ok",
        "summary.txt",
        "main.py",
        "main.m",
    ):
        assert sensitive not in markdown


def test_activation_suite_reuses_root_without_state_leakage(tmp_path):
    root = tmp_path / "suite"

    first = collect_activation_suite(
        root=root,
        generated_at="2026-06-19T00:00:00+00:00",
        openapi_schema=_api_openapi_schema(),
    )
    second = collect_activation_suite(
        root=root,
        generated_at="2026-06-19T00:00:01+00:00",
        openapi_schema=_api_openapi_schema(),
    )

    assert first["ok"] is True
    assert second["ok"] is True
    assert second["checks"]["product_activation"]["active_key_count"] == 2
    assert second["checks"]["product_activation"]["revoked_key_count"] == 1
    assert second["checks"]["openapi_contract"]["ok"] is True
    assert second["checks"]["storage_migration"]["job_store_manifest_ready"] is True
    assert (root / ACTIVATION_SUITE_STATE_DIR).is_dir()


def test_activation_suite_accepts_live_report_evidence_without_echoing_content(tmp_path):
    root = tmp_path / "suite"
    live_report = {
        "schema_version": 1,
        "secret_path": "/private/hunter2-live-report.json",
        "quality_maturity": {
            "metrics": {
                "live_retrieval_result_count": 107,
                "live_answer_result_count": 0,
            }
        },
        "summary": {
            "live_retrieval": {"total": 107, "ok": 107, "failed": 0},
            "live_answers": {"total": 0, "ok": 0, "failed": 0},
        },
    }

    status = collect_activation_suite(
        root=root,
        generated_at="2026-06-19T00:00:00+00:00",
        live_reports=[live_report],
        openapi_schema=_api_openapi_schema(),
    )
    payload = json.dumps(status, ensure_ascii=False, sort_keys=True)

    assert status["local_foundation_ready"] is True
    assert status["small_group_ready"] is True
    assert status["community_ready"] is False
    assert status["checks"]["quality_readiness"]["live_evidence_included"] is True
    next_evidence = status["checks"]["quality_readiness"]["next_evidence"]
    assert next_evidence["target"] == "community"
    assert next_evidence["gap_count"] >= 1
    assert {"corpus_manifest", "eval_baseline", "live_eval_report"}.issubset(
        set(next_evidence["evidence_sources"])
    )
    assert {
        "live_answer_result_count",
        "retrieval_eval_question_count",
    }.issubset({gap["metric"] for gap in next_evidence["gaps"]})
    assert "small_group_live_retrieval_result_count_gap" not in payload
    action_plan = status["activation_action_plan"]
    community_step = {
        step["area"]: step for step in action_plan["steps"]
    }["community_quality"]
    assert "small_group_live_retrieval_result_count_gap" not in json.dumps(
        community_step,
        sort_keys=True,
    )
    assert "community_live_answer_result_count_gap" in json.dumps(
        community_step,
        sort_keys=True,
    )
    assert "/private/hunter2-live-report.json" not in payload
    assert "hunter2" not in payload


def test_activation_suite_includes_openapi_contract_as_local_foundation_gate(tmp_path):
    bad_schema = {
        "openapi": "3.1.0",
        "info": {"title": "FluxMind API"},
        "paths": {
            "/query": {
                "post": {
                    "summary": "Run query",
                    "operationId": "run_query",
                    "parameters": [],
                    "responses": {"200": {"description": "OK"}},
                }
            }
        },
    }

    status = collect_activation_suite(
        root=tmp_path / "suite",
        generated_at="2026-06-19T00:00:00+00:00",
        openapi_schema=bad_schema,
    )

    openapi_contract = status["checks"]["openapi_contract"]
    assert status["ok"] is False
    assert status["local_foundation_ready"] is False
    assert "openapi_contract" in status["blockers"]["local_foundation"]
    assert "openapi_contract.ok" in status["required_gates"]["local_foundation"]
    assert openapi_contract["ok"] is False
    assert "required_operations_missing" in openapi_contract["blockers"]
    assert "protected_auth_headers_missing" in openapi_contract["blockers"]
    assert openapi_contract["required_operation_missing_count"] > 0
    assert openapi_contract["raw_schema_exported"] is False
    action_steps = {
        step["area"]: step for step in status["activation_action_plan"]["steps"]
    }
    assert "openapi_contract" in action_steps
    assert "openapi_contract.py" in action_steps["openapi_contract"]["command"]


def test_activation_suite_derives_legacy_quality_gap_summary():
    summary = _quality_summary(
        {
            "local_foundation_ready": True,
            "small_group_ready": False,
            "community_ready": False,
            "live_evidence_included": False,
            "gap_summary": [
                {
                    "target": "small_group",
                    "missing_metric_count": 1,
                    "count_gaps": [
                        {
                            "metric": "live_retrieval_result_count",
                            "actual": 0,
                            "expected": 50,
                            "gap": 50,
                        }
                    ],
                    "quality_gaps": [],
                }
            ],
            "blockers": {"local_foundation": [], "maturity": []},
        }
    )

    next_evidence = summary["next_evidence"]
    assert next_evidence["target"] == "small_group"
    assert next_evidence["gap_count"] == 1
    assert next_evidence["gaps"][0]["metric"] == "live_retrieval_result_count"
