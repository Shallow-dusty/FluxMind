import json

import api
from src.openapi_contract import (
    collect_openapi_contract,
    format_openapi_contract_markdown,
    format_openapi_contract_snapshot_verify_markdown,
    verify_openapi_contract_snapshot,
)


def test_openapi_contract_reports_current_api_without_exporting_schema():
    status = collect_openapi_contract(
        api.app.openapi(),
        generated_at="2026-06-20T00:00:00+00:00",
    )
    payload = json.dumps(status, ensure_ascii=False, sort_keys=True)

    assert status["mode"] == "openapi_contract"
    assert status["local_contract_ready"] is True
    assert status["openapi_version"].startswith("3.")
    assert status["required_operation_missing_count"] == 0
    assert status["undocumented_operation_count"] == 0
    assert status["response_missing_operation_count"] == 0
    assert len(status["operation_fingerprint"]) == 64
    assert status["protected_operations_missing_auth_headers"] == []
    assert status["route_groups"]["admin"]["present"] is True
    assert status["route_groups"]["query"]["present"] is True
    assert status["route_groups"]["jobs"]["present"] is True
    assert status["raw_schema_exported"] is False
    assert status["request_examples_exported"] is False
    assert status["content_exported"] is False
    assert status["secrets_exported"] is False
    assert status["paths_exported"] is False
    assert "hunter2" not in payload
    assert "sk-" not in payload
    assert "components" not in payload


def test_openapi_contract_flags_missing_required_and_auth_headers():
    schema = {
        "openapi": "3.1.0",
        "info": {"title": "FluxMind API"},
        "paths": {
            "/health": {
                "get": {
                    "summary": "Health",
                    "operationId": "health",
                    "responses": {"200": {"description": "OK"}},
                }
            },
            "/admin/status": {
                "get": {
                    "summary": "Status",
                    "operationId": "status",
                    "responses": {"200": {"description": "OK"}},
                    "parameters": [],
                }
            },
        },
    }

    status = collect_openapi_contract(schema)

    assert status["local_contract_ready"] is False
    assert "required_operations_missing" in status["blockers"]
    assert "protected_auth_headers_missing" in status["blockers"]
    assert "GET /admin/status" in status["protected_operations_missing_auth_headers"]
    assert "GET /ready" in status["missing_required_operations"]


def test_format_openapi_contract_markdown_is_no_secret():
    status = collect_openapi_contract(
        api.app.openapi(),
        generated_at="2026-06-20T00:00:00+00:00",
    )

    markdown = format_openapi_contract_markdown(status)

    assert "# FluxMind OpenAPI Contract" in markdown
    assert "Local contract ready: true" in markdown
    assert "Raw schema exported: false" in markdown
    assert "Operation fingerprint:" in markdown
    assert "admin: present=true" in markdown
    assert "Contract: none" in markdown
    assert "components" not in markdown
    assert "hunter2" not in markdown


def test_openapi_contract_snapshot_verify_detects_no_secret_drift():
    current = collect_openapi_contract(
        api.app.openapi(),
        generated_at="2026-06-20T00:00:00+00:00",
    )
    unchanged = verify_openapi_contract_snapshot(
        current,
        dict(current),
        generated_at="2026-06-20T00:00:01+00:00",
    )
    drifted_snapshot = dict(current)
    drifted_snapshot["operation_fingerprint"] = "0" * 64
    drifted = verify_openapi_contract_snapshot(
        current,
        drifted_snapshot,
        generated_at="2026-06-20T00:00:02+00:00",
    )

    assert unchanged["mode"] == "openapi_contract_snapshot_verify"
    assert unchanged["ok"] is True
    assert unchanged["diff_count"] == 0
    assert unchanged["raw_schema_exported"] is False
    assert drifted["ok"] is False
    assert drifted["diff_count"] == 1
    assert "snapshot_contract_drift" in drifted["blockers"]
    assert drifted["diffs"][0]["field"] == "operation_fingerprint"

    markdown = format_openapi_contract_snapshot_verify_markdown(drifted)
    assert "# FluxMind OpenAPI Contract Snapshot Verify" in markdown
    assert "Snapshot verify: snapshot_contract_drift" in markdown
    assert "components" not in markdown
    assert "hunter2" not in markdown


def test_openapi_contract_snapshot_verify_sanitizes_untrusted_snapshot_values():
    current = collect_openapi_contract(
        api.app.openapi(),
        generated_at="2026-06-20T00:00:00+00:00",
    )
    snapshot = dict(current)
    snapshot["paths"] = {"/private/hunter2": {"get": {}}}
    snapshot["components"] = {"schemas": {"SecretThing": "hunter2"}}
    snapshot["operation_fingerprint"] = "hunter2-secret-fingerprint"
    snapshot["route_count"] = {"raw_path": "/private/hunter2"}
    snapshot["operation_count"] = -1
    snapshot["required_operation_count"] = "999999"
    snapshot["protected_operation_count"] = 1000000000000
    snapshot["local_contract_ready"] = "hunter2"

    status = verify_openapi_contract_snapshot(
        current,
        snapshot,
        generated_at="2026-06-20T00:00:03+00:00",
    )
    payload = json.dumps(status, ensure_ascii=False, sort_keys=True)
    markdown = format_openapi_contract_snapshot_verify_markdown(status)

    assert status["ok"] is False
    assert status["snapshot_raw_schema_included"] is True
    assert status["snapshot_shape_valid"] is False
    assert "snapshot_raw_schema_included" in status["blockers"]
    assert "snapshot_contract_shape_invalid" in status["blockers"]
    assert status["snapshot_operation_fingerprint"] == ""
    assert {
        "operation_fingerprint",
        "route_count",
        "operation_count",
        "required_operation_count",
        "protected_operation_count",
        "local_contract_ready",
    }.issubset({diff["field"] for diff in status["diffs"]})
    assert any(
        diff["field"] == "operation_fingerprint" and not diff["snapshot_valid"]
        for diff in status["diffs"]
    )
    for rendered in (payload, markdown):
        assert "hunter2" not in rendered
        assert "/private" not in rendered
        assert "SecretThing" not in rendered
        assert "components" not in rendered
        assert "999999" not in rendered
        assert "1000000000000" not in rendered
