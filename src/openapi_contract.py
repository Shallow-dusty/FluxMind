"""No-secret OpenAPI contract readiness for FluxMind API surfaces."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any


OPENAPI_CONTRACT_SCHEMA_VERSION = 1
REQUIRED_PATH_METHODS: dict[str, tuple[str, ...]] = {
    "/health": ("get",),
    "/ready": ("get",),
    "/query": ("post",),
    "/query/inspect": ("post",),
    "/query/retrieve": ("post",),
    "/query/report": ("post",),
    "/jobs": ("get",),
    "/jobs/{job_id}": ("get",),
    "/jobs/async/index/rebuild": ("post",),
    "/jobs/code/python-local": ("post",),
    "/jobs/code/octave-local": ("post",),
    "/jobs/image/mock": ("post",),
    "/artifacts": ("get",),
    "/artifacts/{artifact_id}": ("get",),
    "/corpus/papers": ("get",),
    "/corpus/status": ("get",),
    "/corpus/active": ("put",),
    "/corpus/profiles": ("get", "post"),
    "/corpus/structure": ("get",),
    "/corpus/structure/report": ("get",),
    "/admin/status": ("get",),
    "/admin/status/report": ("get",),
    "/admin/metrics": ("get",),
    "/admin/events": ("get",),
    "/admin/runtime-manifest": ("get",),
    "/admin/retention": ("get",),
    "/admin/quality-readiness": ("get", "post"),
    "/admin/quality-readiness/report": ("get", "post"),
    "/admin/activation-suite": ("get", "post"),
    "/admin/activation-suite/report": ("get", "post"),
    "/admin/openapi-contract": ("get",),
    "/admin/openapi-contract/report": ("get",),
    "/admin/openapi-contract/verify": ("post",),
    "/admin/openapi-contract/verify/report": ("post",),
    "/admin/product-activation-rehearsal": ("get",),
    "/admin/collaboration-readiness": ("get",),
    "/admin/collaboration-readiness/report": ("get",),
    "/admin/provider-runtime-rehearsal": ("get",),
    "/admin/platform-migration-rehearsal": ("get",),
    "/admin/product-registry/status": ("get",),
    "/admin/product-registry/workspaces": ("get", "post"),
    "/admin/share-links/status": ("get",),
    "/admin/share-links": ("get", "post"),
    "/admin/share-links/{link_id}/revoke": ("post",),
    "/admin/share-links/resolve": ("post",),
}
PROTECTED_PATH_PREFIXES = (
    "/admin",
    "/artifacts",
    "/corpus",
    "/jobs",
    "/query",
)
ROUTE_GROUP_PREFIXES = {
    "admin": "/admin",
    "artifacts": "/artifacts",
    "corpus": "/corpus",
    "jobs": "/jobs",
    "query": "/query",
    "health": "/health",
    "ready": "/ready",
}
OPENAPI_METHODS = {"get", "post", "put", "patch", "delete", "options", "head"}
SNAPSHOT_COMPARE_FIELDS = (
    "operation_fingerprint",
    "route_count",
    "operation_count",
    "required_operation_count",
    "required_operation_missing_count",
    "protected_operation_count",
    "protected_auth_header_operation_count",
    "undocumented_operation_count",
    "response_missing_operation_count",
    "local_contract_ready",
)
FINGERPRINT_RE = re.compile(r"^[0-9a-f]{64}$")
MAX_SNAPSHOT_COUNT = 1_000_000


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _format_bool(value: Any) -> str:
    return "true" if bool(value) else "false"


def _schema_paths(schema: dict[str, Any]) -> dict[str, dict[str, Any]]:
    paths = schema.get("paths", {})
    return paths if isinstance(paths, dict) else {}


def _path_methods(path_item: Any) -> set[str]:
    if not isinstance(path_item, dict):
        return set()
    return {
        str(method).lower()
        for method in path_item
        if str(method).lower() in OPENAPI_METHODS
    }


def _operations(schema: dict[str, Any]) -> list[dict[str, Any]]:
    operations: list[dict[str, Any]] = []
    for path, path_item in sorted(_schema_paths(schema).items()):
        if not isinstance(path_item, dict):
            continue
        for method, operation in sorted(path_item.items()):
            method_name = str(method).lower()
            if method_name not in OPENAPI_METHODS or not isinstance(operation, dict):
                continue
            operations.append(
                {
                    "path": str(path),
                    "method": method_name.upper(),
                    "operation": operation,
                }
            )
    return operations


def _header_names(operation: dict[str, Any]) -> set[str]:
    headers: set[str] = set()
    for parameter in operation.get("parameters", []) or []:
        if not isinstance(parameter, dict):
            continue
        if parameter.get("in") == "header":
            headers.add(str(parameter.get("name", "")).lower())
    return headers


def _missing_required_operations(schema: dict[str, Any]) -> list[str]:
    paths = _schema_paths(schema)
    missing: list[str] = []
    for path, methods in REQUIRED_PATH_METHODS.items():
        present_methods = _path_methods(paths.get(path, {}))
        for method in methods:
            if method not in present_methods:
                missing.append(f"{method.upper()} {path}")
    return sorted(missing)


def _undocumented_operations(operations: list[dict[str, Any]]) -> list[str]:
    missing: list[str] = []
    for item in operations:
        operation = item["operation"]
        if not operation.get("summary") or not operation.get("operationId"):
            missing.append(f"{item['method']} {item['path']}")
    return sorted(missing)


def _response_missing_operations(operations: list[dict[str, Any]]) -> list[str]:
    missing: list[str] = []
    for item in operations:
        responses = item["operation"].get("responses", {})
        if not isinstance(responses, dict) or not responses:
            missing.append(f"{item['method']} {item['path']}")
    return sorted(missing)


def _protected_operations_without_auth_headers(
    operations: list[dict[str, Any]],
) -> list[str]:
    missing: list[str] = []
    for item in operations:
        path = item["path"]
        if not path.startswith(PROTECTED_PATH_PREFIXES):
            continue
        headers = _header_names(item["operation"])
        if not {"authorization", "x-api-key"}.issubset(headers):
            missing.append(f"{item['method']} {path}")
    return sorted(missing)


def _route_group_summary(paths: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for group, prefix in ROUTE_GROUP_PREFIXES.items():
        group_paths = [path for path in paths if path == prefix or path.startswith(prefix + "/")]
        operation_count = sum(len(_path_methods(paths[path])) for path in group_paths)
        summary[group] = {
            "path_count": len(group_paths),
            "operation_count": operation_count,
            "present": bool(group_paths),
        }
    return summary


def _operation_parameter_signature(operation: dict[str, Any]) -> list[str]:
    parameters: list[str] = []
    for parameter in operation.get("parameters", []) or []:
        if not isinstance(parameter, dict):
            continue
        location = str(parameter.get("in", ""))
        name = str(parameter.get("name", ""))
        required = "required" if bool(parameter.get("required")) else "optional"
        if location and name:
            parameters.append(f"{location}:{name}:{required}")
    return sorted(parameters)


def _operation_response_codes(operation: dict[str, Any]) -> list[str]:
    responses = operation.get("responses", {})
    if not isinstance(responses, dict):
        return []
    return sorted(str(code) for code in responses)


def _operation_fingerprint(operations: list[dict[str, Any]]) -> str:
    records = [
        {
            "method": item["method"],
            "path": item["path"],
            "operation_id": str(item["operation"].get("operationId", "")),
            "summary_present": bool(item["operation"].get("summary")),
            "parameters": _operation_parameter_signature(item["operation"]),
            "response_codes": _operation_response_codes(item["operation"]),
        }
        for item in operations
    ]
    payload = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def collect_openapi_contract(
    schema: dict[str, Any],
    *,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Collect an OpenAPI contract summary without exporting the raw schema."""
    paths = _schema_paths(schema)
    operations = _operations(schema)
    missing_required = _missing_required_operations(schema)
    undocumented = _undocumented_operations(operations)
    missing_responses = _response_missing_operations(operations)
    missing_auth_headers = _protected_operations_without_auth_headers(operations)
    openapi_version = str(schema.get("openapi", ""))
    info = schema.get("info", {}) if isinstance(schema.get("info"), dict) else {}
    title = str(info.get("title", ""))

    blockers: list[str] = []
    if not openapi_version.startswith("3."):
        blockers.append("openapi_version_not_3")
    if "FluxMind" not in title:
        blockers.append("openapi_info_title_missing")
    if missing_required:
        blockers.append("required_operations_missing")
    if undocumented:
        blockers.append("operation_documentation_missing")
    if missing_responses:
        blockers.append("operation_responses_missing")
    if missing_auth_headers:
        blockers.append("protected_auth_headers_missing")

    protected_operations = [
        item
        for item in operations
        if item["path"].startswith(PROTECTED_PATH_PREFIXES)
    ]
    auth_header_operations = [
        item
        for item in protected_operations
        if {"authorization", "x-api-key"}.issubset(_header_names(item["operation"]))
    ]
    required_operation_count = sum(len(methods) for methods in REQUIRED_PATH_METHODS.values())
    local_contract_ready = not blockers

    return {
        "mode": "openapi_contract",
        "schema_version": OPENAPI_CONTRACT_SCHEMA_VERSION,
        "generated_at": generated_at or _utc_now(),
        "local_contract_ready": local_contract_ready,
        "openapi_version": openapi_version,
        "info_title_present": "FluxMind" in title,
        "required_operation_count": required_operation_count,
        "required_operation_missing_count": len(missing_required),
        "route_count": len(paths),
        "operation_count": len(operations),
        "operation_fingerprint": _operation_fingerprint(operations),
        "protected_operation_count": len(protected_operations),
        "protected_auth_header_operation_count": len(auth_header_operations),
        "undocumented_operation_count": len(undocumented),
        "response_missing_operation_count": len(missing_responses),
        "component_schema_count": len(
            (schema.get("components", {}) or {}).get("schemas", {}) or {}
        )
        if isinstance(schema.get("components", {}), dict)
        else 0,
        "route_groups": _route_group_summary(paths),
        "blockers": blockers,
        "missing_required_operations": missing_required,
        "undocumented_operations": undocumented,
        "missing_response_operations": missing_responses,
        "protected_operations_missing_auth_headers": missing_auth_headers,
        "raw_schema_exported": False,
        "request_examples_exported": False,
        "content_exported": False,
        "secrets_exported": False,
        "paths_exported": False,
        "notes": [
            "The contract checks route/method coverage, operation documentation, responses, and auth header declarations.",
            "The operation fingerprint is computed from method/path, operation IDs, parameter names, and response codes.",
            "The raw OpenAPI schema is intentionally not embedded in this readiness report.",
        ],
    }


def _snapshot_diffs(
    current: dict[str, Any],
    snapshot: dict[str, Any],
) -> list[dict[str, Any]]:
    diffs: list[dict[str, Any]] = []
    for field in SNAPSHOT_COMPARE_FIELDS:
        current_value, current_valid = _safe_snapshot_compare_value(field, current)
        snapshot_value, snapshot_valid = _safe_snapshot_compare_value(field, snapshot)
        if current_value != snapshot_value:
            diffs.append(
                {
                    "field": field,
                    "snapshot": snapshot_value,
                    "snapshot_valid": snapshot_valid,
                    "current": current_value,
                    "current_valid": current_valid,
                }
            )
    return diffs


def _safe_snapshot_compare_value(field: str, payload: dict[str, Any]) -> tuple[Any, bool]:
    value = payload.get(field)
    if field == "operation_fingerprint":
        if isinstance(value, str) and FINGERPRINT_RE.fullmatch(value):
            return value, True
        return None, False
    if field == "local_contract_ready":
        if isinstance(value, bool):
            return value, True
        return None, False
    if field in SNAPSHOT_COMPARE_FIELDS:
        if (
            isinstance(value, int)
            and not isinstance(value, bool)
            and 0 <= value <= MAX_SNAPSHOT_COUNT
        ):
            return value, True
        return None, False
    return None, False


def _snapshot_has_raw_schema(payload: dict[str, Any]) -> bool:
    return any(key in payload for key in ("paths", "components", "webhooks"))


def _snapshot_shape_invalid(payload: dict[str, Any]) -> bool:
    return any(
        not _safe_snapshot_compare_value(field, payload)[1]
        for field in SNAPSHOT_COMPARE_FIELDS
    )


def verify_openapi_contract_snapshot(
    current: dict[str, Any],
    snapshot: dict[str, Any],
    *,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Compare the current no-secret contract report with a prior report."""
    blockers: list[str] = []
    if snapshot.get("mode") != "openapi_contract":
        blockers.append("snapshot_mode_invalid")
    if not current.get("local_contract_ready"):
        blockers.append("current_contract_not_ready")
    if _snapshot_has_raw_schema(snapshot):
        blockers.append("snapshot_raw_schema_included")
    if _snapshot_shape_invalid(snapshot):
        blockers.append("snapshot_contract_shape_invalid")
    diffs = _snapshot_diffs(current, snapshot)
    if diffs:
        blockers.append("snapshot_contract_drift")

    return {
        "mode": "openapi_contract_snapshot_verify",
        "schema_version": OPENAPI_CONTRACT_SCHEMA_VERSION,
        "generated_at": generated_at or _utc_now(),
        "ok": not blockers,
        "current_local_contract_ready": bool(current.get("local_contract_ready")),
        "snapshot_local_contract_ready": bool(
            _safe_snapshot_compare_value("local_contract_ready", snapshot)[0]
        ),
        "current_operation_fingerprint": str(
            _safe_snapshot_compare_value("operation_fingerprint", current)[0] or ""
        ),
        "snapshot_operation_fingerprint": str(
            _safe_snapshot_compare_value("operation_fingerprint", snapshot)[0] or ""
        ),
        "current_route_count": int(current.get("route_count", 0) or 0),
        "snapshot_route_count": int(
            _safe_snapshot_compare_value("route_count", snapshot)[0] or 0
        ),
        "current_operation_count": int(current.get("operation_count", 0) or 0),
        "snapshot_operation_count": int(
            _safe_snapshot_compare_value("operation_count", snapshot)[0] or 0
        ),
        "compared_field_count": len(SNAPSHOT_COMPARE_FIELDS),
        "diff_count": len(diffs),
        "diffs": diffs,
        "blockers": sorted(set(blockers)),
        "raw_schema_exported": False,
        "snapshot_raw_schema_included": _snapshot_has_raw_schema(snapshot),
        "snapshot_shape_valid": not _snapshot_shape_invalid(snapshot),
        "request_examples_exported": False,
        "content_exported": False,
        "secrets_exported": False,
        "paths_exported": False,
        "notes": [
            "Snapshot verification compares only no-secret OpenAPI contract report fields.",
            "It does not require or export the raw OpenAPI schema.",
        ],
    }


def format_openapi_contract_markdown(status: dict[str, Any]) -> str:
    """Render OpenAPI contract readiness as no-secret Markdown."""
    lines = [
        "# FluxMind OpenAPI Contract",
        "",
        f"- Mode: {status.get('mode', '')}",
        f"- Schema version: {status.get('schema_version', '')}",
        f"- Generated at: {status.get('generated_at', '')}",
        f"- Local contract ready: {_format_bool(status.get('local_contract_ready', False))}",
        f"- OpenAPI version: {status.get('openapi_version', '')}",
        f"- Info title present: {_format_bool(status.get('info_title_present', False))}",
        f"- Route count: {int(status.get('route_count', 0) or 0)}",
        f"- Operation count: {int(status.get('operation_count', 0) or 0)}",
        f"- Operation fingerprint: {status.get('operation_fingerprint', '')}",
        f"- Required operations: {int(status.get('required_operation_count', 0) or 0)}",
        f"- Missing required operations: {int(status.get('required_operation_missing_count', 0) or 0)}",
        f"- Protected operations: {int(status.get('protected_operation_count', 0) or 0)}",
        f"- Protected operations with auth headers: {int(status.get('protected_auth_header_operation_count', 0) or 0)}",
        f"- Undocumented operations: {int(status.get('undocumented_operation_count', 0) or 0)}",
        f"- Operations missing responses: {int(status.get('response_missing_operation_count', 0) or 0)}",
        f"- Component schemas: {int(status.get('component_schema_count', 0) or 0)}",
        f"- Raw schema exported: {_format_bool(status.get('raw_schema_exported', False))}",
        f"- Request examples exported: {_format_bool(status.get('request_examples_exported', False))}",
        f"- Content exported: {_format_bool(status.get('content_exported', False))}",
        f"- Secrets exported: {_format_bool(status.get('secrets_exported', False))}",
        f"- Paths exported: {_format_bool(status.get('paths_exported', False))}",
        "",
        "## Route Groups",
        "",
    ]
    for group, summary in sorted((status.get("route_groups", {}) or {}).items()):
        lines.append(
            f"- {group}: present={_format_bool(summary.get('present', False))}; "
            f"paths={int(summary.get('path_count', 0) or 0)}; "
            f"operations={int(summary.get('operation_count', 0) or 0)}"
        )

    lines.extend(["", "## Blockers", ""])
    lines.append(f"- Contract: {', '.join(status.get('blockers', [])) or 'none'}")
    for label, key in (
        ("Missing required operations", "missing_required_operations"),
        ("Undocumented operations", "undocumented_operations"),
        ("Missing response operations", "missing_response_operations"),
        (
            "Protected operations missing auth headers",
            "protected_operations_missing_auth_headers",
        ),
    ):
        values = status.get(key, []) or []
        lines.append(f"- {label}: {', '.join(values) or 'none'}")
    return "\n".join(lines)


def format_openapi_contract_snapshot_verify_markdown(status: dict[str, Any]) -> str:
    """Render OpenAPI contract snapshot verification as no-secret Markdown."""
    lines = [
        "# FluxMind OpenAPI Contract Snapshot Verify",
        "",
        f"- Mode: {status.get('mode', '')}",
        f"- Schema version: {status.get('schema_version', '')}",
        f"- Generated at: {status.get('generated_at', '')}",
        f"- OK: {_format_bool(status.get('ok', False))}",
        f"- Current local contract ready: {_format_bool(status.get('current_local_contract_ready', False))}",
        f"- Snapshot local contract ready: {_format_bool(status.get('snapshot_local_contract_ready', False))}",
        f"- Current route count: {int(status.get('current_route_count', 0) or 0)}",
        f"- Snapshot route count: {int(status.get('snapshot_route_count', 0) or 0)}",
        f"- Current operation count: {int(status.get('current_operation_count', 0) or 0)}",
        f"- Snapshot operation count: {int(status.get('snapshot_operation_count', 0) or 0)}",
        f"- Current operation fingerprint: {status.get('current_operation_fingerprint', '')}",
        f"- Snapshot operation fingerprint: {status.get('snapshot_operation_fingerprint', '')}",
        f"- Compared fields: {int(status.get('compared_field_count', 0) or 0)}",
        f"- Diff count: {int(status.get('diff_count', 0) or 0)}",
        f"- Raw schema exported: {_format_bool(status.get('raw_schema_exported', False))}",
        f"- Snapshot raw schema included: {_format_bool(status.get('snapshot_raw_schema_included', False))}",
        f"- Request examples exported: {_format_bool(status.get('request_examples_exported', False))}",
        f"- Content exported: {_format_bool(status.get('content_exported', False))}",
        f"- Secrets exported: {_format_bool(status.get('secrets_exported', False))}",
        f"- Paths exported: {_format_bool(status.get('paths_exported', False))}",
        "",
        "## Blockers",
        "",
        f"- Snapshot verify: {', '.join(status.get('blockers', [])) or 'none'}",
        "",
        "## Diffs",
        "",
    ]
    diffs = status.get("diffs", []) or []
    if not diffs:
        lines.append("- none")
    for diff in diffs:
        lines.append(
            f"- {diff.get('field', '')}: snapshot={diff.get('snapshot')} "
            f"current={diff.get('current')}"
        )
    return "\n".join(lines)
