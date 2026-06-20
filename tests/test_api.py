import json

import pytest
from pathlib import Path
from fastapi import HTTPException
from fastapi.testclient import TestClient

import api
from src.api_keys import LocalApiKeyRegistry
from src.product_registry import LocalProductRegistry
from src.jobs import AsyncJobManager
from src.metadata import PaperRecord, safe_corpus_profile_report_filename
from src.artifacts import artifact_id_for_uri
from src.runtime import ProviderQuotaGuardError, RuntimeEvent


@pytest.fixture(autouse=True)
def no_job_runtime_event_disk_writes(monkeypatch):
    monkeypatch.setattr("src.jobs.append_runtime_event", lambda **_kwargs: None)
    monkeypatch.setattr(api, "API_ACCESS_AUDIT_ENABLED", False)
    monkeypatch.setattr(api, "API_RATE_LIMIT_ENABLED", False)
    monkeypatch.setattr(api, "IDENTITY_QUOTAS_BILLING_ENABLED", False)
    monkeypatch.setattr(api, "PRODUCT_QUOTA_GUARD_ENABLED", False)
    monkeypatch.setattr(api, "PRODUCT_RBAC_GUARD_ENABLED", False)
    api._API_RATE_LIMIT_BUCKETS.clear()


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


def test_api_token_status_does_not_return_token_values(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "secret")

    status = api.api_token_status("Bearer secret", "wrong")

    assert status == {
        "token_status": "valid",
        "credential_type": "multiple",
        "credential_present": True,
        "auth_configured": True,
        "auth_source": "static_token",
        "api_key_registry_configured": False,
    }
    assert "secret" not in str(status)
    assert "wrong" not in str(status)


def test_api_token_status_uses_constant_time_comparison(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "secret")
    calls = []

    def fake_compare_digest(candidate, expected):
        calls.append((candidate, expected))
        return candidate == expected

    monkeypatch.setattr(api.hmac, "compare_digest", fake_compare_digest)

    status = api.api_token_status("Bearer secret", "wrong")

    assert status["token_status"] == "valid"
    assert calls == [("wrong", "secret"), ("secret", "secret")]


def test_verify_api_token_accepts_configured_registry_token(tmp_path, monkeypatch):
    registry = LocalApiKeyRegistry(tmp_path / "api_keys.sqlite3")
    token = registry.create_key(owner_id="lab-api")["token"]
    monkeypatch.setattr(api, "API_TOKEN", "")
    monkeypatch.setattr("src.api_keys.config.API_KEY_REGISTRY_BACKEND", "sqlite")
    monkeypatch.setattr("src.api_keys.config.API_KEY_REGISTRY_FILE", tmp_path / "api_keys.sqlite3")

    api.verify_api_token(None, token)

    status = api.api_token_status(None, token)
    assert status["token_status"] == "valid"
    assert status["auth_configured"] is True
    assert status["auth_source"] == "api_key_registry"
    assert status["api_key_registry_configured"] is True
    assert token not in str(status)


def test_api_access_middleware_records_valid_auth_without_secrets(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "secret")
    monkeypatch.setattr(api, "API_ACCESS_AUDIT_ENABLED", True)
    events = []
    monkeypatch.setattr(api, "append_runtime_event", lambda **kwargs: events.append(kwargs))

    client = TestClient(api.app)
    response = client.get(
        "/health",
        headers={"X-API-Key": "secret", "X-Request-ID": "req-auth"},
    )

    assert response.status_code == 200
    assert events[-1]["kind"] == "api_access"
    assert events[-1]["code"] == "auth_valid"
    assert events[-1]["request_id"] == "req-auth"
    assert events[-1]["metadata"]["method"] == "GET"
    assert events[-1]["metadata"]["route_present"] is True
    assert events[-1]["metadata"]["route_fingerprint"]
    assert "path" not in events[-1]["metadata"]
    assert "/health" not in str(events[-1])
    assert events[-1]["metadata"]["status_code"] == 200
    assert events[-1]["metadata"]["token_status"] == "valid"
    assert events[-1]["metadata"]["credential_type"] == "x_api_key"
    assert "secret" not in str(events[-1])


def test_api_access_middleware_ignores_blank_request_id(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")
    monkeypatch.setattr(api, "API_ACCESS_AUDIT_ENABLED", True)
    events = []
    monkeypatch.setattr(api, "append_runtime_event", lambda **kwargs: events.append(kwargs))

    client = TestClient(api.app)
    response = client.get("/health", headers={"X-Request-ID": "   "})

    assert response.status_code == 200
    assert events[-1]["kind"] == "api_access"
    assert events[-1]["request_id"] is None


def test_api_access_middleware_ignores_unsafe_request_id(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")
    monkeypatch.setattr(api, "API_ACCESS_AUDIT_ENABLED", True)
    events = []
    monkeypatch.setattr(api, "append_runtime_event", lambda **kwargs: events.append(kwargs))

    client = TestClient(api.app)
    response = client.get("/health", headers={"X-Request-ID": "Bearer secret/token"})

    assert response.status_code == 200
    assert events[-1]["kind"] == "api_access"
    assert events[-1]["request_id"] is None
    assert "secret/token" not in str(events[-1])


def test_api_access_middleware_records_invalid_auth_without_secrets(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "secret")
    monkeypatch.setattr(api, "API_ACCESS_AUDIT_ENABLED", True)
    events = []
    monkeypatch.setattr(api, "append_runtime_event", lambda **kwargs: events.append(kwargs))

    client = TestClient(api.app)
    response = client.get(
        "/admin/status",
        headers={"Authorization": "Bearer wrong", "X-Request-ID": "req-bad"},
    )

    assert response.status_code == 401
    assert events[-1]["kind"] == "api_access"
    assert events[-1]["code"] == "auth_invalid"
    assert events[-1]["request_id"] == "req-bad"
    assert events[-1]["metadata"]["route_present"] is True
    assert events[-1]["metadata"]["route_fingerprint"]
    assert "path" not in events[-1]["metadata"]
    assert "/admin/status" not in str(events[-1])
    assert events[-1]["metadata"]["status_code"] == 401
    assert events[-1]["metadata"]["token_status"] == "invalid"
    assert events[-1]["metadata"]["credential_present"] is True
    assert "wrong" not in str(events[-1])


def test_api_rate_limit_middleware_blocks_after_configured_threshold(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")
    monkeypatch.setattr(api, "API_ACCESS_AUDIT_ENABLED", True)
    monkeypatch.setattr(api, "API_RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr(api, "API_RATE_LIMIT_MAX_REQUESTS", 2)
    monkeypatch.setattr(api, "API_RATE_LIMIT_WINDOW_S", 60)
    api._API_RATE_LIMIT_BUCKETS.clear()
    events = []
    monkeypatch.setattr(api, "append_runtime_event", lambda **kwargs: events.append(kwargs))

    client = TestClient(api.app)
    first = client.get("/health", headers={"X-Request-ID": "req-rate-1"})
    second = client.get("/health", headers={"X-Request-ID": "req-rate-2"})
    third = client.get("/health", headers={"X-Request-ID": "req-rate-3"})

    assert first.status_code == 200
    assert second.status_code == 200
    assert third.status_code == 429
    assert third.json() == {"detail": "API rate limit exceeded"}
    assert third.headers["X-RateLimit-Limit"] == "2"
    assert third.headers["X-RateLimit-Remaining"] == "0"
    assert events[-1]["kind"] == "api_access"
    assert events[-1]["request_id"] == "req-rate-3"
    assert events[-1]["metadata"]["status_code"] == 429
    assert events[-1]["metadata"]["rate_limit_enabled"] is True
    assert events[-1]["metadata"]["rate_limited"] is True
    assert events[-1]["metadata"]["rate_limit"] == 2
    assert "127.0.0.1" not in str(events[-1])


def test_startup_warmup_skips_missing_faiss_index(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "FAISS_INDEX_DIR", tmp_path / "faiss_index")

    def fail_if_called():
        raise AssertionError("missing index should not trigger vector-store load")

    monkeypatch.setattr(api, "get_vector_store", fail_if_called)

    assert api.warm_existing_vector_store() is False


def test_startup_warmup_loads_existing_faiss_index(tmp_path, monkeypatch):
    index_dir = tmp_path / "faiss_index"
    index_dir.mkdir()
    (index_dir / "index.faiss").write_text("index", encoding="utf-8")
    called = {"count": 0}

    def fake_load():
        called["count"] += 1
        return object()

    monkeypatch.setattr(api, "FAISS_INDEX_DIR", index_dir)
    monkeypatch.setattr(api, "get_vector_store", fake_load)

    assert api.warm_existing_vector_store() is True
    assert called["count"] == 1


def test_startup_warmup_does_not_abort_when_index_load_fails(tmp_path, monkeypatch):
    index_dir = tmp_path / "faiss_index"
    index_dir.mkdir()
    (index_dir / "index.faiss").write_text("index", encoding="utf-8")

    def fail_load():
        raise RuntimeError("corrupt index")

    monkeypatch.setattr(api, "FAISS_INDEX_DIR", index_dir)
    monkeypatch.setattr(api, "get_vector_store", fail_load)

    assert api.warm_existing_vector_store() is False


def test_startup_warmup_can_run_in_background(monkeypatch):
    started = {}

    class FakeThread:
        def __init__(self, *, target, name, daemon):
            started["target"] = target
            started["name"] = name
            started["daemon"] = daemon

        def start(self):
            started["started"] = True

    monkeypatch.setattr(api.threading, "Thread", FakeThread)

    api.start_background_vector_store_warmup()

    assert started == {
        "target": api.warm_existing_vector_store,
        "name": "fluxmind-vector-store-warmup",
        "daemon": True,
        "started": True,
    }
    assert api.startup_warmup_status()["status"] == "warming"


def test_ready_endpoint_reports_background_warmup_state(monkeypatch):
    api._set_startup_warmup_state("ready", ready=True)
    ready_response = api.ready()
    assert ready_response["status"] == "ready"

    api._set_startup_warmup_state("warming", ready=False)
    with pytest.raises(HTTPException) as exc:
        api.ready()
    assert exc.value.status_code == 503
    assert exc.value.detail["status"] == "warming"


def test_query_response_includes_request_id(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")
    usage_events = []

    class FakeResult:
        answer = "explanation: Explain SMC"
        provider_usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

        class FakeValidation:
            ok = True

        citation_validation = FakeValidation()

    def fake_query_with_metadata(question, *, answer_mode):
        assert question == "Explain SMC"
        assert answer_mode == "explanation"
        return FakeResult()

    monkeypatch.setattr(api, "query_with_metadata", fake_query_with_metadata)
    monkeypatch.setattr(api, "append_runtime_event", lambda **kwargs: usage_events.append(kwargs))

    client = TestClient(api.app)
    response = client.post(
        "/query",
        json={"question": "Explain SMC", "owner_id": "lab-query", "owner_label": "Query Lab"},
        headers={"X-Request-ID": "req-test"},
    )

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "req-test"
    assert response.json() == {"answer": "explanation: Explain SMC", "request_id": "req-test"}
    assert usage_events[0]["kind"] == "query_usage"
    assert usage_events[0]["request_id"] == "req-test"
    assert usage_events[0]["metadata"]["endpoint"] == "/query"
    assert usage_events[0]["metadata"]["estimated_total_tokens"] > 0
    assert usage_events[0]["metadata"]["usage_source"] == "provider"
    assert usage_events[0]["metadata"]["cost_source"] == "not_configured"
    assert usage_events[0]["metadata"]["provider_total_tokens"] == 15
    assert usage_events[0]["metadata"]["duration_ms"] >= 0
    assert usage_events[0]["metadata"]["owner_id_present"] is True
    assert usage_events[0]["metadata"]["owner_label_present"] is True
    assert usage_events[0]["metadata"]["ownership_source"] == "request"
    assert "owner_id" not in usage_events[0]["metadata"]
    assert "owner_label" not in usage_events[0]["metadata"]
    assert "Explain SMC" not in str(usage_events[0]["metadata"])
    assert "lab-query" not in str(usage_events[0]["metadata"])
    assert "Query Lab" not in str(usage_events[0]["metadata"])


def test_query_blank_request_id_header_generates_request_id(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")
    monkeypatch.setattr(api, "new_request_id", lambda: "generated-req")
    usage_events = []

    class FakeResult:
        answer = "ok"
        provider_usage = None

        class FakeValidation:
            ok = True

        citation_validation = FakeValidation()

    monkeypatch.setattr(api, "query_with_metadata", lambda question, *, answer_mode: FakeResult())
    monkeypatch.setattr(api, "append_runtime_event", lambda **kwargs: usage_events.append(kwargs))

    client = TestClient(api.app)
    response = client.post(
        "/query",
        json={"question": "Explain SMC"},
        headers={"X-Request-ID": "   "},
    )

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "generated-req"
    assert response.json()["request_id"] == "generated-req"
    assert usage_events[0]["request_id"] == "generated-req"


def test_query_unsafe_request_id_header_generates_request_id(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")
    monkeypatch.setattr(api, "new_request_id", lambda: "generated-req")
    usage_events = []

    class FakeResult:
        answer = "ok"
        provider_usage = None

        class FakeValidation:
            ok = True

        citation_validation = FakeValidation()

    monkeypatch.setattr(api, "query_with_metadata", lambda question, *, answer_mode: FakeResult())
    monkeypatch.setattr(api, "append_runtime_event", lambda **kwargs: usage_events.append(kwargs))

    client = TestClient(api.app)
    response = client.post(
        "/query",
        json={"question": "Explain SMC"},
        headers={"X-Request-ID": "Bearer secret/token"},
    )

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "generated-req"
    assert response.json()["request_id"] == "generated-req"
    assert usage_events[0]["request_id"] == "generated-req"


def test_query_accepts_answer_mode(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")
    seen = {}

    class FakeResult:
        answer = "ok"
        provider_usage = None

    def fake_query_with_metadata(question, *, answer_mode):
        seen["question"] = question
        seen["answer_mode"] = answer_mode
        return FakeResult()

    monkeypatch.setattr(api, "query_with_metadata", fake_query_with_metadata)

    client = TestClient(api.app)
    response = client.post(
        "/query",
        json={"question": "Derive SMC reaching law", "answer_mode": "derivation"},
    )

    assert response.status_code == 200
    assert seen == {"question": "Derive SMC reaching law", "answer_mode": "derivation"}


def test_query_product_quota_guard_records_usage_and_blocks_over_limit(tmp_path, monkeypatch):
    api_key_path = tmp_path / "api_keys.sqlite3"
    product_registry_path = tmp_path / "product_registry.sqlite3"
    token = LocalApiKeyRegistry(api_key_path).create_key(owner_id="local-user")["token"]
    product_registry = LocalProductRegistry(product_registry_path)
    workspace = product_registry.create_workspace(
        workspace_id="local-workspace",
        owner_user_id="local-user",
    )
    product_registry.set_quota(
        workspace_id=workspace.workspace_id,
        metric="requests",
        limit_value=1,
        window_s=3600,
    )
    monkeypatch.setattr(api, "API_TOKEN", "")
    monkeypatch.setattr(api, "IDENTITY_QUOTAS_BILLING_ENABLED", True)
    monkeypatch.setattr(api, "PRODUCT_QUOTA_GUARD_ENABLED", True)
    monkeypatch.setattr(api, "PRODUCT_REGISTRY_BACKEND", "sqlite")
    monkeypatch.setattr(api, "QUOTA_STORE_BACKEND", "sqlite")
    monkeypatch.setattr(api, "PRODUCT_QUOTA_METRIC", "requests")
    monkeypatch.setattr("src.api_keys.config.API_KEY_REGISTRY_BACKEND", "sqlite")
    monkeypatch.setattr("src.api_keys.config.API_KEY_REGISTRY_FILE", api_key_path)
    monkeypatch.setattr("src.product_registry.config.PRODUCT_REGISTRY_BACKEND", "sqlite")
    monkeypatch.setattr("src.product_registry.config.PRODUCT_REGISTRY_FILE", product_registry_path)
    events = []
    calls = {"query": 0}

    class FakeResult:
        answer = "quota ok"
        provider_usage = None

    def fake_query_with_metadata(question, *, answer_mode):
        calls["query"] += 1
        return FakeResult()

    monkeypatch.setattr(api, "query_with_metadata", fake_query_with_metadata)
    monkeypatch.setattr(api, "append_runtime_event", lambda **kwargs: events.append(kwargs))

    client = TestClient(api.app)
    first = client.post(
        "/query",
        json={"question": "Explain quota guard"},
        headers={"X-API-Key": token, "X-Request-ID": "req-quota-1"},
    )
    second = client.post(
        "/query",
        json={"question": "Explain quota guard again"},
        headers={"X-API-Key": token, "X-Request-ID": "req-quota-2"},
    )

    assert first.status_code == 200
    assert first.headers["X-Product-Quota-Limit"] == "1"
    assert first.headers["X-Product-Quota-Remaining"] == "0"
    assert second.status_code == 429
    assert second.json()["detail"]["code"] == "quota_exceeded"
    assert second.headers["X-Product-Quota-Reason"] == "quota_exceeded"
    assert calls["query"] == 1
    assert product_registry.status()["usage_event_count"] == 1
    denied_events = [event for event in events if event["kind"] == "product_quota"]
    assert denied_events[0]["request_id"] == "req-quota-2"
    assert denied_events[0]["metadata"]["product_quota_limited"] is True
    assert denied_events[0]["metadata"]["product_workspace_present"] is True
    assert "product_workspace_id" not in denied_events[0]["metadata"]
    assert denied_events[0]["metadata"]["owner_id_present"] is True
    assert denied_events[0]["metadata"]["owner_label_present"] is True
    assert denied_events[0]["metadata"]["ownership_source"] == "api_key"
    assert "owner_id" not in denied_events[0]["metadata"]
    assert "owner_label" not in denied_events[0]["metadata"]
    assert token not in str(events)
    assert "local-workspace" not in str(events)
    assert "local-user" not in str(events)


def test_product_rbac_guard_allows_query_but_blocks_disallowed_writes(tmp_path, monkeypatch):
    api_key_path = tmp_path / "api_keys.sqlite3"
    product_registry_path = tmp_path / "product_registry.sqlite3"
    api_key_registry = LocalApiKeyRegistry(api_key_path)
    viewer_token = api_key_registry.create_key(owner_id="viewer-user")["token"]
    member_token = api_key_registry.create_key(owner_id="member-user")["token"]
    product_registry = LocalProductRegistry(product_registry_path)
    workspace = product_registry.create_workspace(
        workspace_id="rbac-workspace",
        owner_user_id="owner-user",
    )
    product_registry.add_member(
        workspace_id=workspace.workspace_id,
        user_id="viewer-user",
        role="viewer",
    )
    product_registry.add_member(
        workspace_id=workspace.workspace_id,
        user_id="member-user",
        role="member",
    )
    monkeypatch.setattr(api, "API_TOKEN", "")
    monkeypatch.setattr(api, "IDENTITY_QUOTAS_BILLING_ENABLED", True)
    monkeypatch.setattr(api, "PRODUCT_RBAC_GUARD_ENABLED", True)
    monkeypatch.setattr(api, "PRODUCT_REGISTRY_BACKEND", "sqlite")
    monkeypatch.setattr("src.api_keys.config.API_KEY_REGISTRY_BACKEND", "sqlite")
    monkeypatch.setattr("src.api_keys.config.API_KEY_REGISTRY_FILE", api_key_path)
    monkeypatch.setattr("src.product_registry.config.PRODUCT_REGISTRY_BACKEND", "sqlite")
    monkeypatch.setattr("src.product_registry.config.PRODUCT_REGISTRY_FILE", product_registry_path)
    events = []

    class FakeResult:
        answer = "rbac query ok"
        provider_usage = None

    monkeypatch.setattr(api, "query_with_metadata", lambda question, *, answer_mode: FakeResult())
    monkeypatch.setattr(api, "append_runtime_event", lambda **kwargs: events.append(kwargs))

    client = TestClient(api.app)
    query_response = client.post(
        "/query",
        json={"question": "Explain RBAC", "workspace_id": "rbac-workspace"},
        headers={"X-API-Key": viewer_token, "X-Request-ID": "req-rbac-query"},
    )
    job_response = client.post(
        "/jobs/image/mock",
        json={"prompt": "Draw SMC", "workspace_id": "rbac-workspace"},
        headers={"X-API-Key": viewer_token, "X-Request-ID": "req-rbac-job"},
    )
    index_response = client.post(
        "/jobs/index/rebuild",
        json={"source_paths": ["papers/library/example.pdf"], "workspace_id": "rbac-workspace"},
        headers={"X-API-Key": member_token, "X-Request-ID": "req-rbac-index"},
    )

    assert query_response.status_code == 200
    assert query_response.headers["X-Product-RBAC-Role"] == "viewer"
    assert job_response.status_code == 403
    assert job_response.json()["detail"]["code"] == "product_role_forbidden"
    assert job_response.headers["X-Product-RBAC-Action"] == "job_submit"
    assert job_response.headers["X-Product-RBAC-Role"] == "viewer"
    assert index_response.status_code == 403
    assert index_response.json()["detail"]["code"] == "product_role_forbidden"
    assert index_response.headers["X-Product-RBAC-Action"] == "corpus_write"
    denied_events = [event for event in events if event["kind"] == "product_rbac"]
    assert [event["request_id"] for event in denied_events] == ["req-rbac-job", "req-rbac-index"]
    assert denied_events[0]["metadata"]["product_workspace_present"] is True
    assert "product_workspace_id" not in denied_events[0]["metadata"]
    assert denied_events[0]["metadata"]["owner_id_present"] is True
    assert denied_events[0]["metadata"]["owner_label_present"] is True
    assert denied_events[0]["metadata"]["ownership_source"] == "api_key"
    assert "owner_id" not in denied_events[0]["metadata"]
    assert "owner_label" not in denied_events[0]["metadata"]
    assert viewer_token not in str(events)
    assert member_token not in str(events)
    assert "rbac-workspace" not in str(events)
    assert "viewer-user" not in str(events)
    assert "member-user" not in str(events)


def test_admin_product_registry_status_reports_disabled_backend(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")
    monkeypatch.setattr(api, "PRODUCT_REGISTRY_BACKEND", "none")

    client = TestClient(api.app)
    response = client.get("/admin/product-registry/status")
    workspaces = client.get("/admin/product-registry/workspaces")

    assert response.status_code == 200
    assert response.json()["status"]["available"] is False
    assert response.json()["status"]["reason"] == "product_registry_not_configured"
    assert workspaces.status_code == 503
    assert workspaces.json()["detail"]["code"] == "product_registry_not_configured"


def test_admin_product_registry_management_routes_use_local_backend(tmp_path, monkeypatch):
    db_path = tmp_path / "product_registry.sqlite3"
    monkeypatch.setattr(api, "API_TOKEN", "admin-token")
    monkeypatch.setattr(api, "PRODUCT_REGISTRY_BACKEND", "sqlite")
    monkeypatch.setattr("src.product_registry.config.PRODUCT_REGISTRY_BACKEND", "sqlite")
    monkeypatch.setattr("src.product_registry.config.PRODUCT_REGISTRY_FILE", db_path)
    events = []
    monkeypatch.setattr(api, "append_runtime_event", lambda **kwargs: events.append(kwargs))

    client = TestClient(api.app)
    headers = {"X-API-Key": "admin-token", "X-Request-ID": "req-product-admin"}
    created = client.post(
        "/admin/product-registry/workspaces",
        json={
            "workspace_id": "lab-ws",
            "label": "Lab Workspace",
            "owner_user_id": "lab-owner",
            "owner_label": "Lab Owner",
        },
        headers=headers,
    )
    member = client.post(
        "/admin/product-registry/workspaces/lab-ws/members",
        json={"user_id": "viewer-user", "label": "Viewer", "role": "viewer"},
        headers=headers,
    )
    quota = client.put(
        "/admin/product-registry/workspaces/lab-ws/quota",
        json={"metric": "requests", "limit_value": 7, "window_s": 3600},
        headers=headers,
    )
    billing = client.put(
        "/admin/product-registry/workspaces/lab-ws/billing",
        json={"billing_mode": "local-ledger", "status": "active", "attribution_enabled": True},
        headers=headers,
    )
    permission = client.post(
        "/admin/product-registry/permissions/check",
        json={"workspace_id": "lab-ws", "user_id": "viewer-user", "action": "job_submit"},
        headers=headers,
    )
    listed = client.get("/admin/product-registry/workspaces", headers=headers)
    detail = client.get("/admin/product-registry/workspaces/lab-ws", headers=headers)

    assert created.status_code == 200
    assert created.json()["workspace"]["workspace"]["workspace_id"] == "lab-ws"
    assert member.status_code == 200
    assert quota.status_code == 200
    assert billing.status_code == 200
    assert permission.status_code == 200
    assert permission.json()["permission"]["allowed"] is False
    assert permission.json()["permission"]["reason"] == "product_role_forbidden"
    assert listed.status_code == 200
    assert listed.json()["workspaces"][0]["member_count"] == 2
    assert listed.json()["workspaces"][0]["quota_limit_count"] == 1
    assert listed.json()["workspaces"][0]["billing_configured"] is True
    assert detail.status_code == 200
    payload = detail.json()["workspace"]
    assert [member["role"] for member in payload["members"]] == ["owner", "viewer"]
    assert payload["quota_limits"][0]["limit_value"] == 7
    assert payload["billing"]["attribution_enabled"] is True
    assert "admin-token" not in str(events)
    assert [event["kind"] for event in events] == [
        "product_registry_admin",
        "product_registry_admin",
        "product_registry_admin",
        "product_registry_admin",
        "product_registry_admin",
    ]
    assert events[-1]["metadata"]["product_workspace_present"] is True
    assert "product_workspace_id" not in events[-1]["metadata"]
    assert "lab-ws" not in str(events)


def test_admin_product_registry_management_routes_respect_product_rbac(tmp_path, monkeypatch):
    api_key_path = tmp_path / "api_keys.sqlite3"
    product_registry_path = tmp_path / "product_registry.sqlite3"
    api_key_registry = LocalApiKeyRegistry(api_key_path)
    owner_token = api_key_registry.create_key(owner_id="owner-user")["token"]
    viewer_token = api_key_registry.create_key(owner_id="viewer-user")["token"]
    product_registry = LocalProductRegistry(product_registry_path)
    workspace = product_registry.create_workspace(
        workspace_id="rbac-admin-workspace",
        owner_user_id="owner-user",
    )
    product_registry.add_member(
        workspace_id=workspace.workspace_id,
        user_id="viewer-user",
        role="viewer",
    )
    monkeypatch.setattr(api, "API_TOKEN", "")
    monkeypatch.setattr(api, "IDENTITY_QUOTAS_BILLING_ENABLED", True)
    monkeypatch.setattr(api, "PRODUCT_RBAC_GUARD_ENABLED", True)
    monkeypatch.setattr(api, "PRODUCT_REGISTRY_BACKEND", "sqlite")
    monkeypatch.setattr("src.api_keys.config.API_KEY_REGISTRY_BACKEND", "sqlite")
    monkeypatch.setattr("src.api_keys.config.API_KEY_REGISTRY_FILE", api_key_path)
    monkeypatch.setattr("src.product_registry.config.PRODUCT_REGISTRY_BACKEND", "sqlite")
    monkeypatch.setattr("src.product_registry.config.PRODUCT_REGISTRY_FILE", product_registry_path)
    events = []
    monkeypatch.setattr(api, "append_runtime_event", lambda **kwargs: events.append(kwargs))

    client = TestClient(api.app)
    viewer_list_response = client.get(
        "/admin/product-registry/workspaces",
        headers={"X-API-Key": viewer_token, "X-Request-ID": "req-viewer-list"},
    )
    viewer_detail_response = client.get(
        "/admin/product-registry/workspaces/rbac-admin-workspace",
        headers={"X-API-Key": viewer_token, "X-Request-ID": "req-viewer-detail"},
    )
    viewer_permission_response = client.post(
        "/admin/product-registry/permissions/check",
        json={
            "workspace_id": "rbac-admin-workspace",
            "user_id": "viewer-user",
            "action": "job_submit",
        },
        headers={"X-API-Key": viewer_token, "X-Request-ID": "req-viewer-permission"},
    )
    viewer_response = client.post(
        "/admin/product-registry/workspaces/rbac-admin-workspace/members",
        json={"user_id": "new-admin", "role": "admin"},
        headers={"X-API-Key": viewer_token},
    )
    owner_list_response = client.get(
        "/admin/product-registry/workspaces",
        headers={"X-API-Key": owner_token},
    )
    owner_detail_response = client.get(
        "/admin/product-registry/workspaces/rbac-admin-workspace",
        headers={"X-API-Key": owner_token},
    )
    owner_permission_response = client.post(
        "/admin/product-registry/permissions/check",
        json={
            "workspace_id": "rbac-admin-workspace",
            "user_id": "viewer-user",
            "action": "job_submit",
        },
        headers={"X-API-Key": owner_token},
    )
    owner_response = client.post(
        "/admin/product-registry/workspaces/rbac-admin-workspace/members",
        json={"user_id": "new-admin", "role": "admin"},
        headers={"X-API-Key": owner_token},
    )

    for response in [viewer_list_response, viewer_detail_response, viewer_permission_response]:
        assert response.status_code == 403
        assert response.json()["detail"]["code"] == "product_role_forbidden"
        assert response.headers["X-Product-RBAC-Action"] == "admin_write"
    assert viewer_response.status_code == 403
    assert viewer_response.json()["detail"]["code"] == "product_role_forbidden"
    assert viewer_response.headers["X-Product-RBAC-Action"] == "admin_write"
    assert owner_list_response.status_code == 200
    assert owner_list_response.json()["workspaces"][0]["workspace_id"] == "rbac-admin-workspace"
    assert owner_detail_response.status_code == 200
    assert owner_detail_response.json()["workspace"]["workspace"]["workspace_id"] == "rbac-admin-workspace"
    assert owner_permission_response.status_code == 200
    assert owner_permission_response.json()["permission"]["reason"] == "product_role_forbidden"
    assert owner_response.status_code == 200
    roles = {
        member["user_id"]: member["role"]
        for member in owner_response.json()["workspace"]["members"]
    }
    assert roles["new-admin"] == "admin"
    assert "rbac-admin-workspace" not in str(events)
    assert owner_token not in str(events)
    assert viewer_token not in str(events)


def test_admin_share_link_registry_routes_use_local_backend(tmp_path, monkeypatch):
    db_path = tmp_path / "share_links.sqlite3"
    monkeypatch.setattr(api, "API_TOKEN", "admin-token")
    monkeypatch.setattr(api, "SHARE_LINK_TOKEN_STORE_BACKEND", "sqlite")
    monkeypatch.setattr("src.share_links.config.SHARE_LINK_TOKEN_STORE_BACKEND", "sqlite")
    monkeypatch.setattr("src.share_links.config.SHARE_LINK_TOKEN_STORE_FILE", db_path)
    events = []
    monkeypatch.setattr(api, "append_runtime_event", lambda **kwargs: events.append(kwargs))

    client = TestClient(api.app)
    headers = {"X-API-Key": "admin-token", "X-Request-ID": "req-share-admin"}
    status = client.get("/admin/share-links/status", headers=headers)
    created = client.post(
        "/admin/share-links",
        json={
            "workspace_id": "lab-ws",
            "created_by_user_id": "owner-user",
            "resource_kind": "corpus_profile",
            "resource_ref": "private-profile-id",
            "description": "pilot share /private/hunter2 token=sk-secret-value",
        },
        headers=headers,
    )
    token = created.json()["token"]
    link_id = created.json()["share_link"]["link_id"]
    listed = client.get("/admin/share-links?workspace_id=lab-ws", headers=headers)
    resolved = client.post(
        "/admin/share-links/resolve",
        json={"token": token, "record_redeem": True},
        headers=headers,
    )
    revoked = client.post(
        f"/admin/share-links/{link_id}/revoke",
        headers=headers,
    )
    revoked_resolution = client.post(
        "/admin/share-links/resolve",
        json={"token": token},
        headers=headers,
    )

    assert status.status_code == 200
    assert status.json()["status"]["available"] is True
    assert created.status_code == 200
    assert token.startswith("fms_")
    assert "workspace_id" not in created.json()["share_link"]
    assert created.json()["share_link"]["workspace_present"] is True
    assert created.json()["share_link"]["workspace_fingerprint"]
    assert "lab-ws" not in json.dumps(created.json()["share_link"], sort_keys=True)
    assert "private-profile-id" not in str(created.json()["share_link"])
    assert "owner-user" not in str(created.json()["share_link"])
    assert "/private/hunter2" not in str(created.json()["share_link"])
    assert "sk-secret-value" not in str(created.json()["share_link"])
    assert listed.status_code == 200
    assert listed.json()["share_links"][0]["link_id"] == link_id
    assert resolved.status_code == 200
    assert resolved.json()["resolution"]["valid"] is True
    assert resolved.json()["resolution"]["share_link"]["redeem_count"] == 1
    assert revoked.status_code == 200
    assert revoked.json()["share_link"]["revoked_at"]
    assert revoked_resolution.status_code == 200
    assert revoked_resolution.json()["resolution"]["valid"] is False
    assert revoked_resolution.json()["resolution"]["reason"] == "share_link_revoked"
    for payload in [listed.json(), resolved.json(), revoked.json(), revoked_resolution.json()]:
        rendered = json.dumps(payload, sort_keys=True)
        assert token not in rendered
        assert "workspace_id" not in rendered
        assert "lab-ws" not in rendered
        assert "private-profile-id" not in rendered
        assert "owner-user" not in rendered
        assert "/private/hunter2" not in rendered
        assert "sk-secret-value" not in rendered
        assert "https://" not in rendered
    assert "admin-token" not in str(events)
    assert token not in str(events)
    assert "private-profile-id" not in str(events)
    assert "owner-user" not in str(events)
    assert "/private/hunter2" not in str(events)
    assert "sk-secret-value" not in str(events)
    assert "lab-ws" not in str(events)
    assert [event["kind"] for event in events] == [
        "share_link_admin",
        "share_link_admin",
        "share_link_admin",
        "share_link_admin",
        "share_link_admin",
    ]
    assert [event["metadata"]["action"] for event in events] == [
        "create",
        "list",
        "resolve",
        "revoke",
        "resolve",
    ]
    assert all(event["metadata"]["product_workspace_present"] is True for event in events)
    assert events[-1]["metadata"]["share_link_present"] is True
    assert events[-1]["metadata"]["status_code"] == 200
    assert events[-1]["metadata"]["share_link_valid"] is False
    assert "share_link_id" not in events[-1]["metadata"]


def test_admin_share_link_registry_routes_report_disabled_backend(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")
    monkeypatch.setattr(api, "SHARE_LINK_TOKEN_STORE_BACKEND", "none")

    client = TestClient(api.app)
    status = client.get("/admin/share-links/status")
    listed = client.get("/admin/share-links")

    assert status.status_code == 200
    assert status.json()["status"]["available"] is False
    assert status.json()["status"]["reason"] == "share_link_token_store_not_configured"
    assert listed.status_code == 503
    assert listed.json()["detail"]["code"] == "share_link_token_store_not_configured"


def test_query_inspect_returns_citation_validation(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")
    seen = {}
    usage_events = []

    class FakeResult:
        answer = "ok [1]"

        class FakeValidation:
            ok = True

        citation_validation = FakeValidation()

        def to_dict(self):
            return {
                "answer": self.answer,
                "answer_mode": "explanation",
                "citation_validation": {
                    "ok": True,
                    "cited_refs": [1],
                    "valid_refs": [1],
                    "invalid_refs": [],
                    "missing_required_refs": [],
                    "missing_source_page_refs": [],
                },
                "context_refs": [
                    {
                        "ref": 1,
                        "source": "paper.pdf",
                        "source_path": "papers/library/paper.pdf",
                        "page": 1,
                        "preview": "chunk",
                    }
                ],
            }

    def fake_query_with_metadata(question, *, answer_mode):
        seen["question"] = question
        seen["answer_mode"] = answer_mode
        return FakeResult()

    monkeypatch.setattr(api, "query_with_metadata", fake_query_with_metadata)
    monkeypatch.setattr(api, "append_runtime_event", lambda **kwargs: usage_events.append(kwargs))

    client = TestClient(api.app)
    response = client.post(
        "/query/inspect",
        json={"question": "Explain SMC"},
        headers={"X-Request-ID": "req-inspect"},
    )

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "req-inspect"
    payload = response.json()
    assert payload["request_id"] == "req-inspect"
    assert payload["result"]["citation_validation"]["ok"] is True
    assert payload["result"]["context_refs"][0]["page"] == 1
    assert seen == {"question": "Explain SMC", "answer_mode": "explanation"}
    assert usage_events[0]["kind"] == "query_usage"
    assert usage_events[0]["metadata"]["endpoint"] == "/query/inspect"
    assert usage_events[0]["metadata"]["citation_ok"] is True
    assert usage_events[0]["metadata"]["duration_ms"] >= 0
    trace_events = [event for event in usage_events if event["kind"] == "retrieval_trace"]
    assert trace_events[0]["request_id"] is None
    assert trace_events[0]["metadata"]["endpoint"] == "/query/inspect"
    assert trace_events[0]["metadata"]["context_count"] == 1
    assert trace_events[0]["metadata"]["citation_ok"] is True
    assert trace_events[0]["metadata"]["provider_called"] is True
    assert "papers/library/paper.pdf" not in str(trace_events[0])
    assert "chunk" not in str(trace_events[0])


def test_query_report_returns_markdown(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")
    usage_events = []

    class FakeResult:
        answer = "Report answer [1]"
        answer_mode = "literature_review"
        context_refs = [
            {
                "ref": 1,
                "source": "paper.pdf",
                "source_path": "papers/library/paper.pdf",
                "page": 2,
                "preview": "reported context",
            }
        ]

        class FakeValidation:
            ok = True

            def to_dict(self):
                return {
                    "ok": True,
                    "cited_refs": [1],
                    "valid_refs": [1],
                    "invalid_refs": [],
                    "missing_required_refs": [],
                    "missing_source_page_refs": [],
                }

        citation_validation = FakeValidation()

    def fake_query_with_metadata(question, *, answer_mode):
        assert question == "Summarize SMC"
        assert answer_mode == "literature_review"
        return FakeResult()

    monkeypatch.setattr(api, "query_with_metadata", fake_query_with_metadata)
    monkeypatch.setattr(api, "append_runtime_event", lambda **kwargs: usage_events.append(kwargs))
    monkeypatch.setattr(
        api,
        "enforce_product_quota",
        lambda **kwargs: {
            "enabled": True,
            "allowed": True,
            "reason": "allowed",
            "quota_configured": True,
            "limit_value": 3,
            "remaining": 2,
            "reset_after_s": 57,
        },
    )

    client = TestClient(api.app)
    response = client.post(
        "/query/report",
        json={"question": "Summarize SMC", "answer_mode": "literature_review"},
        headers={"X-Request-ID": "req-report"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/markdown")
    assert "fluxmind-query-report.md" in response.headers["content-disposition"]
    assert response.headers["X-Request-ID"] == "req-report"
    assert response.headers["X-Product-Quota-Reason"] == "allowed"
    assert response.headers["X-Product-Quota-Limit"] == "3"
    assert response.headers["X-Product-Quota-Remaining"] == "2"
    assert "# FluxMind Query Report" in response.text
    assert "Report answer [1]" in response.text
    assert "papers/library/paper.pdf" in response.text
    assert usage_events[0]["kind"] == "query_usage"
    assert usage_events[0]["metadata"]["endpoint"] == "/query/report"
    assert usage_events[0]["metadata"]["citation_ok"] is True
    assert usage_events[0]["metadata"]["duration_ms"] >= 0


def test_query_report_includes_paper_to_code_handoff(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")
    usage_events = []

    class FakeResult:
        answer = (
            "Use the cited PMSM model [1].\n\n"
            "```matlab\n"
            "Ts = 1e-4;\n"
            "i_alpha_hat = 0;\n"
            "```\n\n"
            "Attach the validation plot [Artifact:plot123]."
        )
        answer_mode = "code_generation"
        context_refs = [
            {
                "ref": 1,
                "source": "paper.pdf",
                "source_path": "papers/library/pmsm-smo.pdf",
                "page": 4,
                "preview": "stationary-frame PMSM model",
            }
        ]

        class FakeValidation:
            ok = True

            def to_dict(self):
                return {
                    "ok": True,
                    "cited_refs": [1],
                    "valid_refs": [1],
                    "invalid_refs": [],
                    "missing_required_refs": [],
                    "missing_source_page_refs": [],
                }

        citation_validation = FakeValidation()

    def fake_query_with_metadata(question, *, answer_mode):
        assert question == "Turn this paper into MATLAB observer code"
        assert answer_mode == "code_generation"
        return FakeResult()

    monkeypatch.setattr(api, "query_with_metadata", fake_query_with_metadata)
    monkeypatch.setattr(api, "append_runtime_event", lambda **kwargs: usage_events.append(kwargs))

    client = TestClient(api.app)
    response = client.post(
        "/query/report",
        json={
            "question": "Turn this paper into MATLAB observer code",
            "answer_mode": "code_generation",
        },
        headers={"X-Request-ID": "req-paper-code"},
    )

    assert response.status_code == 200
    assert "## Paper-to-Code Handoff" in response.text
    assert "### Source Trace" in response.text
    assert "papers/library/pmsm-smo.pdf" in response.text
    assert "### Generated Code Blocks" in response.text
    assert "- Code block 1: language=`matlab`" in response.text
    assert "Ts = 1e-4;" in response.text
    assert "### Execution Outputs and Plot Artifacts" in response.text
    assert "Cited artifact: `[Artifact:plot123]`" in response.text
    assert "Code blocks attached: 1" in response.text
    assert "Artifact refs attached: 1" in response.text
    assert usage_events[0]["metadata"]["endpoint"] == "/query/report"


def test_query_normalizes_provider_errors(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")

    def fail(_question, *, answer_mode):
        assert answer_mode == "explanation"
        raise TimeoutError("provider timed out")

    monkeypatch.setattr(api, "query_with_metadata", fail)

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


def test_query_provider_error_is_recorded_for_admin_history(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")
    monkeypatch.setattr("src.runtime.RUNTIME_EVENTS_FILE", tmp_path / "runtime_events.jsonl")

    def fail(_question, *, answer_mode):
        assert answer_mode == "implementation"
        raise TimeoutError("provider timed out")

    monkeypatch.setattr(api, "query_with_metadata", fail)

    client = TestClient(api.app)
    response = client.post(
        "/query",
        json={"question": "Explain SMC", "answer_mode": "implementation"},
        headers={"X-Request-ID": "req-history"},
    )

    assert response.status_code == 504
    events = (tmp_path / "runtime_events.jsonl").read_text(encoding="utf-8")
    assert "provider_failure" in events
    assert "provider_timeout" in events
    assert "duration_ms" in events
    assert "req-history" in events


def test_query_exception_event_uses_no_secret_ownership_metadata(monkeypatch):
    events = []
    monkeypatch.setattr(api, "append_runtime_event", lambda **kwargs: events.append(kwargs))

    error = api.record_query_exception_event(
        exc=TimeoutError("provider timed out"),
        endpoint="/query",
        request_id="req-owned-error",
        answer_mode="implementation",
        duration_ms=12,
        ownership={
            "owner_id": "secret-owner",
            "owner_label": "Secret Owner",
            "ownership_source": "request",
        },
    )

    assert error.code == "provider_timeout"
    assert events[0]["kind"] == "provider_failure"
    assert events[0]["request_id"] == "req-owned-error"
    metadata = events[0]["metadata"]
    assert metadata["owner_id_present"] is True
    assert metadata["owner_label_present"] is True
    assert metadata["ownership_source"] == "request"
    assert "owner_id" not in metadata
    assert "owner_label" not in metadata
    assert "secret-owner" not in str(events[0])
    assert "Secret Owner" not in str(events[0])


def test_query_provider_quota_guard_error_is_not_provider_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")
    monkeypatch.setattr("src.runtime.RUNTIME_EVENTS_FILE", tmp_path / "runtime_events.jsonl")

    def fail(_question, *, answer_mode):
        assert answer_mode == "implementation"
        raise ProviderQuotaGuardError(
            "Provider quota guard denied this request.",
            code="provider_prompt_token_limit_exceeded",
            status_code=429,
            decision={
                "enabled": True,
                "limited": True,
                "reason": "provider_prompt_token_limit_exceeded",
                "operation": "rag_generation",
                "provider": "deepseek-v3.2",
                "estimated_prompt_tokens": 129000,
                "requested_completion_tokens": 4096,
                "estimated_total_tokens": 133096,
                "max_prompt_tokens_per_request": 128000,
                "max_completion_tokens_per_request": 4096,
                "cost_limit_configured": False,
                "pricing_configured": False,
            },
        )

    monkeypatch.setattr(api, "query_with_metadata", fail)

    client = TestClient(api.app)
    response = client.post(
        "/query",
        json={"question": "Explain SMC", "answer_mode": "implementation"},
        headers={"X-Request-ID": "req-provider-guard"},
    )

    assert response.status_code == 429
    assert response.json()["detail"]["code"] == "provider_prompt_token_limit_exceeded"
    events = (tmp_path / "runtime_events.jsonl").read_text(encoding="utf-8")
    assert "provider_quota_guard" in events
    assert "provider_failure" not in events
    assert "provider_prompt_token_limit_exceeded" in events
    assert "estimated_prompt_tokens" in events
    assert "req-provider-guard" in events
    assert "Explain SMC" not in events


def test_query_provider_error_survives_event_log_failure(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")

    def fail(_question, *, answer_mode):
        raise TimeoutError("provider timed out")

    def event_log_fails(**_kwargs):
        raise OSError("metadata directory is read-only")

    monkeypatch.setattr(api, "query_with_metadata", fail)
    monkeypatch.setattr(api, "append_runtime_event", event_log_fails)

    client = TestClient(api.app)
    response = client.post(
        "/query",
        json={"question": "Explain SMC"},
        headers={"X-Request-ID": "req-log-fail"},
    )

    assert response.status_code == 504
    assert response.json()["detail"]["code"] == "provider_timeout"


def test_query_success_survives_usage_event_log_failure(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")

    class FakeResult:
        answer = "ok"
        provider_usage = None

    monkeypatch.setattr(api, "query_with_metadata", lambda question, *, answer_mode: FakeResult())

    def event_log_fails(**_kwargs):
        raise OSError("metadata directory is read-only")

    monkeypatch.setattr(api, "append_runtime_event", event_log_fails)

    client = TestClient(api.app)
    response = client.post("/query", json={"question": "Explain SMC"})

    assert response.status_code == 200
    assert response.json()["answer"] == "ok"


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


def test_corpus_papers_endpoint_filters_metadata(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")
    monkeypatch.setattr(
        api,
        "refresh_paper_metadata",
        lambda: [
            PaperRecord(
                paper_id="p1",
                source_path="papers/library/flux.pdf",
                filename="flux.pdf",
                source_kind="library",
                checksum_sha256="a" * 64,
                title="Flux Observer",
                authors="Flux Author",
                topic_tags=["flux", "observer"],
                active=True,
                indexed_status="indexed",
                updated_at="2026-05-30T00:00:00+00:00",
            ),
            PaperRecord(
                paper_id="p2",
                source_path="papers/uploads/sliding.pdf",
                filename="sliding.pdf",
                source_kind="upload",
                checksum_sha256="b" * 64,
                title="Sliding Mode",
                active=False,
                indexed_status="available",
                updated_at="2026-05-30T00:00:00+00:00",
            ),
        ],
    )

    client = TestClient(api.app)
    response = client.get(
        "/corpus/papers",
        params={
            "q": "observer",
            "active": "true",
            "source_kind": "library",
            "indexed_status": "indexed",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert [paper["source_path"] for paper in payload["papers"]] == [
        "papers/library/flux.pdf"
    ]


def test_corpus_chunks_endpoint_returns_chunk_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")

    class FakeChunkStore:
        def list_chunks(self, *, source_path, page, q, limit):
            assert source_path == "papers/library/paper.pdf"
            assert page == 1
            assert q == "preview"
            assert limit == 25
            from src.metadata import ChunkRecord

            return [
                ChunkRecord(
                    chunk_id="chunk1",
                    source_path="papers/library/paper.pdf",
                    source="paper.pdf",
                    page=1,
                    chunk_index=0,
                    content_sha256="a" * 64,
                    char_count=42,
                    preview="chunk preview",
                    updated_at="2026-05-31T00:00:00+00:00",
                )
            ]

    monkeypatch.setattr(api, "ChunkMetadataStore", FakeChunkStore)

    client = TestClient(api.app)
    response = client.get(
        "/corpus/chunks",
        params={"source_path": "papers/library/paper.pdf", "page": 1, "q": "preview", "limit": 25},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["chunks"][0]["chunk_id"] == "chunk1"
    assert payload["chunks"][0]["source_path"] == "papers/library/paper.pdf"


def test_corpus_structure_endpoint_returns_pdf_markers(tmp_path, monkeypatch):
    import fitz

    monkeypatch.setattr(api, "API_TOKEN", "")
    paper = tmp_path / "paper.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text(
        (72, 72),
        "PMSM voltage model\nud = Rsid + Ld did/dt\nTable 1. Parameter summary\nFigure 2. PMSM block diagram",
    )
    document.save(paper)
    document.close()
    monkeypatch.setattr(api, "resolve_selectable_source_paths", lambda _paths: [paper])

    client = TestClient(api.app)
    response = client.get(
        "/corpus/structure",
        params={
            "source_path": "papers/library/paper.pdf",
            "kind": "equation",
            "page": 1,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["markers"][0]["kind"] == "equation"
    assert "ud = Rsid" in payload["markers"][0]["text"]
    assert payload["markers"][0]["page"] == 1

    figure_response = client.get(
        "/corpus/structure",
        params={
            "source_path": "papers/library/paper.pdf",
            "kind": "figure",
            "page": 1,
            "q": "block diagram",
        },
    )

    assert figure_response.status_code == 200
    figure_payload = figure_response.json()
    assert figure_payload["markers"][0]["kind"] == "figure"
    assert "Figure 2" in figure_payload["markers"][0]["text"]

    report_response = client.get(
        "/corpus/structure/report",
        params={
            "source_path": "papers/library/paper.pdf",
            "kind": "figure",
            "page": 1,
            "q": "block diagram",
        },
    )

    assert report_response.status_code == 200
    assert report_response.headers["content-type"].startswith("text/markdown")
    assert "# FluxMind Corpus Structure Report" in report_response.text
    assert "Text filter: `block diagram`" in report_response.text
    assert "`figure`: 1" in report_response.text
    assert "Figure 2. PMSM block diagram" in report_response.text


def test_corpus_status_endpoint_returns_lifecycle_status(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")
    monkeypatch.setattr(
        api,
        "collect_corpus_status",
        lambda: {
            "status": "indexed",
            "papers": 1,
            "active": 1,
            "index": {"status": "fresh"},
            "index_jobs": {"by_status": {}, "latest": []},
        },
    )

    client = TestClient(api.app)
    response = client.get("/corpus/status")

    assert response.status_code == 200
    payload = response.json()["status"]
    assert payload["status"] == "indexed"
    assert payload["index"]["status"] == "fresh"


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


def test_corpus_profile_endpoints_create_list_and_activate(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")
    store = api.CorpusProfileStore(tmp_path / "corpus_profiles.json")
    activated = {}

    monkeypatch.setattr(
        api,
        "validate_corpus_profile_source_paths",
        lambda source_paths: source_paths,
    )

    def fake_set_active(source_paths):
        activated["source_paths"] = source_paths
        return [
            PaperRecord(
                paper_id="p1",
                source_path="papers/library/paper.pdf",
                filename="paper.pdf",
                source_kind="library",
                checksum_sha256="abc",
                title="Paper",
                active=True,
                indexed_status="active",
            )
        ]

    monkeypatch.setattr(api, "CorpusProfileStore", lambda: store)
    monkeypatch.setattr(api, "set_active_paper_source_paths", fake_set_active)

    client = TestClient(api.app)
    created = client.post(
        "/corpus/profiles",
        json={
            "profile_id": "smc-core",
            "name": "SMC Core",
            "description": "Core SMC papers",
            "source_paths": ["papers/library/paper.pdf"],
        },
    )
    listed = client.get("/corpus/profiles")
    activated_response = client.post("/corpus/profiles/smc-core/activate")

    assert created.status_code == 200
    assert created.json()["profile"]["profile_id"] == "smc-core"
    assert created.json()["profile"]["paper_count"] == 1
    assert listed.status_code == 200
    assert listed.json()["profiles"][0]["name"] == "SMC Core"
    assert activated_response.status_code == 200
    assert activated_response.json()["active_source_paths"] == ["papers/library/paper.pdf"]
    assert activated["source_paths"] == ["papers/library/paper.pdf"]


def test_corpus_profile_rebuild_endpoint_activates_and_queues_job(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")
    store = api.CorpusProfileStore(tmp_path / "corpus_profiles.json")
    store.upsert_profile(
        profile_id="smc-core",
        name="SMC Core",
        source_paths=["papers/library/paper.pdf"],
    )
    activated = {}
    queued = {}

    def fake_set_active(source_paths):
        activated["source_paths"] = source_paths
        return [
            PaperRecord(
                paper_id="p1",
                source_path="papers/library/paper.pdf",
                filename="paper.pdf",
                source_kind="library",
                checksum_sha256="abc",
                title="Paper",
                active=True,
                indexed_status="active",
            )
        ]

    class FakeAsyncJobManager:
        def enqueue_index_rebuild(
            self,
            source_paths,
            *,
            queue_timeout_s=None,
            request_id=None,
            idempotency_key=None,
            max_attempts=1,
            retry_backoff_s=0,
            ownership=None,
        ):
            queued["source_paths"] = source_paths
            queued["queue_timeout_s"] = queue_timeout_s
            queued["request_id"] = request_id
            queued["idempotency_key"] = idempotency_key
            queued["max_attempts"] = max_attempts
            queued["retry_backoff_s"] = retry_backoff_s
            queued["ownership"] = ownership
            return api.JobRecord(
                job_id="job-profile-rebuild",
                kind="index_rebuild",
                status="queued",
                created_at="2026-06-01T00:00:00+00:00",
                updated_at="2026-06-01T00:00:00+00:00",
                request={"source_paths": source_paths},
                request_id=request_id,
                deadline_at="2026-06-01T00:05:00+00:00",
            )

    monkeypatch.setattr(api, "CorpusProfileStore", lambda: store)
    monkeypatch.setattr(api, "set_active_paper_source_paths", fake_set_active)
    monkeypatch.setattr(api, "get_async_job_manager", lambda: FakeAsyncJobManager())

    client = TestClient(api.app)
    response = client.post(
        "/corpus/profiles/smc-core/rebuild",
        json={"queue_timeout_s": 300},
        headers={"X-Request-ID": "req-profile-rebuild"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["profile"]["profile_id"] == "smc-core"
    assert payload["active_source_paths"] == ["papers/library/paper.pdf"]
    assert payload["rebuild_required"] is True
    assert payload["queued_rebuild"] is True
    assert payload["job"]["kind"] == "index_rebuild"
    assert payload["job"]["status"] == "queued"
    assert payload["job"]["request_id"] == "req-profile-rebuild"
    assert payload["job"]["owner_id"] == "local-user"
    assert activated["source_paths"] == ["papers/library/paper.pdf"]
    assert queued == {
        "source_paths": ["papers/library/paper.pdf"],
        "queue_timeout_s": 300,
        "request_id": "req-profile-rebuild",
        "idempotency_key": None,
        "max_attempts": 1,
        "retry_backoff_s": 0,
        "ownership": {
            "owner_id": "local-user",
            "owner_label": "Local user",
            "ownership_source": "default",
        },
    }


def test_corpus_profile_status_endpoint_reports_profile_index_state(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")
    index_dir = tmp_path / "faiss_index"
    index_dir.mkdir()
    (index_dir / "index.faiss").write_text("index", encoding="utf-8")
    store = api.CorpusProfileStore(tmp_path / "corpus_profiles.json")
    store.upsert_profile(
        profile_id="smc-core",
        name="SMC Core",
        source_paths=["papers/library/paper.pdf"],
    )

    monkeypatch.setattr("src.admin.FAISS_INDEX_DIR", index_dir)
    monkeypatch.setattr("src.admin.CorpusProfileStore", lambda: store)
    monkeypatch.setattr(
        "src.admin.refresh_paper_metadata",
        lambda: [
            PaperRecord(
                paper_id="p1",
                source_path="papers/library/paper.pdf",
                filename="paper.pdf",
                source_kind="library",
                checksum_sha256="abc",
                title="Paper",
                active=True,
                indexed_status="indexed",
            )
        ],
    )

    class FakeChunkStore:
        def source_paths(self):
            return ["papers/library/paper.pdf"]

    monkeypatch.setattr("src.admin.ChunkMetadataStore", FakeChunkStore)

    client = TestClient(api.app)
    response = client.get("/corpus/profiles/smc-core/status")

    assert response.status_code == 200
    status = response.json()["status"]
    assert status["profile"]["profile_id"] == "smc-core"
    assert status["available_papers"] == 1
    assert status["missing_source_paths"] == []
    assert status["active_match"] is True
    assert status["rebuild_required"] is False
    assert status["index"]["status"] == "fresh"


def test_corpus_profile_report_endpoint_returns_markdown(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")
    index_dir = tmp_path / "faiss_index"
    index_dir.mkdir()
    (index_dir / "index.faiss").write_text("index", encoding="utf-8")
    store = api.CorpusProfileStore(tmp_path / "corpus_profiles.json")
    store.upsert_profile(
        profile_id="smc-core",
        name="SMC Core",
        description="Core SMC papers",
        source_paths=["papers/library/paper.pdf"],
    )

    monkeypatch.setattr("src.admin.FAISS_INDEX_DIR", index_dir)
    monkeypatch.setattr("src.admin.CorpusProfileStore", lambda: store)
    monkeypatch.setattr(
        "src.admin.refresh_paper_metadata",
        lambda: [
            PaperRecord(
                paper_id="p1",
                source_path="papers/library/paper.pdf",
                filename="paper.pdf",
                source_kind="library",
                checksum_sha256="abc",
                title="Paper",
                active=True,
                indexed_status="indexed",
            )
        ],
    )

    class FakeChunkStore:
        def source_paths(self):
            return ["papers/library/paper.pdf"]

    monkeypatch.setattr("src.admin.ChunkMetadataStore", FakeChunkStore)

    client = TestClient(api.app)
    response = client.get("/corpus/profiles/smc-core/report")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/markdown")
    assert "fluxmind-corpus-profile-smc-core.md" in response.headers["content-disposition"]
    assert "# FluxMind Corpus Profile Status" in response.text
    assert "Profile ID: smc-core" in response.text
    assert "Rebuild required: False" in response.text
    assert "papers/library/paper.pdf" in response.text
    assert "api_key" not in response.text.lower()


def test_corpus_profile_report_endpoint_uses_normalized_download_filename(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")
    index_dir = tmp_path / "faiss_index"
    index_dir.mkdir()
    (index_dir / "index.faiss").write_text("index", encoding="utf-8")
    store = api.CorpusProfileStore(tmp_path / "corpus_profiles.json")
    store.upsert_profile(
        profile_id="SMC Core",
        name="SMC Core",
        source_paths=["papers/library/paper.pdf"],
    )

    monkeypatch.setattr("src.admin.FAISS_INDEX_DIR", index_dir)
    monkeypatch.setattr("src.admin.CorpusProfileStore", lambda: store)
    monkeypatch.setattr(
        "src.admin.refresh_paper_metadata",
        lambda: [
            PaperRecord(
                paper_id="p1",
                source_path="papers/library/paper.pdf",
                filename="paper.pdf",
                source_kind="library",
                checksum_sha256="abc",
                title="Paper",
                active=True,
                indexed_status="indexed",
            )
        ],
    )

    class FakeChunkStore:
        def source_paths(self):
            return ["papers/library/paper.pdf"]

    monkeypatch.setattr("src.admin.ChunkMetadataStore", FakeChunkStore)

    client = TestClient(api.app)
    response = client.get("/corpus/profiles/SMC%20Core/report")

    assert response.status_code == 200
    disposition = response.headers["content-disposition"]
    assert 'filename="fluxmind-corpus-profile-smc-core.md"' in disposition
    assert "SMC Core" not in disposition


def test_corpus_profile_report_filename_is_header_safe():
    filename = safe_corpus_profile_report_filename(
        'abc"\r\nContent-Disposition: x-secret'
    )

    assert filename.startswith("fluxmind-corpus-profile-")
    assert filename.endswith(".md")
    assert "content-disposition" not in filename.lower()
    assert "secret" not in filename
    assert '"' not in filename
    assert "\r" not in filename
    assert "\n" not in filename


def test_corpus_profile_status_endpoint_reports_stale_profile(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")
    index_dir = tmp_path / "faiss_index"
    index_dir.mkdir()
    (index_dir / "index.faiss").write_text("index", encoding="utf-8")
    store = api.CorpusProfileStore(tmp_path / "corpus_profiles.json")
    store.upsert_profile(
        profile_id="small",
        name="Small",
        source_paths=["papers/library/paper.pdf"],
    )

    monkeypatch.setattr("src.admin.FAISS_INDEX_DIR", index_dir)
    monkeypatch.setattr("src.admin.CorpusProfileStore", lambda: store)
    monkeypatch.setattr(
        "src.admin.refresh_paper_metadata",
        lambda: [
            PaperRecord(
                paper_id="p1",
                source_path="papers/library/paper.pdf",
                filename="paper.pdf",
                source_kind="library",
                checksum_sha256="abc",
                title="Paper",
                active=False,
                indexed_status="available",
            ),
            PaperRecord(
                paper_id="p2",
                source_path="papers/library/other.pdf",
                filename="other.pdf",
                source_kind="library",
                checksum_sha256="def",
                title="Other",
                active=True,
                indexed_status="indexed",
            ),
        ],
    )

    class FakeChunkStore:
        def source_paths(self):
            return ["papers/library/other.pdf"]

    monkeypatch.setattr("src.admin.ChunkMetadataStore", FakeChunkStore)

    client = TestClient(api.app)
    response = client.get("/corpus/profiles/small/status")

    assert response.status_code == 200
    status = response.json()["status"]
    assert status["active_match"] is False
    assert status["rebuild_required"] is True
    assert status["index"]["status"] == "stale"
    assert status["index"]["missing_chunk_sources"] == ["papers/library/paper.pdf"]
    assert status["index"]["extra_chunk_sources"] == ["papers/library/other.pdf"]


def test_corpus_profile_status_endpoint_returns_404_for_missing_profile(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")

    client = TestClient(api.app)
    response = client.get("/corpus/profiles/missing/status")

    assert response.status_code == 404
    assert response.json()["detail"] == "Corpus profile not found"


def test_query_retrieve_endpoint_returns_no_llm_diagnostics(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")
    trace_events = []

    class FakeRetrieval:
        context_count = 1
        ok = True

        def to_dict(self):
            return {
                "answer_mode": "literature_review",
                "context_count": 1,
                "citation_instruction": "Valid numbered source refs for this answer: [1] only.",
                "context_refs": [
                    {
                        "ref": 1,
                        "source": "paper.pdf",
                        "source_path": "papers/library/paper.pdf",
                        "page": 2,
                        "preview": "sliding mode observer",
                    }
                ],
                "missing_source_page_refs": [],
                "ok": True,
            }

    def fake_retrieve(question, *, answer_mode):
        assert question == "Explain SMC"
        assert answer_mode == "literature_review"
        return FakeRetrieval()

    monkeypatch.setattr(api, "retrieve_with_metadata", fake_retrieve)
    monkeypatch.setattr(api, "append_runtime_event", lambda **kwargs: trace_events.append(kwargs))

    client = TestClient(api.app)
    response = client.post(
        "/query/retrieve",
        json={"question": "Explain SMC", "answer_mode": "literature_review"},
        headers={"X-Request-ID": "req-retrieve"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["request_id"] == "req-retrieve"
    assert payload["retrieval"]["ok"] is True
    assert payload["retrieval"]["context_refs"][0]["source_path"] == "papers/library/paper.pdf"
    assert trace_events[0]["kind"] == "retrieval_trace"
    assert trace_events[0]["request_id"] is None
    assert trace_events[0]["metadata"]["endpoint"] == "/query/retrieve"
    assert trace_events[0]["metadata"]["answer_mode"] == "literature_review"
    assert trace_events[0]["metadata"]["context_count"] == 1
    assert trace_events[0]["metadata"]["missing_source_page_count"] == 0
    assert trace_events[0]["metadata"]["provider_called"] is False
    assert trace_events[0]["metadata"]["retrieval_ok"] is True
    assert "req-retrieve" not in str(trace_events[0])
    assert "Explain SMC" not in str(trace_events[0])
    assert "papers/library/paper.pdf" not in str(trace_events[0])
    assert "sliding mode observer" not in str(trace_events[0])


def test_corpus_profile_endpoint_rejects_unselectable_path(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")

    def fail(_source_paths):
        raise ValueError("PDF path is not in the selectable corpus: missing.pdf")

    monkeypatch.setattr(api, "validate_corpus_profile_source_paths", fail)

    client = TestClient(api.app)
    response = client.post(
        "/corpus/profiles",
        json={"name": "Bad", "source_paths": ["missing.pdf"]},
    )

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


def test_admin_retention_endpoint_returns_preview(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")

    def fake_preview(*, upload_days, artifact_days, limit):
        assert upload_days == 7
        assert artifact_days == 14
        assert limit == 25
        return {
            "mode": "preview",
            "delete_enabled": False,
            "uploads": {"total_candidates": 1},
            "artifacts": {"total_candidates": 0},
        }

    monkeypatch.setattr(api, "collect_retention_preview", fake_preview)

    client = TestClient(api.app)
    response = client.get(
        "/admin/retention",
        params={"upload_days": 7, "artifact_days": 14, "limit": 25},
    )

    assert response.status_code == 200
    payload = response.json()["retention"]
    assert payload["mode"] == "preview"
    assert payload["delete_enabled"] is False
    assert payload["uploads"]["total_candidates"] == 1


def test_admin_retention_delete_endpoint_rejects_when_disabled(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")

    def fake_delete(*, upload_days, artifact_days, limit):
        assert upload_days == 7
        assert artifact_days == 14
        assert limit == 25
        return {
            "mode": "delete_disabled",
            "delete_enabled": False,
            "deleted_files": 0,
            "deleted_bytes": 0,
            "failed_files": 0,
        }

    monkeypatch.setattr(api, "apply_retention_delete", fake_delete)

    client = TestClient(api.app)
    response = client.post(
        "/admin/retention/delete",
        params={"upload_days": 7, "artifact_days": 14, "limit": 25},
    )

    assert response.status_code == 403
    detail = response.json()["detail"]
    assert detail["code"] == "retention_delete_disabled"
    assert detail["retention"]["deleted_files"] == 0


def test_admin_retention_delete_endpoint_returns_enabled_result(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")

    def fake_delete(*, upload_days, artifact_days, limit):
        assert upload_days == 7
        assert artifact_days == 14
        assert limit == 25
        return {
            "mode": "delete",
            "delete_enabled": True,
            "deleted_files": 2,
            "deleted_bytes": 12,
            "failed_files": 0,
            "uploads": {"deleted_files": 1},
            "artifacts": {"deleted_files": 1},
        }

    monkeypatch.setattr(api, "apply_retention_delete", fake_delete)

    client = TestClient(api.app)
    response = client.post(
        "/admin/retention/delete",
        params={"upload_days": 7, "artifact_days": 14, "limit": 25},
    )

    assert response.status_code == 200
    payload = response.json()["retention"]
    assert payload["mode"] == "delete"
    assert payload["deleted_files"] == 2
    assert "api_key" not in str(payload).casefold()


def test_admin_events_endpoint_filters_runtime_events(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")

    def fake_events(*, kind, code, q, limit):
        assert kind == "provider_failure"
        assert code == "provider_timeout"
        assert q is None
        assert limit == 1000
        return [
            RuntimeEvent(
                event_id="evt-1",
                kind="provider_failure",
                code="provider_timeout",
                message="timeout",
                created_at="2026-06-01T00:00:00+00:00",
                request_id="req-events",
                metadata={"endpoint": "/query"},
            )
        ]

    monkeypatch.setattr(api, "list_runtime_events", fake_events)

    client = TestClient(api.app)
    response = client.get(
        "/admin/events",
        params={
            "kind": "provider_failure",
            "code": "provider_timeout",
            "q": "req-events",
            "limit": 25,
        },
    )

    assert response.status_code == 200
    event = response.json()["events"][0]
    assert event["event_id"] == "evt-1"
    assert event["request_id"] == "req-events"
    assert event["metadata"]["endpoint"] == "/query"


def test_admin_events_endpoint_redacts_sensitive_metadata(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")

    def fake_events(*, kind, code, q, limit):
        return [
            RuntimeEvent(
                event_id="evt-sensitive",
                kind="query_usage",
                code="estimated_usage",
                message=(
                    "safe message tokenValue=secret-message-token "
                    "sourcePath=/private/message.pdf https://internal.example/request"
                ),
                created_at="2026-06-01T00:00:00+00:00",
                request_id="Bearer secret-request-token",
                metadata={
                    "endpoint": "/query",
                    "status_code": 200,
                    "prompt": "raw prompt",
                    "answer": "raw answer",
                    "owner_id": "owner-secret",
                    "auth_key_id": "key-secret",
                    "auth_owner_id": "auth-owner-secret",
                    "product_workspace_id": "workspace-secret",
                    "workspaceId": "workspace-camel-secret",
                    "user_id": "user-secret",
                    "memberUserId": "member-secret",
                    "source_path": "papers/uploads/private.pdf",
                    "workspace_count": 2,
                    "user_count": 3,
                    "nested": {
                        "token": "secret-token",
                        "safe_count": 2,
                    },
                    "items": [
                        {
                            "filename": "private.pdf",
                            "safe": "kept",
                        }
                    ],
                },
            )
        ]

    monkeypatch.setattr(api, "list_runtime_events", fake_events)

    client = TestClient(api.app)
    response = client.get("/admin/events")

    assert response.status_code == 200
    event = response.json()["events"][0]
    assert event["message"] == "Runtime event message redacted for no-secret projection."
    assert event["message_redacted"] is True
    assert "request_id" not in event
    assert event["request_id_present"] is True
    assert event["request_id_redacted"] is True
    assert event["metadata"] == {
        "endpoint": "/query",
        "status_code": 200,
        "workspace_count": 2,
        "user_count": 3,
        "nested": {"safe_count": 2},
        "items": [{"safe": "kept"}],
    }
    assert event["metadata_redacted_fields"] == 12
    body = response.text.casefold()
    for sensitive in (
        "raw prompt",
        "raw answer",
        "owner-secret",
        "key-secret",
        "auth-owner-secret",
        "workspace-secret",
        "workspace-camel-secret",
        "user-secret",
        "member-secret",
        "private.pdf",
        "secret-token",
        "secret-message-token",
        "secret-request-token",
        "/private/message.pdf",
        "internal.example",
        "source_path",
        "owner_id",
        "auth_key_id",
        "auth_owner_id",
        "product_workspace_id",
        "workspaceid",
        "user_id",
        "memberuserid",
    ):
        assert sensitive not in body

    sensitive_search = client.get("/admin/events", params={"q": "secret-token"})
    sensitive_message_search = client.get(
        "/admin/events",
        params={"q": "secret-message-token"},
    )
    safe_search = client.get("/admin/events", params={"q": "/query"})

    assert sensitive_search.status_code == 200
    assert sensitive_search.json()["events"] == []
    assert sensitive_message_search.status_code == 200
    assert sensitive_message_search.json()["events"] == []
    assert safe_search.status_code == 200
    assert safe_search.json()["events"][0]["event_id"] == "evt-sensitive"


def test_admin_status_report_endpoint_returns_markdown(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")

    class FakeStatus:
        def to_dict(self):
            return {
                "jobs": {
                    "total": 1,
                    "by_status": {"succeeded": 1},
                    "by_kind": {"code_execution": 1},
                    "failed": 0,
                    "scheduled": 0,
                    "queue_health": {"queued": 0},
                    "storage": {"jsonl_exists": True},
                    "latest_failed": [],
                },
                "corpus": {
                    "papers": 1,
                    "active": 1,
                    "indexed": 1,
                    "failed": 0,
                    "storage": {},
                    "chunks": {},
                    "index": {"status": "fresh"},
                },
                "artifacts": {"total": 0, "bytes": 0, "storage": {}},
                "provider_failures": {
                    "total_recent": 0,
                    "by_code": {},
                    "event_log_exists": False,
                    "event_log_bytes": 0,
                    "latest": [],
                },
                "query_usage": {
                    "total_recent": 0,
                    "by_endpoint": {},
                    "by_answer_mode": {},
                    "estimated_prompt_tokens": 0,
                    "estimated_answer_tokens": 0,
                    "estimated_total_tokens": 0,
                    "estimated_cost_usd": "0",
                    "cost_source": "not_configured",
                    "pricing": {
                        "configured": False,
                        "reason": "not_configured",
                        "provider": "test-model",
                        "currency": "USD",
                        "prompt_usd_per_1m": "0",
                        "completion_usd_per_1m": "0",
                        "external_billing_enabled": False,
                    },
                    "latest": [],
                },
                "runtime_dirs": [],
                "config": {
                    "llm_model": "test-model",
                    "embedding_model": "test-embed",
                    "llm_base_url_configured": True,
                    "external_providers_enabled": False,
                    "identity_quotas_billing_enabled": False,
                },
            }

    monkeypatch.setattr(api, "collect_admin_status", lambda: FakeStatus())

    client = TestClient(api.app)
    response = client.get("/admin/status/report")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/markdown")
    assert "fluxmind-admin-status.md" in response.headers["content-disposition"]
    assert "# FluxMind Admin Status" in response.text
    assert "test-model" in response.text
    assert "api_key" not in response.text.lower()


def test_admin_metrics_endpoint_returns_no_secret_metrics(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")

    class FakeStatus:
        def to_dict(self):
            return {
                "jobs": {
                    "total": 1,
                    "by_status": {"succeeded": 1},
                    "by_kind": {"code_execution": 1},
                },
                "query_usage": {
                    "total_recent": 1,
                    "by_endpoint": {"/query": 1},
                    "estimated_total_tokens": 12,
                    "provider_total_tokens": 0,
                    "duration_ms": {"avg": 8, "max": 8},
                    "alerts": [],
                },
                "api_access": {
                    "audit_enabled": True,
                    "total_recent": 1,
                    "by_token_status": {"valid": 1},
                    "by_status_code": {"200": 1},
                    "by_method": {"GET": 1},
                    "valid_recent": 1,
                },
                "config": {
                    "api_rate_limit_enabled": False,
                    "retention_delete_enabled": False,
                    "storage_readiness": {},
                    "docker_execution": {},
                },
            }

    monkeypatch.setattr(api, "collect_admin_status", lambda: FakeStatus())

    client = TestClient(api.app)
    response = client.get("/admin/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "fluxmind-admin-metrics.prom" in response.headers["content-disposition"]
    assert "fluxmind_jobs_total 1" in response.text
    assert 'fluxmind_api_access_by_token_status{token_status="valid"} 1' in response.text
    assert "api_key" not in response.text.lower()
    assert "owner" not in response.text.lower()


def test_admin_openapi_contract_endpoint_returns_no_secret_status(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")
    monkeypatch.setattr(
        api,
        "collect_openapi_contract",
        lambda schema: {
            "mode": "openapi_contract",
            "local_contract_ready": True,
            "raw_schema_exported": False,
            "secrets_exported": False,
            "paths_exported": False,
            "route_count": 10,
            "operation_count": 20,
        },
    )

    client = TestClient(api.app)
    response = client.get("/admin/openapi-contract")

    assert response.status_code == 200
    payload = response.json()["openapi_contract"]
    assert payload["mode"] == "openapi_contract"
    assert payload["local_contract_ready"] is True
    assert payload["raw_schema_exported"] is False
    assert payload["secrets_exported"] is False
    assert "components" not in str(payload).lower()
    assert "hunter2" not in str(payload)


def test_admin_openapi_contract_endpoint_records_metadata_only_check_event(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")
    monkeypatch.setattr(api, "API_ACCESS_AUDIT_ENABLED", True)
    events = []
    monkeypatch.setattr(api, "append_runtime_event", lambda **kwargs: events.append(kwargs))
    monkeypatch.setattr(
        api,
        "collect_openapi_contract",
        lambda schema: {
            "mode": "openapi_contract",
            "local_contract_ready": False,
            "raw_schema_exported": False,
            "secrets_exported": False,
            "paths_exported": False,
            "route_count": 10,
            "operation_count": 20,
            "required_operation_missing_count": 1,
            "undocumented_operation_count": 0,
            "response_missing_operation_count": 0,
            "protected_operation_count": 8,
            "protected_auth_header_operation_count": 7,
            "operation_fingerprint": "hunter2-fingerprint",
            "blockers": ["hunter2_blocker"],
        },
    )

    client = TestClient(api.app)
    response = client.get("/admin/openapi-contract")

    assert response.status_code == 200
    admin_events = [event for event in events if event["kind"] == "admin_check"]
    assert len(admin_events) == 1
    event = admin_events[0]
    assert event["code"] == "openapi_contract_blocked"
    assert event["message"] == "Metadata-only admin readiness check event."
    metadata = event["metadata"]
    assert metadata["check"] == "openapi_contract"
    assert metadata["ok"] is False
    assert metadata["route_count"] == 10
    assert metadata["operation_count"] == 20
    assert metadata["blocker_count"] == 1
    rendered = json.dumps(event, ensure_ascii=False, sort_keys=True)
    assert "hunter2" not in rendered
    assert "operation_fingerprint" not in rendered


def test_admin_check_event_helpers_project_no_secret_summaries(monkeypatch):
    monkeypatch.setattr(api, "API_ACCESS_AUDIT_ENABLED", True)
    events = []
    monkeypatch.setattr(api, "append_runtime_event", lambda **kwargs: events.append(kwargs))

    api._record_quality_readiness_check(
        {
            "local_foundation_ready": True,
            "small_group_ready": True,
            "community_ready": False,
            "live_evidence_included": True,
            "evidence_requests": [{"secret_path": "/private/hunter2.json"}],
        }
    )
    api._record_product_activation_check(
        {
            "ok": True,
            "readiness": {"local_foundation_ready": True, "activation_ready": False},
            "api_key_lifecycle": {"active_key_count": 2},
            "product_registry": {"workspace_count": 1},
            "raw_token": "hunter2",
        }
    )
    api._record_provider_runtime_check(
        {
            "ok": True,
            "readiness": {"local_foundation_ready": True},
            "external_activation_ready": False,
            "docker_execution": {"available": False},
            "artifact_uri": "file:///private/hunter2.svg",
        }
    )
    api._record_platform_migration_check(
        {
            "rehearsal_ok": True,
            "summary": {
                "source_preflight_ok": True,
                "restore_check_ok": True,
                "object_manifest_ready": True,
                "job_store_manifest_ready": True,
                "copied_files": 3,
            },
            "blockers": [],
            "source_path": "/private/hunter2",
        }
    )
    api._record_activation_suite_check(
        {
            "local_foundation_ready": True,
            "full_activation_ready": False,
            "blockers": {
                "local_foundation": ["openapi_contract"],
                "full_activation": [
                    "product_readiness_activation_not_ready",
                    "collaboration_activation_not_ready",
                ],
            },
            "activation_action_plan": {"step_count": 2},
            "live_report": {"secret_path": "/private/hunter2.json"},
        }
    )

    codes = {event["code"] for event in events}
    assert codes == {
        "quality_readiness_ok",
        "product_activation_rehearsal_ok",
        "provider_runtime_rehearsal_ok",
        "platform_migration_rehearsal_ok",
        "activation_suite_ok",
    }
    rendered = json.dumps(events, ensure_ascii=False, sort_keys=True)
    assert "hunter2" not in rendered
    assert "/private" not in rendered
    assert "file://" not in rendered
    assert all(event["kind"] == "admin_check" for event in events)
    assert all(event["metadata"]["content_exported"] is False for event in events)
    assert all(event["metadata"]["secrets_exported"] is False for event in events)
    assert all(event["metadata"]["paths_exported"] is False for event in events)
    activation_event = [
        event for event in events if event["metadata"]["check"] == "activation_suite"
    ][0]
    assert activation_event["metadata"]["failed_check_count"] == 1
    assert activation_event["metadata"]["full_activation_blocker_count"] == 2
    assert activation_event["metadata"]["activation_step_count"] == 2


def test_admin_openapi_contract_report_downloads_markdown(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")
    monkeypatch.setattr(
        api,
        "collect_openapi_contract",
        lambda schema: {
            "mode": "openapi_contract",
            "local_contract_ready": True,
            "secrets_exported": False,
        },
    )
    monkeypatch.setattr(
        api,
        "format_openapi_contract_markdown",
        lambda status: "# FluxMind OpenAPI Contract\n\n- Secrets exported: false\n",
    )

    client = TestClient(api.app)
    response = client.get("/admin/openapi-contract/report")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/markdown")
    assert "fluxmind-openapi-contract.md" in response.headers["content-disposition"]
    assert "# FluxMind OpenAPI Contract" in response.text
    assert "Secrets exported: false" in response.text
    assert "api_key" not in response.text.lower()


def test_admin_openapi_contract_verify_endpoint_returns_no_secret_status(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")
    seen = {}
    monkeypatch.setattr(
        api,
        "collect_openapi_contract",
        lambda schema: {
            "mode": "openapi_contract",
            "local_contract_ready": True,
            "operation_fingerprint": "new",
        },
    )

    def fake_verify(current, snapshot):
        seen["current"] = current
        seen["snapshot"] = snapshot
        return {
            "mode": "openapi_contract_snapshot_verify",
            "ok": False,
            "diff_count": 1,
            "raw_schema_exported": False,
            "secrets_exported": False,
            "paths_exported": False,
            "blockers": ["snapshot_contract_drift"],
        }

    monkeypatch.setattr(api, "verify_openapi_contract_snapshot", fake_verify)

    client = TestClient(api.app)
    response = client.post(
        "/admin/openapi-contract/verify",
        json={"snapshot": {"mode": "openapi_contract", "operation_fingerprint": "old"}},
    )

    assert response.status_code == 200
    payload = response.json()["openapi_contract_snapshot_verify"]
    assert payload["mode"] == "openapi_contract_snapshot_verify"
    assert payload["ok"] is False
    assert payload["raw_schema_exported"] is False
    assert "snapshot_contract_drift" in payload["blockers"]
    assert seen["snapshot"]["operation_fingerprint"] == "old"
    assert "components" not in str(payload).lower()
    assert "hunter2" not in str(payload)


def test_admin_openapi_contract_verify_records_snapshot_summary_only(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")
    monkeypatch.setattr(api, "API_ACCESS_AUDIT_ENABLED", True)
    events = []
    monkeypatch.setattr(api, "append_runtime_event", lambda **kwargs: events.append(kwargs))
    monkeypatch.setattr(
        api,
        "collect_openapi_contract",
        lambda schema: {
            "mode": "openapi_contract",
            "local_contract_ready": True,
            "operation_fingerprint": "new",
        },
    )

    def fake_verify(current, snapshot):
        return {
            "mode": "openapi_contract_snapshot_verify",
            "ok": False,
            "diff_count": 1,
            "compared_field_count": 10,
            "snapshot_shape_valid": False,
            "snapshot_raw_schema_included": True,
            "raw_schema_exported": False,
            "secrets_exported": False,
            "paths_exported": False,
            "blockers": ["snapshot_contract_drift"],
            "diffs": [{"field": "route_count", "snapshot": "/private/hunter2"}],
        }

    monkeypatch.setattr(api, "verify_openapi_contract_snapshot", fake_verify)

    client = TestClient(api.app)
    response = client.post(
        "/admin/openapi-contract/verify",
        json={
            "snapshot": {
                "mode": "openapi_contract",
                "paths": {"/private/hunter2": {"get": {}}},
                "components": {"schemas": {"SecretThing": "hunter2"}},
            }
        },
    )

    assert response.status_code == 200
    admin_events = [event for event in events if event["kind"] == "admin_check"]
    assert len(admin_events) == 1
    event = admin_events[0]
    assert event["code"] == "openapi_contract_snapshot_verify_blocked"
    metadata = event["metadata"]
    assert metadata["check"] == "openapi_contract_snapshot_verify"
    assert metadata["ok"] is False
    assert metadata["diff_count"] == 1
    assert metadata["snapshot_shape_valid"] is False
    assert metadata["snapshot_raw_schema_included"] is True
    assert metadata["blocker_count"] == 1
    rendered = json.dumps(event, ensure_ascii=False, sort_keys=True)
    assert "hunter2" not in rendered
    assert "/private" not in rendered
    assert "components" not in rendered
    assert "diffs" not in rendered


def test_admin_openapi_contract_verify_report_downloads_markdown(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")
    monkeypatch.setattr(
        api,
        "collect_openapi_contract",
        lambda schema: {"mode": "openapi_contract", "local_contract_ready": True},
    )
    monkeypatch.setattr(
        api,
        "verify_openapi_contract_snapshot",
        lambda current, snapshot: {
            "mode": "openapi_contract_snapshot_verify",
            "ok": True,
            "secrets_exported": False,
        },
    )
    monkeypatch.setattr(
        api,
        "format_openapi_contract_snapshot_verify_markdown",
        lambda status: "# FluxMind OpenAPI Contract Snapshot Verify\n\n- Secrets exported: false\n",
    )

    client = TestClient(api.app)
    response = client.post(
        "/admin/openapi-contract/verify/report",
        json={"snapshot": {"mode": "openapi_contract"}},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/markdown")
    assert "fluxmind-openapi-contract-verify.md" in response.headers["content-disposition"]
    assert "# FluxMind OpenAPI Contract Snapshot Verify" in response.text
    assert "Secrets exported: false" in response.text
    assert "api_key" not in response.text.lower()


def test_admin_activation_suite_endpoint_returns_no_secret_status(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")
    monkeypatch.setattr(
        api,
        "collect_activation_suite",
        lambda **kwargs: {
            "mode": "activation_suite",
            "local_foundation_ready": True,
            "full_activation_ready": False,
            "content_exported": False,
            "secrets_exported": False,
            "paths_exported": False,
            "checks": {
                "product_activation": {"ok": True},
                "provider_runtime": {"ok": True},
            },
            "activation_action_plan": {
                "target": "full_activation",
                "ready": False,
                "step_count": 1,
                "steps": [
                    {
                        "area": "community_quality",
                        "ready": False,
                        "command": ".venv/bin/python scripts/quality_readiness.py --format markdown",
                        "verification_command": (
                            ".venv/bin/python scripts/quality_readiness.py "
                            "--live-report <report.json> --require-target community --format markdown"
                        ),
                    }
                ],
            },
        },
    )

    client = TestClient(api.app)
    response = client.get("/admin/activation-suite")

    assert response.status_code == 200
    payload = response.json()["activation_suite"]
    assert payload["mode"] == "activation_suite"
    assert payload["local_foundation_ready"] is True
    assert payload["full_activation_ready"] is False
    assert payload["secrets_exported"] is False
    plan = payload["activation_action_plan"]
    assert plan["target"] == "full_activation"
    assert plan["steps"][0]["area"] == "community_quality"
    assert "<report.json>" in plan["steps"][0]["verification_command"]
    assert "api_key" not in str(payload).lower()
    assert "file://" not in str(payload)


def test_admin_quality_readiness_endpoint_returns_no_secret_status(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")
    monkeypatch.setattr(
        api,
        "collect_quality_readiness",
        lambda: {
            "mode": "quality_readiness",
            "local_foundation_ready": True,
            "small_group_ready": False,
            "community_ready": False,
            "secrets_exported": False,
            "paths_exported": False,
            "evidence_requests": [
                {
                    "target": "small_group",
                    "ready": False,
                    "evidence_sources": ["live_eval_report"],
                    "items": [],
                }
            ],
            "community_evidence_plan": {
                "target": "community",
                "ready": False,
                "content_exported": False,
                "secrets_exported": False,
                "paths_exported": False,
                "steps": [
                    {
                        "evidence_source": "live_eval_report",
                        "metrics": ["live_answer_result_count"],
                        "command": (
                            ".venv/bin/python scripts/evaluate_rag.py "
                            "--live-url <api-base-url> --api-key-env "
                            "FLUXMIND_API_TOKEN --json-report <report.json>"
                        ),
                    }
                ],
            },
        },
    )

    client = TestClient(api.app)
    response = client.get("/admin/quality-readiness")

    assert response.status_code == 200
    payload = response.json()["quality_readiness"]
    assert payload["mode"] == "quality_readiness"
    assert payload["local_foundation_ready"] is True
    assert payload["small_group_ready"] is False
    assert payload["secrets_exported"] is False
    assert payload["evidence_requests"][0]["evidence_sources"] == ["live_eval_report"]
    plan = payload["community_evidence_plan"]
    assert plan["target"] == "community"
    assert plan["steps"][0]["evidence_source"] == "live_eval_report"
    assert "<api-base-url>" in plan["steps"][0]["command"]
    assert "<report.json>" in plan["steps"][0]["command"]
    assert "api_key" not in str(payload).lower()
    assert "file://" not in str(payload)


def test_admin_product_activation_rehearsal_endpoint_returns_no_secret_status(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")
    monkeypatch.setattr(
        api,
        "collect_product_activation_rehearsal",
        lambda: {
            "mode": "product_activation_rehearsal",
            "ok": True,
            "local_only": True,
            "content_exported": False,
            "secrets_exported": False,
            "paths_exported": False,
            "api_key_lifecycle": {"active_key_count": 2, "revoked_key_count": 1},
            "readiness": {"activation_ready": True},
        },
    )

    client = TestClient(api.app)
    response = client.get("/admin/product-activation-rehearsal")

    assert response.status_code == 200
    payload = response.json()["product_activation_rehearsal"]
    assert payload["mode"] == "product_activation_rehearsal"
    assert payload["ok"] is True
    assert payload["secrets_exported"] is False
    assert payload["paths_exported"] is False
    assert payload["api_key_lifecycle"]["active_key_count"] == 2
    assert "fmk_" not in str(payload)
    assert "api_keys.sqlite3" not in str(payload)
    assert "file://" not in str(payload)


def test_admin_product_activation_rehearsal_report_downloads_markdown(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")
    monkeypatch.setattr(
        api,
        "collect_product_activation_rehearsal",
        lambda: {
            "mode": "product_activation_rehearsal",
            "ok": True,
            "secrets_exported": False,
        },
    )
    monkeypatch.setattr(
        api,
        "format_product_activation_rehearsal_markdown",
        lambda status: "# FluxMind Product Activation Rehearsal\n\n- Secrets exported: false\n",
    )

    client = TestClient(api.app)
    response = client.get("/admin/product-activation-rehearsal/report")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/markdown")
    assert "fluxmind-product-activation-rehearsal.md" in response.headers["content-disposition"]
    assert "# FluxMind Product Activation Rehearsal" in response.text
    assert "Secrets exported: false" in response.text
    assert "fmk_" not in response.text
    assert "api_keys.sqlite3" not in response.text


def test_admin_collaboration_readiness_endpoint_returns_no_secret_status(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")
    monkeypatch.setattr(
        api,
        "collect_collaboration_readiness",
        lambda: {
            "mode": "collaboration_readiness",
            "ok": True,
            "local_foundation_ready": True,
            "safe_default_ready": True,
            "activation_ready": False,
            "content_exported": False,
            "secrets_exported": False,
            "paths_exported": False,
            "identifiers_exported": False,
            "share_tokens_exported": False,
            "share_urls_exported": False,
            "summary": {
                "private_corpora_enabled": False,
                "share_links_enabled": False,
                "policy_scenario_count": 13,
            },
            "blockers": {"activation": ["share_links_disabled"]},
        },
    )

    client = TestClient(api.app)
    response = client.get("/admin/collaboration-readiness")

    assert response.status_code == 200
    payload = response.json()["collaboration_readiness"]
    assert payload["mode"] == "collaboration_readiness"
    assert payload["ok"] is True
    assert payload["safe_default_ready"] is True
    assert payload["activation_ready"] is False
    assert payload["secrets_exported"] is False
    assert payload["identifiers_exported"] is False
    assert payload["share_tokens_exported"] is False
    assert payload["share_urls_exported"] is False
    assert payload["summary"]["policy_scenario_count"] == 13
    assert "share-token" not in str(payload)
    assert "workspace-" not in str(payload)
    assert "file://" not in str(payload)


def test_admin_collaboration_readiness_report_downloads_markdown(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")
    monkeypatch.setattr(
        api,
        "collect_collaboration_readiness",
        lambda: {
            "mode": "collaboration_readiness",
            "ok": True,
            "secrets_exported": False,
        },
    )
    monkeypatch.setattr(
        api,
        "format_collaboration_readiness_markdown",
        lambda status: "# FluxMind Collaboration Readiness\n\n- Secrets exported: false\n",
    )

    client = TestClient(api.app)
    response = client.get("/admin/collaboration-readiness/report")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/markdown")
    assert "fluxmind-collaboration-readiness.md" in response.headers["content-disposition"]
    assert "# FluxMind Collaboration Readiness" in response.text
    assert "Secrets exported: false" in response.text
    assert "share-token" not in response.text
    assert "workspace-" not in response.text


def test_admin_provider_runtime_rehearsal_endpoint_returns_no_secret_status(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")
    monkeypatch.setattr(
        api,
        "collect_provider_runtime_rehearsal",
        lambda: {
            "mode": "provider_runtime_rehearsal",
            "ok": True,
            "local_only": True,
            "external_activation_ready": False,
            "content_exported": False,
            "secrets_exported": False,
            "paths_exported": False,
            "connectivity_checked": False,
            "image_provider": {"ok": True, "provider": "local-mock-svg-v1"},
            "python_execution": {"ok": True, "artifact_count": 1},
            "octave_execution": {"ok": True, "reason": "runtime_unavailable"},
            "provider_quota_guard": {
                "ok": True,
                "blocked_reason": "provider_prompt_token_limit_exceeded",
            },
        },
    )

    client = TestClient(api.app)
    response = client.get("/admin/provider-runtime-rehearsal")

    assert response.status_code == 200
    payload = response.json()["provider_runtime_rehearsal"]
    assert payload["mode"] == "provider_runtime_rehearsal"
    assert payload["ok"] is True
    assert payload["external_activation_ready"] is False
    assert payload["secrets_exported"] is False
    assert payload["paths_exported"] is False
    assert payload["provider_quota_guard"]["blocked_reason"] == "provider_prompt_token_limit_exceeded"
    assert "sk-" not in str(payload)
    assert "file://" not in str(payload)
    assert "/tmp/" not in str(payload)


def test_admin_provider_runtime_rehearsal_report_downloads_markdown(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")
    monkeypatch.setattr(
        api,
        "collect_provider_runtime_rehearsal",
        lambda: {
            "mode": "provider_runtime_rehearsal",
            "ok": True,
            "secrets_exported": False,
        },
    )
    monkeypatch.setattr(
        api,
        "format_provider_runtime_rehearsal_markdown",
        lambda status: "# FluxMind Provider Runtime Rehearsal\n\n- Secrets exported: false\n",
    )

    client = TestClient(api.app)
    response = client.get("/admin/provider-runtime-rehearsal/report")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/markdown")
    assert "fluxmind-provider-runtime-rehearsal.md" in response.headers["content-disposition"]
    assert "# FluxMind Provider Runtime Rehearsal" in response.text
    assert "Secrets exported: false" in response.text
    assert "sk-" not in response.text
    assert "file://" not in response.text


def test_admin_platform_migration_rehearsal_endpoint_returns_no_secret_status(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")
    monkeypatch.setattr(
        api,
        "collect_platform_migration_rehearsal",
        lambda: {
            "mode": "local_runtime_migration_rehearsal",
            "rehearsal_ok": True,
            "activation_enabled": False,
            "content_exported_in_report": False,
            "secrets_exported": False,
            "paths_exported": False,
            "raw_manifests_included": False,
            "summary": {
                "object_manifest_ready": True,
                "job_store_manifest_ready": True,
            },
            "object_storage_manifest_summary": {"object_count": 3},
            "job_store_manifest_summary": {"job_count": 1},
        },
    )

    client = TestClient(api.app)
    response = client.get("/admin/platform-migration-rehearsal")

    assert response.status_code == 200
    payload = response.json()["platform_migration_rehearsal"]
    assert payload["mode"] == "local_runtime_migration_rehearsal"
    assert payload["rehearsal_ok"] is True
    assert payload["secrets_exported"] is False
    assert payload["paths_exported"] is False
    assert payload["raw_manifests_included"] is False
    assert payload["summary"]["object_manifest_ready"] is True
    assert "object_storage_manifest" not in payload
    assert "job_store_manifest" not in payload
    assert "secret-job-id" not in str(payload)
    assert "file://" not in str(payload)
    assert "/tmp/" not in str(payload)


def test_admin_platform_migration_rehearsal_report_downloads_markdown(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")
    monkeypatch.setattr(
        api,
        "collect_platform_migration_rehearsal",
        lambda: {
            "mode": "local_runtime_migration_rehearsal",
            "rehearsal_ok": True,
            "secrets_exported": False,
            "paths_exported": False,
            "raw_manifests_included": False,
            "summary": {},
            "copy": {"groups": []},
        },
    )
    monkeypatch.setattr(
        api,
        "format_storage_migration_rehearsal_markdown",
        lambda status: "# FluxMind Runtime Migration Rehearsal\n\n- Secrets exported: false\n",
    )

    client = TestClient(api.app)
    response = client.get("/admin/platform-migration-rehearsal/report")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/markdown")
    assert "fluxmind-platform-migration-rehearsal.md" in response.headers["content-disposition"]
    assert "# FluxMind Runtime Migration Rehearsal" in response.text
    assert "Secrets exported: false" in response.text
    assert "secret-job-id" not in response.text
    assert "file://" not in response.text


def test_admin_quality_readiness_post_accepts_live_report_without_echo(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")
    seen = {}

    def fake_quality(**kwargs):
        seen.update(kwargs)
        return {
            "mode": "quality_readiness",
            "local_foundation_ready": True,
            "small_group_ready": True,
            "community_ready": False,
            "secrets_exported": False,
            "paths_exported": False,
        }

    monkeypatch.setattr(api, "collect_quality_readiness", fake_quality)

    client = TestClient(api.app)
    response = client.post(
        "/admin/quality-readiness",
        json={
            "live_report": {
                "secret_path": "/private/hunter2-quality-report.json",
                "summary": {"live_retrieval": {"total": 107, "ok": 107}},
            }
        },
    )

    assert response.status_code == 200
    assert seen["live_reports"][0]["secret_path"] == "/private/hunter2-quality-report.json"
    payload = response.json()["quality_readiness"]
    assert payload["small_group_ready"] is True
    assert "/private/hunter2-quality-report.json" not in str(payload)
    assert "hunter2" not in str(payload)


def test_admin_quality_readiness_report_downloads_markdown(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")
    monkeypatch.setattr(
        api,
        "collect_quality_readiness",
        lambda: {
            "mode": "quality_readiness",
            "local_foundation_ready": True,
            "secrets_exported": False,
        },
    )
    monkeypatch.setattr(
        api,
        "format_quality_readiness_markdown",
        lambda status: "# FluxMind Quality Readiness\n\n- Secrets exported: false\n",
    )

    client = TestClient(api.app)
    response = client.get("/admin/quality-readiness/report")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/markdown")
    assert "fluxmind-quality-readiness.md" in response.headers["content-disposition"]
    assert "# FluxMind Quality Readiness" in response.text
    assert "Secrets exported: false" in response.text
    assert "api_key" not in response.text.lower()


def test_admin_quality_readiness_post_report_downloads_markdown(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")
    seen = {}

    def fake_quality(**kwargs):
        seen.update(kwargs)
        return {
            "mode": "quality_readiness",
            "small_group_ready": True,
            "secrets_exported": False,
        }

    monkeypatch.setattr(api, "collect_quality_readiness", fake_quality)
    monkeypatch.setattr(
        api,
        "format_quality_readiness_markdown",
        lambda status: "# FluxMind Quality Readiness\n\n- Small-group ready: true\n",
    )

    client = TestClient(api.app)
    response = client.post(
        "/admin/quality-readiness/report",
        json={
            "live_reports": [
                {"secret_path": "/private/hunter2-quality-report.json"},
            ]
        },
    )

    assert response.status_code == 200
    assert seen["live_reports"][0]["secret_path"] == "/private/hunter2-quality-report.json"
    assert response.headers["content-type"].startswith("text/markdown")
    assert "fluxmind-quality-readiness.md" in response.headers["content-disposition"]
    assert "Small-group ready: true" in response.text
    assert "hunter2" not in response.text


def test_admin_activation_suite_post_accepts_live_report_without_echo(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")
    seen = {}

    def fake_suite(**kwargs):
        seen.update(kwargs)
        return {
            "mode": "activation_suite",
            "local_foundation_ready": True,
            "small_group_ready": True,
            "full_activation_ready": False,
            "secrets_exported": False,
            "paths_exported": False,
        }

    monkeypatch.setattr(api, "collect_activation_suite", fake_suite)

    client = TestClient(api.app)
    response = client.post(
        "/admin/activation-suite",
        json={
            "live_report": {
                "secret_path": "/private/hunter2-report.json",
                "summary": {"live_retrieval": {"total": 107, "ok": 107}},
            }
        },
    )

    assert response.status_code == 200
    assert seen["live_reports"][0]["secret_path"] == "/private/hunter2-report.json"
    payload = response.json()["activation_suite"]
    assert payload["small_group_ready"] is True
    assert "/private/hunter2-report.json" not in str(payload)
    assert "hunter2" not in str(payload)


def test_admin_activation_suite_report_downloads_markdown(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")
    monkeypatch.setattr(
        api,
        "collect_activation_suite",
        lambda **kwargs: {
            "mode": "activation_suite",
            "local_foundation_ready": True,
            "full_activation_ready": False,
            "secrets_exported": False,
        },
    )
    monkeypatch.setattr(
        api,
        "format_activation_suite_markdown",
        lambda status: "# FluxMind Activation Suite\n\n- Secrets exported: false\n",
    )

    client = TestClient(api.app)
    response = client.get("/admin/activation-suite/report")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/markdown")
    assert "fluxmind-activation-suite.md" in response.headers["content-disposition"]
    assert "# FluxMind Activation Suite" in response.text
    assert "Secrets exported: false" in response.text
    assert "api_key" not in response.text.lower()


def test_admin_activation_suite_post_report_downloads_markdown(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")
    seen = {}

    def fake_suite(**kwargs):
        seen.update(kwargs)
        return {
            "mode": "activation_suite",
            "small_group_ready": True,
            "secrets_exported": False,
        }

    monkeypatch.setattr(api, "collect_activation_suite", fake_suite)
    monkeypatch.setattr(
        api,
        "format_activation_suite_markdown",
        lambda status: "# FluxMind Activation Suite\n\n- Small-group ready: true\n",
    )

    client = TestClient(api.app)
    response = client.post(
        "/admin/activation-suite/report",
        json={
            "live_reports": [
                {"secret_path": "/private/hunter2-report.json"},
            ]
        },
    )

    assert response.status_code == 200
    assert seen["live_reports"][0]["secret_path"] == "/private/hunter2-report.json"
    assert response.headers["content-type"].startswith("text/markdown")
    assert "fluxmind-activation-suite.md" in response.headers["content-disposition"]
    assert "Small-group ready: true" in response.text
    assert "hunter2" not in response.text


def test_admin_runtime_manifest_endpoint_returns_no_secret_manifest(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")

    manifest = {
        "mode": "local_runtime_backup_manifest",
        "content_exported": False,
        "secrets_exported": False,
        "env_file_present": True,
        "env_file_content_exported": False,
        "total_files": 2,
        "total_bytes": 123,
        "groups": [],
    }
    monkeypatch.setattr(api, "collect_runtime_backup_manifest", lambda: manifest)

    client = TestClient(api.app)
    response = client.get("/admin/runtime-manifest")

    assert response.status_code == 200
    payload = response.json()["manifest"]
    assert payload["mode"] == "local_runtime_backup_manifest"
    assert payload["content_exported"] is False
    assert payload["secrets_exported"] is False
    assert payload["env_file_content_exported"] is False
    assert "api_key" not in str(payload).lower()


def test_admin_runtime_manifest_report_endpoint_returns_markdown(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")

    manifest = {
        "mode": "local_runtime_backup_manifest",
        "content_exported": False,
        "secrets_exported": False,
        "env_file_present": True,
        "env_file_content_exported": False,
        "total_files": 2,
        "total_bytes": 123,
        "groups": [],
    }
    monkeypatch.setattr(api, "collect_runtime_backup_manifest", lambda: manifest)

    client = TestClient(api.app)
    response = client.get("/admin/runtime-manifest/report")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/markdown")
    assert "fluxmind-runtime-manifest.md" in response.headers["content-disposition"]
    assert "# FluxMind Runtime Backup Manifest" in response.text
    assert "Secrets exported: false" in response.text
    assert "api_key" not in response.text.lower()


def test_admin_runtime_manifest_restore_check_endpoint_returns_no_secret_check(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")

    manifest = {"mode": "local_runtime_backup_manifest", "groups": []}

    def fake_restore_check(received_manifest):
        assert received_manifest == manifest
        return {
            "mode": "local_runtime_restore_dry_run",
            "content_restored": False,
            "delete_enabled": False,
            "ok": True,
            "groups": [],
        }

    monkeypatch.setattr(api, "collect_runtime_restore_check", fake_restore_check)

    client = TestClient(api.app)
    response = client.post(
        "/admin/runtime-manifest/restore-check",
        json={"manifest": manifest},
    )

    assert response.status_code == 200
    payload = response.json()["restore_check"]
    assert payload["mode"] == "local_runtime_restore_dry_run"
    assert payload["content_restored"] is False
    assert payload["delete_enabled"] is False
    assert payload["ok"] is True
    assert "api_key" not in str(payload).lower()


def test_admin_runtime_manifest_restore_check_report_endpoint_returns_markdown(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")

    manifest = {"mode": "local_runtime_backup_manifest", "groups": []}
    restore_check = {
        "mode": "local_runtime_restore_dry_run",
        "content_restored": False,
        "delete_enabled": False,
        "ok": True,
        "groups": [],
    }

    monkeypatch.setattr(api, "collect_runtime_restore_check", lambda received: restore_check)
    monkeypatch.setattr(
        api,
        "format_runtime_restore_check_markdown",
        lambda check: "# FluxMind Runtime Restore Dry Run\n\nOK: true\n",
    )

    client = TestClient(api.app)
    response = client.post(
        "/admin/runtime-manifest/restore-check/report",
        json={"manifest": manifest},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/markdown")
    assert "fluxmind-runtime-restore-dry-run.md" in response.headers["content-disposition"]
    assert "# FluxMind Runtime Restore Dry Run" in response.text
    assert "api_key" not in response.text.lower()


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
        ),
        ownership={"owner_id": "lab-artifact-api", "owner_label": "Artifact API Lab"},
    )
    job.status = "succeeded"
    job.artifacts = [
        {
            "kind": "text",
            "uri": uri,
            "mime_type": "text/plain",
            "title": "result.txt",
            "metadata": {
                "provider": "local",
                "prompt": "private artifact prompt",
                "reference_uris": ["paper://private#page=4"],
            },
        }
    ]
    store.append(job)
    monkeypatch.setattr("src.artifacts.ARTIFACTS_DIR", artifact_root)
    monkeypatch.setattr("src.artifacts.LocalJobStore", lambda: store)

    client = TestClient(api.app)
    listed = client.get("/artifacts")
    filtered = client.get(
        "/artifacts",
        params={
            "q": artifact_id_for_uri(uri),
            "kind": "text",
            "job_kind": "code_execution",
            "owner_id": "lab-artifact-api",
        },
    )
    missing = client.get("/artifacts", params={"kind": "image"})
    artifact_id = artifact_id_for_uri(uri)
    downloaded = client.get(f"/artifacts/{artifact_id}")

    assert listed.status_code == 200
    public_artifact = listed.json()["artifacts"][0]
    assert public_artifact["artifact_id"] == artifact_id
    assert public_artifact["metadata"]["provider_present"] is True
    assert public_artifact["metadata"]["reference_count"] == 1
    serialized_artifact = json.dumps(public_artifact, sort_keys=True)
    for sensitive in (
        uri,
        "result.txt",
        "lab-artifact-api",
        "Artifact API Lab",
        "private artifact prompt",
        "paper://private#page=4",
    ):
        assert sensitive not in serialized_artifact
    assert filtered.status_code == 200
    assert filtered.json()["artifacts"][0]["artifact_id"] == artifact_id
    assert missing.status_code == 200
    assert missing.json()["artifacts"] == []
    assert downloaded.status_code == 200
    assert downloaded.text == "artifact-body"
    assert f"artifact-{artifact_id}.txt" in downloaded.headers["content-disposition"]
    assert "result.txt" not in downloaded.headers["content-disposition"]


def test_artifact_download_rejects_symlink_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    target = artifact_root / "target.txt"
    linked = artifact_root / "linked.txt"
    target.write_text("target-body", encoding="utf-8")
    linked.symlink_to(target)
    uri = linked.as_uri()
    store = api.LocalJobStore(tmp_path / "jobs.jsonl")
    store.append_new(
        api.JobRecord(
            job_id="job-symlink-artifact",
            kind="code_execution",
            status="succeeded",
            created_at="2026-06-19T00:00:00+00:00",
            updated_at="2026-06-19T00:00:00+00:00",
            request={},
            artifacts=[
                {
                    "kind": "text",
                    "uri": uri,
                    "mime_type": "text/plain",
                    "title": "linked.txt",
                }
            ],
        )
    )
    monkeypatch.setattr("src.artifacts.ARTIFACTS_DIR", artifact_root)
    monkeypatch.setattr("src.artifacts.LocalJobStore", lambda: store)

    client = TestClient(api.app)
    response = client.get(f"/artifacts/{artifact_id_for_uri(uri)}")

    assert response.status_code == 400
    assert response.json()["detail"] == "Artifact symlinks cannot be exported."
    assert target.read_text(encoding="utf-8") == "target-body"


def test_mock_image_job_endpoint_returns_persisted_job(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")
    monkeypatch.setattr("src.jobs.JOBS_FILE", tmp_path / "jobs.jsonl")
    monkeypatch.setattr("src.providers.ARTIFACTS_DIR", tmp_path / "artifacts")

    client = TestClient(api.app)
    response = client.post(
        "/jobs/image/mock",
        json={
            "prompt": "Draw an SMC observer",
            "diagram_template": "sliding-mode-observer",
            "owner_id": "lab-image-api",
            "owner_label": "Image API Lab",
        },
        headers={"X-Request-ID": "req-image"},
    )

    assert response.status_code == 200
    job = response.json()["job"]
    assert job["kind"] == "image_generation"
    assert job["status"] == "succeeded"
    assert job["request_id"] == "req-image"
    assert job["owner_id"] == "lab-image-api"
    assert job["owner_label"] == "Image API Lab"
    assert job["ownership_source"] == "request"
    assert job["artifacts"][0]["mime_type"] == "image/svg+xml"
    assert job["artifacts"][0]["metadata"]["provider_present"] is True
    assert job["artifacts"][0]["metadata"]["diagram_template_present"] is True
    assert job["artifacts"][0]["metadata"]["reference_count"] == 0
    assert job["artifacts"][0]["metadata"]["cost_estimate_usd"] == "0"
    assert "uri" not in job["artifacts"][0]
    assert "prompt" not in job["artifacts"][0]["metadata"]
    assert "diagram_template" not in job["artifacts"][0]["metadata"]
    assert [entry["status"] for entry in job["logs"]] == ["running", "succeeded"]

    loaded = client.get(f"/jobs/{job['job_id']}")
    assert loaded.status_code == 200
    assert loaded.json()["job"]["job_id"] == job["job_id"]
    assert loaded.json()["job"]["logs"] == job["logs"]


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
    assert job["result"]["runtime_metadata"]["memory_mb"] == "512"


def test_local_python_job_endpoint_handles_input_path_conflict(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")
    monkeypatch.setattr("src.jobs.JOBS_FILE", tmp_path / "jobs.jsonl")

    client = TestClient(api.app)
    response = client.post(
        "/jobs/code/python-local",
        json={
            "entrypoint": "main.py",
            "files": {
                "main.py": "print('should-not-run')",
                "main.py/helper.txt": "conflict",
            },
        },
    )

    assert response.status_code == 200
    job = response.json()["job"]
    assert job["kind"] == "code_execution"
    assert job["status"] == "failed"
    assert job["error"]["code"] == "execution_failed"
    assert "could not be materialized: main.py/helper.txt" in job["result"]["stderr"]
    assert "/tmp/fluxmind-" not in job["result"]["stderr"]


def test_local_python_job_endpoint_uses_configured_docker_backend(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")
    monkeypatch.setattr("src.jobs.JOBS_FILE", tmp_path / "jobs.jsonl")
    monkeypatch.setattr("src.jobs.CODE_EXECUTION_BACKEND", "docker")
    monkeypatch.setattr("src.jobs.DOCKER_EXECUTION_IMAGE", "python:3.12-slim")
    monkeypatch.setattr("src.providers.shutil.which", lambda _name: "/usr/bin/docker")

    class FakePopen:
        returncode = 0

        def __init__(self, command, **_kwargs):
            mount = command[command.index("-v") + 1]
            workdir = Path(mount.split(":", 1)[0])
            (workdir / "api-docker.txt").write_text("api-docker-artifact", encoding="utf-8")

        def poll(self):
            return self.returncode

        def communicate(self, timeout=None):
            return "api-docker-ok\n", ""

    monkeypatch.setattr("src.providers.subprocess.Popen", FakePopen)

    client = TestClient(api.app)
    response = client.post(
        "/jobs/code/python-local",
        json={
            "entrypoint": "main.py",
            "files": {"main.py": "print('api-docker-ok')"},
        },
    )

    assert response.status_code == 200
    job = response.json()["job"]
    assert job["status"] == "succeeded"
    assert job["result"]["stdout"] == "api-docker-ok\n"
    assert job["result"]["runtime_metadata"]["provider_runtime"] == "docker-python"
    assert job["result"]["runtime_metadata"]["filesystem_isolation"] == "docker_container_bind_mount"
    assert job["result"]["runtime_metadata"]["network_policy_enforced"] == "true"
    assert job["artifacts"][0]["title_present"] is True
    assert "title" not in job["artifacts"][0]


def test_local_python_job_endpoint_returns_timeout_code(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")
    monkeypatch.setattr("src.jobs.JOBS_FILE", tmp_path / "jobs.jsonl")

    client = TestClient(api.app)
    response = client.post(
        "/jobs/code/python-local",
        json={
            "entrypoint": "main.py",
            "files": {"main.py": "import time\ntime.sleep(2)"},
            "timeout_s": 1,
        },
    )

    assert response.status_code == 200
    job = response.json()["job"]
    assert job["status"] == "failed"
    assert job["error"]["code"] == "execution_timeout"


def test_local_python_job_endpoint_returns_policy_violation_code(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")
    monkeypatch.setattr("src.jobs.JOBS_FILE", tmp_path / "jobs.jsonl")

    client = TestClient(api.app)
    response = client.post(
        "/jobs/code/python-local",
        json={
            "entrypoint": "main.py",
            "files": {"main.py": "import subprocess\nsubprocess.run(['echo', 'bad'])\n"},
        },
    )

    assert response.status_code == 200
    job = response.json()["job"]
    assert job["status"] == "failed"
    assert job["error"]["code"] == "execution_policy_violation"
    assert job["result"]["runtime_metadata"]["policy_violation"] == "true"


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
            "owner_id": "lab-list-api",
            "owner_label": "List API Lab",
        },
    ).json()["job"]

    listed = client.get("/jobs").json()["jobs"]
    filtered = client.get(
        "/jobs",
        params={
            "q": created["job_id"],
            "status": "failed",
            "kind": "code_execution",
            "owner_id": "lab-list-api",
        },
    ).json()["jobs"]
    raw_request_search = client.get("/jobs", params={"q": "main.py"}).json()["jobs"]
    raw_owner_search = client.get("/jobs", params={"q": "List API Lab"}).json()["jobs"]
    missing = client.get("/jobs", params={"status": "queued"}).json()["jobs"]
    assert listed[0]["job_id"] == created["job_id"]
    assert "request" not in listed[0]
    assert "result" not in listed[0]
    assert "logs" not in listed[0]
    assert "request_id" not in listed[0]
    assert "idempotency_key" not in listed[0]
    assert listed[0]["request_id_present"] is True
    assert listed[0]["idempotency_key_present"] is False
    assert listed[0]["owner_id_present"] is True
    assert listed[0]["owner_label_present"] is True
    assert "owner_id" not in listed[0]
    assert "owner_label" not in listed[0]
    assert "lab-list-api" not in json.dumps(listed, sort_keys=True)
    assert "List API Lab" not in json.dumps(listed, sort_keys=True)
    assert raw_request_search == []
    assert raw_owner_search == []
    assert filtered[0]["job_id"] == created["job_id"]
    assert filtered[0]["owner_id_present"] is True
    assert "owner_id" not in filtered[0]
    assert "owner_label" not in filtered[0]
    assert missing == []

    retried = client.post(f"/jobs/{created['job_id']}/retry").json()["job"]
    assert retried["job_id"] != created["job_id"]
    assert retried["kind"] == "code_execution"
    assert retried["owner_id"] == "lab-list-api"
    assert retried["ownership_source"] == "inherited"

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
        json={"delay_s": 30, "queue_timeout_s": 120},
        headers={"X-Request-ID": "req-scheduled"},
    )

    assert response.status_code == 200
    job = response.json()["job"]
    assert job["status"] == "queued"
    assert job["parent_job_id"] == created.job_id
    assert job["request_id"] == "req-scheduled"
    assert job["not_before"] is not None
    assert job["deadline_at"] is not None


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
        lambda paths, **_kwargs: (object(), 9),
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
            "queue_timeout_s": 120,
            "owner_id": "lab-async-api",
            "owner_label": "Async API Lab",
        },
    )

    assert response.status_code == 200
    job = response.json()["job"]
    assert job["status"] == "queued"
    assert job["owner_id"] == "lab-async-api"
    assert job["owner_label"] == "Async API Lab"
    assert job["deadline_at"] is not None
    manager._queue.join()
    assert store.get(job["job_id"]).status == "succeeded"


def test_async_python_job_endpoint_reuses_idempotency_key(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")
    store = api.LocalJobStore(tmp_path / "jobs.jsonl")
    manager = AsyncJobManager(store)
    monkeypatch.setattr(api, "get_async_job_manager", lambda: manager)

    client = TestClient(api.app)
    payload = {
        "entrypoint": "main.py",
        "files": {"main.py": "print('idempotent-api-ok')"},
        "idempotency_key": "idem-python-api",
    }
    first = client.post("/jobs/async/code/python-local", json=payload).json()["job"]
    second = client.post("/jobs/async/code/python-local", json=payload).json()["job"]

    assert second["job_id"] == first["job_id"]
    assert second["idempotency_key"] == "idem-python-api"
    assert len(store.list_latest(limit=10)) == 1
    manager._queue.join()


def test_async_python_job_endpoint_applies_retry_policy_until_dead_lettered(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")
    store = api.LocalJobStore(tmp_path / "jobs.jsonl")
    manager = AsyncJobManager(store, worker_id="api-retry")
    monkeypatch.setattr(api, "get_async_job_manager", lambda: manager)

    client = TestClient(api.app)
    response = client.post(
        "/jobs/async/code/python-local",
        json={
            "entrypoint": "main.py",
            "files": {"main.py": "raise SystemExit(6)"},
            "max_attempts": 2,
            "retry_backoff_s": 0,
        },
    )

    assert response.status_code == 200
    job = response.json()["job"]
    assert job["status"] == "queued"
    assert job["max_attempts"] == 2
    assert job["retry_backoff_s"] == 0
    assert job["dead_lettered_at"] is None
    manager._queue.join()
    dead = store.get(job["job_id"])
    assert dead.status == "dead_lettered"
    assert dead.attempts == 2
    assert dead.dead_lettered_at is not None


def test_immediate_python_job_endpoint_reuses_idempotency_key(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")
    monkeypatch.setattr("src.jobs.JOBS_FILE", tmp_path / "jobs.jsonl")

    client = TestClient(api.app)
    payload = {
        "entrypoint": "main.py",
        "files": {"main.py": "print('idempotent-now')"},
        "idempotency_key": "idem-python-now",
    }
    first = client.post("/jobs/code/python-local", json=payload).json()["job"]
    second = client.post("/jobs/code/python-local", json=payload).json()["job"]

    assert first["status"] == "succeeded"
    assert second["job_id"] == first["job_id"]
    assert second["idempotency_key"] == "idem-python-now"
    assert len(api.LocalJobStore().list_latest(limit=10)) == 1


def test_immediate_python_job_endpoint_preserves_distinct_jobs_for_different_or_missing_keys(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")
    monkeypatch.setattr("src.jobs.JOBS_FILE", tmp_path / "jobs.jsonl")

    client = TestClient(api.app)
    base_payload = {
        "entrypoint": "main.py",
        "files": {"main.py": "print('distinct-now')"},
    }
    first = client.post(
        "/jobs/code/python-local",
        json={**base_payload, "idempotency_key": "idem-python-one"},
    ).json()["job"]
    second = client.post(
        "/jobs/code/python-local",
        json={**base_payload, "idempotency_key": "idem-python-two"},
    ).json()["job"]
    third = client.post("/jobs/code/python-local", json=base_payload).json()["job"]
    fourth = client.post("/jobs/code/python-local", json=base_payload).json()["job"]

    assert len({first["job_id"], second["job_id"], third["job_id"], fourth["job_id"]}) == 4
    assert third["idempotency_key"] is None
    assert fourth["idempotency_key"] is None
    assert len(api.LocalJobStore().list_latest(limit=10)) == 4


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
