import io
import json

import pytest
from pathlib import Path
from fastapi import HTTPException
from fastapi.testclient import TestClient

import api
from src.jobs import AsyncJobManager
from src.metadata import PaperRecord, safe_corpus_profile_report_filename
from src.artifacts import artifact_id_for_uri
from src.runtime import ProviderQuotaGuardError, RuntimeEvent
from src.users import LocalUserStore


@pytest.fixture(autouse=True)
def no_job_runtime_event_disk_writes(monkeypatch):
    monkeypatch.setattr("src.jobs.append_runtime_event", lambda **_kwargs: None)
    monkeypatch.setattr(api, "API_RATE_LIMIT_ENABLED", False)
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


def test_request_validation_error_reports_invalid_field(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")
    secret_value = "sk-secret-validation-" + ("x" * 150)

    response = TestClient(api.app).post(
        "/jobs/index/rebuild",
        json={
            "source_paths": ["papers/library/paper.pdf"],
            "idempotency_key": secret_value,
        },
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail[0]["type"] == "string_too_long"
    assert detail[0]["loc"] == ["body", "idempotency_key"]


def test_request_validation_error_reports_nested_field(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")

    response = TestClient(api.app).post(
        "/jobs/code/python-local",
        json={
            "entrypoint": "main.py",
            "files": {"main.py": ["print('sk-secret-code')"]},
        },
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail[0]["type"] == "string_type"
    assert detail[0]["loc"] == ["body", "files", "main.py"]


def test_api_token_status_does_not_return_token_values(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "secret")

    status = api.api_token_status("Bearer secret", "wrong")

    assert status == {
        "token_status": "valid",
        "credential_type": "multiple",
        "credential_present": True,
        "auth_configured": True,
        "auth_source": "static_token",
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


def test_api_rate_limit_middleware_blocks_after_configured_threshold(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")
    monkeypatch.setattr(api, "API_RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr(api, "API_RATE_LIMIT_MAX_REQUESTS", 2)
    monkeypatch.setattr(api, "API_RATE_LIMIT_WINDOW_S", 60)
    api._API_RATE_LIMIT_BUCKETS.clear()
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
    assert api.startup_warmup_status() == {
        "status": "failed",
        "ready": False,
        "error": "corrupt index",
    }


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
    assert response.json() == {
        "answer": "explanation: Explain SMC",
        "request_id": "req-test",
        "history_recorded": False,
    }
    assert usage_events[0]["kind"] == "query_usage"
    assert usage_events[0]["request_id"] == "req-test"
    assert usage_events[0]["metadata"]["endpoint"] == "/query"
    assert usage_events[0]["metadata"]["estimated_total_tokens"] > 0
    assert usage_events[0]["metadata"]["usage_source"] == "provider"
    assert usage_events[0]["metadata"]["cost_source"] == "not_configured"
    assert usage_events[0]["metadata"]["provider_total_tokens"] == 15
    assert usage_events[0]["metadata"]["duration_ms"] >= 0
    assert usage_events[0]["metadata"]["owner_id"] == "lab-query"
    assert usage_events[0]["metadata"]["owner_label"] == "Query Lab"
    assert usage_events[0]["metadata"]["ownership_source"] == "request"
    assert "Explain SMC" not in str(usage_events[0]["metadata"])


def test_generated_query_endpoints_record_local_user_history(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")
    store = LocalUserStore(tmp_path / "users.sqlite3")
    store.create_initial_admin(
        user_id="admin",
        display_name="Admin",
        password="admin-pass",
    )
    monkeypatch.setattr(api, "LocalUserStore", lambda: store)
    monkeypatch.setattr(api, "append_runtime_event", lambda **_kwargs: None)

    class FakeResult:
        answer = "Grounded answer [1]."
        answer_mode = "explanation"
        provider_usage = None
        context_refs = []

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

        def to_dict(self):
            return {
                "answer": self.answer,
                "answer_mode": self.answer_mode,
                "citation_validation": self.citation_validation.to_dict(),
                "context_refs": self.context_refs,
            }

    monkeypatch.setattr(
        api,
        "query_with_metadata",
        lambda question, *, answer_mode: FakeResult(),
    )

    client = TestClient(api.app)
    response = client.post(
        "/query",
        json={
            "question": "Explain SMC",
            "answer_mode": "explanation",
            "user_id": "admin",
        },
    )
    inspect_response = client.post(
        "/query/inspect",
        json={"question": "Inspect SMC", "user_id": "admin"},
    )
    report_response = client.post(
        "/query/report",
        json={"question": "Report SMC", "user_id": "admin"},
    )
    history = client.get("/query/history/admin")

    assert response.status_code == 200
    assert response.json()["history_recorded"] is True
    assert inspect_response.status_code == 200
    assert inspect_response.json()["history_recorded"] is True
    assert report_response.status_code == 200
    assert report_response.headers["X-FluxMind-History-Recorded"] == "true"
    assert history.status_code == 200
    assert history.json()["user"]["user_id"] == "admin"
    assert [entry["question"] for entry in history.json()["history"]] == [
        "Report SMC",
        "Inspect SMC",
        "Explain SMC",
    ]
    assert history.json()["history"][0]["answer"] == "Grounded answer [1]."


def test_query_rejects_unknown_history_user_before_provider(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "API_TOKEN", "")
    store = LocalUserStore(tmp_path / "users.sqlite3")
    store.create_initial_admin(
        user_id="admin",
        display_name="Admin",
        password="admin-pass",
    )
    monkeypatch.setattr(api, "LocalUserStore", lambda: store)

    def fail_query(*_args, **_kwargs):
        raise AssertionError("provider should not be called")

    monkeypatch.setattr(api, "query_with_metadata", fail_query)

    response = TestClient(api.app).post(
        "/query",
        json={"question": "Explain SMC", "user_id": "missing-user"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Local query-history user not found"


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


def test_query_exception_event_uses_compact_ownership_metadata(monkeypatch):
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
    assert metadata["owner_id"] == "secret-owner"
    assert metadata["owner_label"] == "Secret Owner"
    assert metadata["ownership_source"] == "request"


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


def test_corpus_structure_endpoint_explains_invalid_source(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")

    def fail(_source_paths):
        raise ValueError("PDF path is not in the selectable corpus: /tmp/sk-secret-paper.pdf")

    monkeypatch.setattr(api, "resolve_selectable_source_paths", fail)

    client = TestClient(api.app)
    response = client.get(
        "/corpus/structure",
        params={"source_path": "/tmp/sk-secret-paper.pdf"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "PDF path is not in the selectable corpus: /tmp/sk-secret-paper.pdf"
    )


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
        raise ValueError("PDF path is not in the selectable corpus: /tmp/sk-secret-missing.pdf")

    monkeypatch.setattr(api, "set_active_paper_source_paths", fail)

    client = TestClient(api.app)
    response = client.put("/corpus/active", json={"source_paths": ["/tmp/sk-secret-missing.pdf"]})

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "PDF path is not in the selectable corpus: /tmp/sk-secret-missing.pdf"
    )


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
    assert payload["job"]["owner_label"] == "Local user"
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


def test_corpus_profile_activate_and_rebuild_explain_invalid_saved_path(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")
    store = api.CorpusProfileStore(tmp_path / "corpus_profiles.json")
    store.upsert_profile(
        profile_id="secret-profile",
        name="Secret Profile",
        source_paths=["/tmp/sk-secret-profile.pdf"],
    )

    def fail(_source_paths):
        raise ValueError("PDF path is not in the selectable corpus: /tmp/sk-secret-profile.pdf")

    monkeypatch.setattr(api, "CorpusProfileStore", lambda: store)
    monkeypatch.setattr(api, "set_active_paper_source_paths", fail)

    client = TestClient(api.app)
    activate_response = client.post("/corpus/profiles/secret-profile/activate")
    rebuild_response = client.post("/corpus/profiles/secret-profile/rebuild", json={})

    for response in (activate_response, rebuild_response):
        assert response.status_code == 400
        assert response.json()["detail"] == (
            "PDF path is not in the selectable corpus: /tmp/sk-secret-profile.pdf"
        )


def test_corpus_profile_status_and_report_endpoints(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")
    source = tmp_path / "papers" / "library" / "paper.pdf"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"pdf")
    store = api.CorpusProfileStore(tmp_path / "corpus_profiles.json")
    store.upsert_profile(
        profile_id="smc-core",
        name="SMC Core",
        source_paths=["papers/library/paper.pdf"],
    )

    class FakeChunkStore:
        def source_paths(self):
            return ["papers/library/paper.pdf"]

    monkeypatch.setattr("src.admin.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("src.admin.CorpusProfileStore", lambda: store)
    monkeypatch.setattr("src.admin.ChunkMetadataStore", FakeChunkStore)
    monkeypatch.setattr("src.admin.load_active_paper_paths", lambda: [source])

    client = TestClient(api.app)
    status_response = client.get("/corpus/profiles/smc-core/status")
    report_response = client.get("/corpus/profiles/smc-core/report")

    assert status_response.status_code == 200
    status = status_response.json()["status"]
    assert status["profile"]["profile_id"] == "smc-core"
    assert status["active"] is True
    assert status["indexed"] is True
    assert status["rebuild_required"] is False
    assert report_response.status_code == 200
    assert "# FluxMind Corpus Profile" in report_response.text
    assert "`papers/library/paper.pdf`" in report_response.text


def test_corpus_profile_report_filename_is_header_safe():
    filename = safe_corpus_profile_report_filename(
        'abc"\r\nContent-Disposition: x-secret'
    )

    assert filename.startswith("fluxmind-corpus-profile-")
    assert filename.endswith(".md")
    assert "abc-content-disposition-x-secret" in filename
    assert '"' not in filename
    assert "\r" not in filename
    assert "\n" not in filename


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
        raise ValueError("PDF path is not in the selectable corpus: /tmp/sk-secret-missing.pdf")

    monkeypatch.setattr(api, "validate_corpus_profile_source_paths", fail)

    client = TestClient(api.app)
    response = client.post(
        "/corpus/profiles",
        json={"name": "Bad", "source_paths": ["/tmp/sk-secret-missing.pdf"]},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "PDF path is not in the selectable corpus: /tmp/sk-secret-missing.pdf"
    )


def test_admin_status_endpoint_returns_runtime_status(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")

    class FakeStatus:
        def to_dict(self):
            return {
                "jobs": {"total": 0},
                "config": {
                    "llm_model": "test-model",
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
        assert q == "req-events"
        assert limit == 25
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
                    },
                    "latest": [],
                },
                "runtime_dirs": [],
                "config": {
                    "llm_model": "test-model",
                    "embedding_model": "test-embed",
                    "llm_base_url_configured": True,
                },
            }

    monkeypatch.setattr(api, "collect_admin_status", lambda: FakeStatus())

    client = TestClient(api.app)
    response = client.get("/admin/status/report")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/markdown")
    assert "fluxmind-admin-status.md" in response.headers["content-disposition"]
    assert "# FluxMind Runtime Status" in response.text
    assert "test-model" in response.text
    assert "api_key" not in response.text.lower()


def test_admin_metrics_endpoint_returns_metrics(monkeypatch):
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
                "config": {
                    "api_rate_limit_enabled": False,
                    "retention_delete_enabled": False,
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
    assert "fluxmind_corpus_rebuild_required 0" in response.text


def test_admin_runtime_manifest_endpoint_returns_manifest(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")

    manifest = {
        "mode": "local_runtime_backup_manifest",
        "env_file_present": True,
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
    assert payload["env_file_present"] is True


def test_admin_runtime_manifest_report_endpoint_returns_markdown(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")

    manifest = {
        "mode": "local_runtime_backup_manifest",
        "env_file_present": True,
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
    assert "Inventory of runtime paths, sizes, and hashes" in response.text


def test_admin_runtime_manifest_restore_check_endpoint_returns_check(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")

    manifest = {"mode": "local_runtime_backup_manifest", "groups": []}

    def fake_restore_check(received_manifest):
        assert received_manifest == manifest
        return {
            "mode": "local_runtime_restore_dry_run",
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
    assert payload["ok"] is True


def test_admin_runtime_manifest_restore_check_report_endpoint_returns_markdown(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")

    manifest = {"mode": "local_runtime_backup_manifest", "groups": []}
    restore_check = {
        "mode": "local_runtime_restore_dry_run",
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
    artifact_data = listed.json()["artifacts"][0]
    assert artifact_data["artifact_id"] == artifact_id
    assert artifact_data["title"] == "result.txt"
    assert artifact_data["owner_id"] == "lab-artifact-api"
    assert artifact_data["owner_label"] == "Artifact API Lab"
    assert artifact_data["metadata"]["provider"] == "local"
    assert artifact_data["metadata"]["prompt"] == "private artifact prompt"
    assert artifact_data["metadata"]["reference_uris"] == ["paper://private#page=4"]
    assert "uri" not in artifact_data
    assert filtered.status_code == 200
    assert filtered.json()["artifacts"][0]["artifact_id"] == artifact_id
    assert missing.status_code == 200
    assert missing.json()["artifacts"] == []
    assert downloaded.status_code == 200
    assert downloaded.text == "artifact-body"
    assert "result.txt" in downloaded.headers["content-disposition"]


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


def test_artifact_download_explains_invalid_uri(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")
    uri = "file://secret-host/tmp/sk-secret-artifact.txt"
    store = api.LocalJobStore(tmp_path / "jobs.jsonl")
    store.append_new(
        api.JobRecord(
            job_id="job-invalid-artifact-uri",
            kind="code_execution",
            status="succeeded",
            created_at="2026-06-20T00:00:00+00:00",
            updated_at="2026-06-20T00:00:00+00:00",
            request={},
            artifacts=[
                {
                    "kind": "text",
                    "uri": uri,
                    "mime_type": "text/plain",
                    "title": "sk-secret-artifact.txt",
                }
            ],
        )
    )
    monkeypatch.setattr("src.artifacts.LocalJobStore", lambda: store)

    client = TestClient(api.app)
    response = client.get(f"/artifacts/{artifact_id_for_uri(uri)}")

    assert response.status_code == 400
    assert response.json()["detail"] == "Only local file artifacts can be exported."


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
    assert job["artifacts"][0]["metadata"]["provider"] == "local"
    assert job["artifacts"][0]["metadata"]["diagram_template"] == "sliding-mode-observer"
    assert job["artifacts"][0]["metadata"]["reference_uris"] == []
    assert job["artifacts"][0]["metadata"]["cost_estimate_usd"] == "0"
    assert "uri" not in job["artifacts"][0]


def test_configured_image_job_endpoint_returns_persisted_job(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")
    monkeypatch.setattr("src.jobs.IMAGE_PROVIDER_BACKEND", "local-mock")
    monkeypatch.setattr("src.jobs.JOBS_FILE", tmp_path / "jobs.jsonl")
    monkeypatch.setattr("src.providers.ARTIFACTS_DIR", tmp_path / "artifacts")

    client = TestClient(api.app)
    response = client.post(
        "/jobs/image",
        json={
            "prompt": "Draw a PMSM control loop",
            "diagram_template": "pmsm-control-loop",
        },
        headers={"X-Request-ID": "req-image-configured"},
    )

    assert response.status_code == 200
    job = response.json()["job"]
    assert job["kind"] == "image_generation"
    assert job["status"] == "succeeded"
    assert job["request_id"] == "req-image-configured"
    assert job["artifacts"][0]["mime_type"] == "image/svg+xml"
    assert job["artifacts"][0]["metadata"]["diagram_template"] == "pmsm-control-loop"


def test_configured_image_job_endpoint_requires_openai_key(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")
    monkeypatch.setattr(api, "IMAGE_PROVIDER_BACKEND", "openai")
    monkeypatch.setattr(api, "OPENAI_IMAGE_API_KEY", "")

    client = TestClient(api.app)
    response = client.post(
        "/jobs/image",
        json={
            "prompt": "Draw a PMSM control loop",
            "diagram_template": "pmsm-control-loop",
        },
    )

    assert response.status_code == 409
    assert "OPENAI_IMAGE_API_KEY" in response.json()["detail"]


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
    assert job["request"]["entrypoint"] == "main.py"
    assert job["request"]["files"] == {"main.py": "print('api-job-ok')"}
    assert job["result"]["stdout"] == "api-job-ok\n"
    assert job["result"]["stderr"] == ""
    assert job["result"]["runtime_metadata"]["memory_mb"] == "512"
    assert job["result"]["runtime_metadata"]["entrypoint"] == "main.py"


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
    assert job["error"]["message"] == (
        "Execution input file could not be materialized: main.py/helper.txt"
    )
    assert "main.py/helper.txt" in job["result"]["stderr"]
    assert "/tmp/fluxmind-" not in response.text


def test_local_python_job_endpoint_uses_configured_docker_backend(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")
    monkeypatch.setattr("src.jobs.JOBS_FILE", tmp_path / "jobs.jsonl")
    monkeypatch.setattr("src.jobs.CODE_EXECUTION_BACKEND", "docker")
    monkeypatch.setattr("src.jobs.DOCKER_EXECUTION_IMAGE", "python:3.12-slim")
    monkeypatch.setattr("src.providers.shutil.which", lambda _name: "/usr/bin/docker")

    class FakePopen:
        returncode = 0

        def __init__(self, command, **_kwargs):
            self.stdout = io.BytesIO(b"api-docker-ok\n")
            self.stderr = io.BytesIO()
            mount = command[command.index("-v") + 1]
            workdir = Path(mount.split(":", 1)[0])
            (workdir / "api-docker.txt").write_text("api-docker-artifact", encoding="utf-8")

        def poll(self):
            return self.returncode

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
    assert job["result"]["stderr"] == ""
    assert job["result"]["runtime_metadata"]["provider_runtime"] == "docker-python"
    assert job["result"]["runtime_metadata"]["filesystem_isolation"] == "docker_container_bind_mount"
    assert job["result"]["runtime_metadata"]["network_policy_enforced"] == "true"
    assert job["artifacts"][0]["title"] == "api-docker.txt"


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


def test_job_response_returns_code_request_and_execution_output(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")
    monkeypatch.setattr("src.jobs.JOBS_FILE", tmp_path / "jobs.jsonl")

    client = TestClient(api.app)
    secret = "sk-secret-job-output"
    response = client.post(
        "/jobs/code/python-local",
        json={
            "entrypoint": "secret-entrypoint.py",
            "files": {"secret-entrypoint.py": f"raise RuntimeError('{secret}')"},
        },
    )

    assert response.status_code == 200
    job = response.json()["job"]
    assert job["request"]["entrypoint"] == "secret-entrypoint.py"
    assert secret in job["request"]["files"]["secret-entrypoint.py"]
    assert secret in job["result"]["stderr"]
    assert job["error"]["code"] == "execution_failed"


def test_job_response_reuses_and_returns_idempotency_key(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")
    monkeypatch.setattr("src.jobs.JOBS_FILE", tmp_path / "jobs.jsonl")

    client = TestClient(api.app)
    idempotency_key = "paper-code-run-12345678"
    payload = {
        "entrypoint": "main.py",
        "files": {"main.py": "print('idempotent-secret-ok')"},
        "idempotency_key": idempotency_key,
    }

    first_response = client.post("/jobs/code/python-local", json=payload)
    second_response = client.post("/jobs/code/python-local", json=payload)

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    first = first_response.json()["job"]
    second = second_response.json()["job"]
    assert second["job_id"] == first["job_id"]
    assert second["idempotency_key"] == idempotency_key

    loaded_response = client.get(f"/jobs/{first['job_id']}")
    assert loaded_response.status_code == 200
    loaded = loaded_response.json()["job"]
    assert loaded["idempotency_key"] == idempotency_key


def test_job_response_returns_and_inherits_owner_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")
    monkeypatch.setattr("src.jobs.JOBS_FILE", tmp_path / "jobs.jsonl")

    client = TestClient(api.app)
    owner_id = "sk-secret-owner-id-12345678"
    owner_label = "Bearer secret-owner-label-token"
    response = client.post(
        "/jobs/code/python-local",
        json={
            "entrypoint": "main.py",
            "files": {"main.py": "raise SystemExit(3)"},
            "owner_id": owner_id,
            "owner_label": owner_label,
        },
    )

    assert response.status_code == 200
    job = response.json()["job"]
    assert job["owner_id"] == owner_id
    assert job["owner_label"] == owner_label

    loaded_response = client.get(f"/jobs/{job['job_id']}")
    assert loaded_response.status_code == 200
    loaded = loaded_response.json()["job"]
    assert loaded["owner_id"] == owner_id
    assert loaded["owner_label"] == owner_label

    retry_response = client.post(f"/jobs/{job['job_id']}/retry")
    assert retry_response.status_code == 200
    retried = retry_response.json()["job"]
    assert retried["job_id"] != job["job_id"]
    assert retried["ownership_source"] == "inherited"
    assert retried["owner_id"] == owner_id
    assert retried["owner_label"] == owner_label


def test_job_response_normalizes_invalid_request_id(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")
    monkeypatch.setattr("src.jobs.JOBS_FILE", tmp_path / "jobs.jsonl")

    secret_request_id = "Bearer secret-request-token"
    store = api.LocalJobStore(tmp_path / "jobs.jsonl")
    store.append_new(
        api.JobRecord(
            job_id="job-unsafe-request-id",
            kind="code_execution",
            status="failed",
            created_at="2026-06-20T00:00:00+00:00",
            updated_at="2026-06-20T00:00:00+00:00",
            request={
                "language": "python",
                "entrypoint": "main.py",
                "files": {"main.py": "raise SystemExit(3)"},
            },
            result={"stdout": "", "stderr": "secret traceback", "exit_code": 3},
            error={"code": "execution_failed", "message": "secret traceback"},
            request_id=secret_request_id,
            owner_id="local-user",
            owner_label="Local user",
        )
    )

    client = TestClient(api.app)
    loaded_response = client.get("/jobs/job-unsafe-request-id")

    assert loaded_response.status_code == 200
    assert secret_request_id not in loaded_response.text
    loaded = loaded_response.json()["job"]
    assert loaded["request_id"] is None
    assert loaded["error"]["message"] == "secret traceback"

    listed_response = client.get("/jobs")
    assert listed_response.status_code == 200
    assert secret_request_id not in listed_response.text
    listed = listed_response.json()["jobs"][0]
    assert listed["request_id"] is None

    retry_response = client.post("/jobs/job-unsafe-request-id/retry")
    assert retry_response.status_code == 200
    assert secret_request_id not in retry_response.text
    retried = retry_response.json()["job"]
    assert retried["request_id"]
    assert secret_request_id != retried["request_id"]


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
    assert listed[0]["request_id"]
    assert listed[0]["idempotency_key"] is None
    assert listed[0]["owner_id"] == "lab-list-api"
    assert listed[0]["owner_label"] == "List API Lab"
    assert raw_request_search[0]["job_id"] == created["job_id"]
    assert raw_owner_search[0]["job_id"] == created["job_id"]
    assert filtered[0]["job_id"] == created["job_id"]
    assert filtered[0]["owner_id"] == "lab-list-api"
    assert filtered[0]["owner_label"] == "List API Lab"
    assert missing == []

    retried = client.post(f"/jobs/{created['job_id']}/retry").json()["job"]
    assert retried["job_id"] != created["job_id"]
    assert retried["kind"] == "code_execution"
    assert retried["owner_id"] == "lab-list-api"
    assert retried["owner_label"] == "List API Lab"
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
    assert job["request"] == {"source_paths": ["papers/library/paper.pdf"]}
    assert job["result"]["source_paths"] == ["papers/library/paper.pdf"]
    assert job["result"]["chunk_count"] == 9


def test_index_rebuild_job_endpoint_reports_invalid_source_path(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")
    monkeypatch.setattr("src.jobs.JOBS_FILE", tmp_path / "jobs.jsonl")
    monkeypatch.setattr("src.jobs.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("src.jobs.ingestion.discover_pdfs", lambda: [])

    client = TestClient(api.app)
    response = client.post(
        "/jobs/index/rebuild",
        json={"source_paths": ["/tmp/sk-secret-index.pdf"]},
    )

    assert response.status_code == 200
    job = response.json()["job"]
    assert job["kind"] == "index_rebuild"
    assert job["status"] == "failed"
    assert job["request"] == {"source_paths": ["/tmp/sk-secret-index.pdf"]}
    assert job["result"] is None


def test_async_index_rebuild_job_response_returns_source_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")
    monkeypatch.setattr("src.jobs.JOBS_FILE", tmp_path / "jobs.jsonl")
    manager = AsyncJobManager(api.LocalJobStore())
    monkeypatch.setattr(api, "get_async_job_manager", lambda: manager)

    client = TestClient(api.app)
    response = client.post(
        "/jobs/async/index/rebuild",
        json={"source_paths": ["/tmp/sk-secret-async.pdf"]},
    )

    assert response.status_code == 200
    job = response.json()["job"]
    assert job["kind"] == "index_rebuild"
    assert job["status"] == "queued"
    assert job["request"] == {"source_paths": ["/tmp/sk-secret-async.pdf"]}
    assert api.LocalJobStore().get(job["job_id"]).request == {
        "source_paths": ["/tmp/sk-secret-async.pdf"]
    }

    loaded = client.get(f"/jobs/{job['job_id']}")
    assert loaded.status_code == 200
    assert loaded.json()["job"]["request"] == {
        "source_paths": ["/tmp/sk-secret-async.pdf"]
    }


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
