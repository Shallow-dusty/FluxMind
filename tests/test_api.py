import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import api
from src.jobs import AsyncJobManager
from src.metadata import PaperRecord
from src.artifacts import artifact_id_for_uri
from src.runtime import RuntimeEvent


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
        json={"question": "Explain SMC"},
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
    assert usage_events[0]["metadata"]["provider_total_tokens"] == 15
    assert "Explain SMC" not in str(usage_events[0]["metadata"])


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
    assert "# FluxMind Query Report" in response.text
    assert "Report answer [1]" in response.text
    assert "papers/library/paper.pdf" in response.text
    assert usage_events[0]["kind"] == "query_usage"
    assert usage_events[0]["metadata"]["endpoint"] == "/query/report"
    assert usage_events[0]["metadata"]["citation_ok"] is True


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
    assert "req-history" in events


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
        def enqueue_index_rebuild(self, source_paths, *, queue_timeout_s=None, request_id=None):
            queued["source_paths"] = source_paths
            queued["queue_timeout_s"] = queue_timeout_s
            queued["request_id"] = request_id
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
    assert activated["source_paths"] == ["papers/library/paper.pdf"]
    assert queued == {
        "source_paths": ["papers/library/paper.pdf"],
        "queue_timeout_s": 300,
        "request_id": "req-profile-rebuild",
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
    filtered = client.get(
        "/artifacts",
        params={"q": "result", "kind": "text", "job_kind": "code_execution"},
    )
    missing = client.get("/artifacts", params={"kind": "image"})
    artifact_id = artifact_id_for_uri(uri)
    downloaded = client.get(f"/artifacts/{artifact_id}")

    assert listed.status_code == 200
    assert listed.json()["artifacts"][0]["artifact_id"] == artifact_id
    assert filtered.status_code == 200
    assert filtered.json()["artifacts"][0]["artifact_id"] == artifact_id
    assert missing.status_code == 200
    assert missing.json()["artifacts"] == []
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
    filtered = client.get(
        "/jobs",
        params={"q": "main.py", "status": "failed", "kind": "code_execution"},
    ).json()["jobs"]
    missing = client.get("/jobs", params={"status": "queued"}).json()["jobs"]
    assert listed[0]["job_id"] == created["job_id"]
    assert filtered[0]["job_id"] == created["job_id"]
    assert missing == []

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
        },
    )

    assert response.status_code == 200
    job = response.json()["job"]
    assert job["status"] == "queued"
    assert job["deadline_at"] is not None
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
