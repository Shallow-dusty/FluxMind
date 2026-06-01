from pathlib import Path

from langchain_core.documents import Document

from src.chain import (
    ANSWER_MODE_INSTRUCTIONS,
    bm25_relevance_scores,
    citation_instruction,
    format_context,
    generated_artifact_context,
    hybrid_retrieve,
    learned_rerank_documents,
    neutralize_invalid_numbered_citations,
    normalize_answer_mode,
    provider_usage_from_response,
    query_with_metadata,
    rerank_documents,
    retrieve_with_metadata,
    tokenize_query,
    validate_numbered_citations,
)
from src.evaluation import (
    answer_term_coverage,
    build_evaluation_report,
    evaluate_config,
    evaluate_case,
    evaluate_live_config,
    evaluate_live_query_payload,
    evaluate_live_retrieval_config,
    evaluate_live_retrieval_payload,
    evaluate_recorded_answer,
    load_eval_config,
)


def test_answer_mode_normalization_and_context_refs():
    assert normalize_answer_mode("derivation") == "derivation"
    assert normalize_answer_mode("unknown") == "explanation"
    assert "code_generation" in ANSWER_MODE_INSTRUCTIONS

    context = format_context(
        [
            Document(
                page_content="observer text",
                metadata={"source": "paper.pdf", "page": 3},
            )
        ]
    )

    assert "[1] Source: paper.pdf, Page 3" in context


def test_citation_instruction_bounds_numbered_refs():
    assert citation_instruction(0) == "No numbered source refs are available; do not use numbered citations like [1]."
    assert citation_instruction(1) == "Valid numbered source refs for this answer: [1] only."
    assert citation_instruction(3) == "Valid numbered source refs for this answer: [1] through [3] only."
    assert "measured currents" in ANSWER_MODE_INSTRUCTIONS["code_generation"]


def test_numbered_citation_validation_rejects_unknown_refs():
    docs = [
        Document(page_content="a", metadata={"source": "a.pdf", "page": 1}),
        Document(page_content="b", metadata={"source": "b.pdf", "page": 2}),
    ]

    result = validate_numbered_citations("Use [1] and [3].", docs, required_refs=[1, 2])

    assert not result.ok
    assert result.valid_refs == [1]
    assert result.invalid_refs == [3]
    assert result.missing_required_refs == [2]


def test_invalid_numbered_citations_are_neutralized_before_validation():
    answer = neutralize_invalid_numbered_citations("Use context [1] and paper ref [104].", max_ref=2)

    assert "[1]" in answer
    assert "[104]" not in answer
    assert "(source-internal ref 104)" in answer


def test_numbered_citation_validation_requires_source_page_metadata():
    docs = [
        Document(page_content="a", metadata={"source": "a.pdf"}),
        Document(page_content="b", metadata={"source": "b.pdf", "page": 2}),
    ]

    result = validate_numbered_citations("Use [1] and [2].", docs)

    assert not result.ok
    assert result.missing_source_page_refs == [1]


def test_query_with_metadata_validates_generated_answer_citations(monkeypatch):
    docs = [
        Document(
            page_content="sliding mode observer text",
            metadata={"source": "paper.pdf", "source_path": "papers/library/paper.pdf", "page": 3},
        )
    ]

    class FakePrompt:
        def __or__(self, _llm):
            class FakeChain:
                def invoke(self, _payload):
                    return type("Response", (), {"content": "Grounded answer [1]."})()

            return FakeChain()

    monkeypatch.setattr("src.chain.hybrid_retrieve", lambda _question, *, k: docs)
    monkeypatch.setattr("src.chain.generated_artifact_context", lambda: "")
    monkeypatch.setattr("src.chain.get_llm", lambda: object())
    monkeypatch.setattr("src.chain.ChatPromptTemplate.from_messages", lambda _messages: FakePrompt())

    result = query_with_metadata("Explain SMC", answer_mode="literature_review")

    assert result.answer == "Grounded answer [1]."
    assert result.answer_mode == "literature_review"
    assert result.citation_validation.ok
    assert result.context_refs[0]["source_path"] == "papers/library/paper.pdf"
    assert result.to_dict()["citation_validation"]["ok"] is True


def test_provider_usage_from_response_reads_langchain_usage_metadata():
    response = type(
        "Response",
        (),
        {
            "usage_metadata": {
                "input_tokens": 12,
                "output_tokens": 8,
                "total_tokens": 20,
            }
        },
    )()

    assert provider_usage_from_response(response) == {
        "prompt_tokens": 12,
        "completion_tokens": 8,
        "total_tokens": 20,
    }


def test_provider_usage_from_response_reads_token_usage_metadata():
    response = type(
        "Response",
        (),
        {
            "response_metadata": {
                "token_usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 7,
                }
            }
        },
    )()

    assert provider_usage_from_response(response) == {
        "prompt_tokens": 5,
        "completion_tokens": 7,
        "total_tokens": 12,
    }


def test_retrieve_with_metadata_reports_source_page_quality(monkeypatch):
    docs = [
        Document(
            page_content="sliding mode observer text",
            metadata={"source": "paper.pdf", "source_path": "papers/library/paper.pdf", "page": 3},
        ),
        Document(page_content="missing page", metadata={"source": "missing.pdf"}),
    ]

    monkeypatch.setattr("src.chain.hybrid_retrieve", lambda _question, *, k: docs[:k])

    diagnostics = retrieve_with_metadata("Explain SMC", answer_mode="derivation", k=2)

    assert diagnostics.answer_mode == "derivation"
    assert diagnostics.context_count == 2
    assert diagnostics.context_refs[0]["source_path"] == "papers/library/paper.pdf"
    assert diagnostics.missing_source_page_refs == [2]
    assert diagnostics.ok is False
    assert diagnostics.to_dict()["citation_instruction"] == "Valid numbered source refs for this answer: [1] through [2] only."


def test_hybrid_retrieve_reranks_keyword_hits(monkeypatch):
    vector_doc = Document(
        page_content="sliding mode reaching law",
        metadata={"source": "vector.pdf", "page": 1},
    )
    keyword_doc = Document(
        page_content="flux linkage observer uses MRAS estimation",
        metadata={"source": "keyword.pdf", "page": 2},
    )

    class FakeDocstore:
        _dict = {"v": vector_doc, "k": keyword_doc}

    class FakeStore:
        docstore = FakeDocstore()

        def similarity_search(self, _question, *, k):
            assert k == 8
            return [vector_doc][:k]

    monkeypatch.setattr("src.chain.get_vector_store", lambda: FakeStore())

    docs = hybrid_retrieve("flux observer", k=2)

    assert docs == [keyword_doc, vector_doc]


def test_bm25_relevance_scores_use_term_frequency_and_metadata():
    repeated = Document(
        page_content="flux flux flux observer",
        metadata={"source": "a.pdf", "topic": "motor"},
    )
    metadata_hit = Document(
        page_content="unrelated text",
        metadata={"source": "b.pdf", "topic": "flux observer"},
    )
    miss = Document(page_content="sliding mode reaching law", metadata={"source": "c.pdf"})

    scores = bm25_relevance_scores("flux observer", [repeated, metadata_hit, miss])

    assert scores[0] > scores[1] > scores[2]


def test_rerank_documents_keeps_original_order_for_equal_scores():
    first = Document(page_content="alpha", metadata={"source": "a.pdf"})
    second = Document(page_content="alpha", metadata={"source": "b.pdf"})

    assert rerank_documents("alpha", [first, second], k=2) == [first, second]


def test_rerank_documents_prioritizes_source_diversity():
    first = Document(
        page_content="alpha beta gamma",
        metadata={"source": "a.pdf", "source_path": "papers/a.pdf"},
    )
    second_same_source = Document(
        page_content="alpha beta",
        metadata={"source": "a.pdf", "source_path": "papers/a.pdf"},
    )
    third_other_source = Document(
        page_content="alpha",
        metadata={"source": "b.pdf", "source_path": "papers/b.pdf"},
    )

    assert rerank_documents("alpha beta gamma", [first, second_same_source, third_other_source], k=2) == [
        first,
        third_other_source,
    ]


def test_learned_rerank_documents_uses_optional_local_cross_encoder(tmp_path, monkeypatch):
    weaker = Document(
        page_content="alpha beta",
        metadata={"source": "a.pdf", "source_path": "papers/a.pdf"},
    )
    stronger = Document(
        page_content="gamma delta",
        metadata={"source": "b.pdf", "source_path": "papers/b.pdf"},
    )
    same_source = Document(
        page_content="epsilon zeta",
        metadata={"source": "b.pdf", "source_path": "papers/b.pdf", "page": 2},
    )
    model_dir = tmp_path / "reranker"
    model_dir.mkdir()
    captured_pairs = []

    class FakeCrossEncoder:
        def predict(self, pairs):
            captured_pairs.extend(pairs)
            return [0.1, 0.9, 0.8]

    monkeypatch.setattr("src.chain.RERANKER_MODEL", str(model_dir))
    monkeypatch.setattr("src.chain.get_cross_encoder_reranker", lambda _path: FakeCrossEncoder())

    docs = learned_rerank_documents("flux observer", [weaker, stronger, same_source], k=2)

    assert docs == [stronger, weaker]
    assert captured_pairs[0][0] == "flux observer"
    assert "gamma delta" in captured_pairs[1][1]


def test_learned_rerank_documents_returns_none_without_local_model(monkeypatch):
    docs = [Document(page_content="alpha", metadata={"source": "a.pdf"})]

    monkeypatch.setattr("src.chain.RERANKER_MODEL", "/missing/local/reranker")

    assert learned_rerank_documents("alpha", docs, k=1) is None


def test_generated_artifact_context_formats_recent_artifacts(monkeypatch):
    class FakeRegistry:
        def list_artifacts(self, *, limit):
            assert limit == 5
            from src.artifacts import ArtifactRecord

            return [
                ArtifactRecord(
                    artifact_id="abc123",
                    job_id="job1",
                    job_kind="image_generation",
                    kind="image",
                    uri="file:///tmp/diagram.svg",
                    mime_type="image/svg+xml",
                    title="diagram.svg",
                    metadata={"prompt": "SMC diagram", "style": "engineering"},
                )
            ]

    monkeypatch.setattr("src.chain.LocalArtifactRegistry", FakeRegistry)

    context = generated_artifact_context()

    assert "[Artifact:abc123]" in context
    assert "SMC diagram" in context


def test_tokenize_query_supports_cjk_terms():
    assert "磁链" in tokenize_query("磁链 observer")
    assert "observer" in tokenize_query("磁链 observer")


def test_offline_eval_config_passes():
    config = load_eval_config(Path("eval/rag_baseline.json"))

    case_results, provider_results, recorded_results = evaluate_config(config)

    assert case_results
    assert provider_results
    assert recorded_results
    assert all(result.ok for result in case_results)
    assert all(result.ok for result in provider_results)
    assert all(result.ok for result in recorded_results)


def test_evaluation_report_summarizes_results_without_secrets():
    config = load_eval_config(Path("eval/rag_baseline.json"))
    case_results, provider_results, recorded_results = evaluate_config(config)

    report = build_evaluation_report(
        config,
        case_results=case_results,
        provider_results=provider_results,
        recorded_results=recorded_results,
        eval_file=Path("eval/rag_baseline.json"),
    )

    assert report["schema_version"] == 1
    assert report["eval_file"] == "eval/rag_baseline.json"
    assert report["summary"]["offline_cases"]["failed"] == 0
    assert report["summary"]["provider_fixtures"]["failed"] == 0
    assert report["summary"]["recorded_answers"]["failed"] == 0
    assert report["summary"]["live_retrieval"] == {"total": 0, "ok": 0, "failed": 0}
    assert report["summary"]["live_answers"] == {"total": 0, "ok": 0, "failed": 0}
    assert report["results"]["offline_cases"][0]["case_id"]
    assert "api_key" not in str(report).lower()


def test_recorded_answer_gate_rejects_missing_terms_and_citations():
    result = evaluate_recorded_answer(
        {
            "id": "bad-recorded",
            "expected_refs": [
                {"source": "a.pdf", "page": 1, "snippet": "alpha"},
                {"source": "b.pdf", "page": 2, "snippet": "beta"},
            ],
            "recorded_answer": "Only cites one source [1].",
            "required_answer_terms": ["observer", "switching"],
            "minimum_answer_term_coverage": 0.5,
        }
    )

    assert result is not None
    assert not result.ok
    assert result.citation_validation.missing_required_refs == [2]
    assert result.missing_terms == ["observer", "switching"]


def test_source_reference_gate_rejects_missing_pdf_snippet(tmp_path):
    import fitz

    source = tmp_path / "papers" / "library" / "paper.pdf"
    source.parent.mkdir(parents=True)
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "actual source text")
    document.save(source)

    result = evaluate_case(
        {
            "id": "bad-source",
            "expected_refs": [
                {
                    "source": "paper.pdf",
                    "source_path": "papers/library/paper.pdf",
                    "page": 1,
                    "snippet": "not present",
                },
            ],
            "fixture_answer": "Cites source [1].",
        },
        project_root=tmp_path,
    )

    assert not result.ok
    assert result.source_references[0].message == "snippet_not_found"


def test_answer_term_coverage_is_case_insensitive():
    coverage, matched, missing = answer_term_coverage(
        "The SOGIFO-X observer supports Sensorless control.",
        ["sogifo-x", "sensorless", "regression"],
    )

    assert coverage == 2 / 3
    assert matched == ["sogifo-x", "sensorless"]
    assert missing == ["regression"]


def test_live_query_payload_scores_context_citations_and_terms():
    case = {
        "id": "live-ok",
        "expected_refs": [
            {"source": "a.pdf", "source_path": "papers/library/a.pdf", "page": 1},
            {"source": "b.pdf", "source_path": "papers/library/b.pdf", "page": 2},
        ],
        "required_answer_terms": ["observer", "switching"],
        "live_required_answer_terms": ["observer", "switching"],
        "minimum_answer_term_coverage": 1.0,
        "minimum_live_answer_term_coverage": 1.0,
    }
    payload = {
        "request_id": "req-live",
        "result": {
            "answer": "The observer uses switching control [1].",
            "citation_validation": {"ok": True, "invalid_refs": [], "missing_source_page_refs": []},
            "context_refs": [
                {"ref": 1, "source_path": "papers/library/a.pdf", "page": 1},
            ],
        },
    }

    result = evaluate_live_query_payload(
        case,
        payload,
        minimum_expected_context_ref_coverage=0.5,
    )

    assert result.ok
    assert result.request_id == "req-live"
    assert result.expected_context_coverage == 0.5
    assert result.answer_term_coverage == 1.0


def test_live_query_payload_rejects_bad_generated_answer():
    case = {
        "id": "live-bad",
        "expected_refs": [
            {"source": "a.pdf", "source_path": "papers/library/a.pdf", "page": 1},
        ],
        "required_answer_terms": ["observer"],
        "live_required_answer_terms": ["observer"],
        "minimum_answer_term_coverage": 1.0,
        "minimum_live_answer_term_coverage": 1.0,
    }
    payload = {
        "request_id": "req-live",
        "result": {
            "answer": "Ungrounded answer [9].",
            "citation_validation": {"ok": False, "invalid_refs": [9], "missing_source_page_refs": []},
            "context_refs": [],
        },
    }

    result = evaluate_live_query_payload(case, payload)

    assert not result.ok
    assert result.missing_expected_source_paths == ["papers/library/a.pdf"]
    assert result.missing_terms == ["observer"]
    assert "invalid_refs=[9]" in result.message


def test_live_retrieval_payload_scores_context_sources_without_answer():
    case = {
        "id": "retrieval-ok",
        "expected_refs": [
            {"source": "a.pdf", "source_path": "papers/library/a.pdf", "page": 1},
            {"source": "b.pdf", "source_path": "papers/library/b.pdf", "page": 2},
        ],
    }
    payload = {
        "request_id": "req-retrieval",
        "retrieval": {
            "ok": True,
            "context_count": 2,
            "context_refs": [
                {"ref": 1, "source_path": "papers/library/a.pdf", "page": 1},
                {"ref": 2, "source_path": "papers/library/b.pdf", "page": 2},
            ],
            "missing_source_page_refs": [],
        },
    }

    result = evaluate_live_retrieval_payload(
        case,
        payload,
        minimum_expected_context_ref_coverage=1.0,
    )

    assert result.ok
    assert result.request_id == "req-retrieval"
    assert result.expected_context_coverage == 1.0
    assert result.context_count == 2


def test_live_retrieval_payload_rejects_missing_source_page_refs():
    case = {
        "id": "retrieval-bad",
        "expected_refs": [
            {"source": "a.pdf", "source_path": "papers/library/a.pdf", "page": 1},
        ],
    }
    payload = {
        "request_id": "req-retrieval",
        "retrieval": {
            "ok": False,
            "context_count": 1,
            "context_refs": [{"ref": 1, "source_path": "papers/library/a.pdf"}],
            "missing_source_page_refs": [1],
        },
    }

    result = evaluate_live_retrieval_payload(case, payload)

    assert not result.ok
    assert result.missing_source_page_refs == [1]
    assert "missing_source_page_refs=[1]" in result.message


def test_live_config_uses_injected_inspect_client():
    config = {
        "live_eval": {"minimum_expected_context_ref_coverage": 1.0, "timeout_s": 7},
        "cases": [
            {
                "id": "live-config",
                "question": "Explain SMC",
                "answer_mode": "explanation",
                "expected_refs": [
                    {"source": "a.pdf", "source_path": "papers/library/a.pdf", "page": 1},
                ],
                "required_answer_terms": ["observer"],
                "live_required_answer_terms": ["observer"],
                "minimum_answer_term_coverage": 1.0,
                "minimum_live_answer_term_coverage": 1.0,
            }
        ],
    }
    seen = {}

    def fake_client(base_url, api_token, case, *, timeout_s):
        seen["base_url"] = base_url
        seen["api_token"] = api_token
        seen["case_id"] = case["id"]
        seen["timeout_s"] = timeout_s
        return {
            "request_id": "req-live-config",
            "result": {
                "answer": "observer answer [1]",
                "citation_validation": {"ok": True, "invalid_refs": [], "missing_source_page_refs": []},
                "context_refs": [
                    {"ref": 1, "source_path": "papers/library/a.pdf", "page": 1},
                ],
            },
        }

    results = evaluate_live_config(
        config,
        base_url="http://api.test",
        api_token="secret",
        inspect_client=fake_client,
    )

    assert all(result.ok for result in results)
    assert seen == {
        "base_url": "http://api.test",
        "api_token": "secret",
        "case_id": "live-config",
        "timeout_s": 7.0,
    }


def test_live_retrieval_config_uses_injected_retrieve_client():
    config = {
        "live_eval": {"minimum_expected_context_ref_coverage": 1.0, "timeout_s": 9},
        "cases": [
            {
                "id": "retrieval-config",
                "question": "Explain SMC",
                "answer_mode": "explanation",
                "expected_refs": [
                    {"source": "a.pdf", "source_path": "papers/library/a.pdf", "page": 1},
                ],
            }
        ],
    }
    seen = {}

    def fake_client(base_url, api_token, case, *, timeout_s):
        seen["base_url"] = base_url
        seen["api_token"] = api_token
        seen["case_id"] = case["id"]
        seen["timeout_s"] = timeout_s
        return {
            "request_id": "req-retrieval-config",
            "retrieval": {
                "ok": True,
                "context_count": 1,
                "context_refs": [
                    {"ref": 1, "source_path": "papers/library/a.pdf", "page": 1},
                ],
                "missing_source_page_refs": [],
            },
        }

    results = evaluate_live_retrieval_config(
        config,
        base_url="http://api.test",
        api_token="secret",
        retrieve_client=fake_client,
    )

    assert all(result.ok for result in results)
    assert seen == {
        "base_url": "http://api.test",
        "api_token": "secret",
        "case_id": "retrieval-config",
        "timeout_s": 9.0,
    }
