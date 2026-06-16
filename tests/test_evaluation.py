import sys
import types
from pathlib import Path

from langchain_core.documents import Document

from src.chain import (
    ANSWER_MODE_INSTRUCTIONS,
    bm25_relevance_scores,
    citation_instruction,
    clear_vector_store_cache,
    format_context,
    generated_artifact_context,
    get_vector_store,
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
    evaluate_code_output_case,
    evaluate_live_config,
    evaluate_live_query_payload,
    evaluate_live_retrieval_config,
    evaluate_live_retrieval_payload,
    evaluate_pdf_structure_case,
    evaluate_quality_maturity_targets,
    evaluate_recorded_answer,
    evaluate_retrieval_only_case,
    evaluate_regression_gates,
    load_eval_config,
    quality_metric_values,
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


def test_get_vector_store_reuses_cache_until_index_files_change(tmp_path, monkeypatch):
    from src import chain

    index_dir = tmp_path / "faiss_index"
    index_dir.mkdir()
    (index_dir / "index.faiss").write_text("index-1", encoding="utf-8")
    (index_dir / "index.pkl").write_text("pickle-1", encoding="utf-8")
    loads = []

    class FakeFAISS:
        @staticmethod
        def load_local(path, embeddings, allow_dangerous_deserialization):
            loads.append((path, embeddings, allow_dangerous_deserialization))
            return {"load": len(loads)}

    monkeypatch.setattr(chain, "FAISS_INDEX_DIR", index_dir)
    monkeypatch.setattr(chain, "get_embedding_model", lambda: "embedding")
    monkeypatch.setitem(
        sys.modules,
        "langchain_community.vectorstores",
        types.SimpleNamespace(FAISS=FakeFAISS),
    )
    clear_vector_store_cache()

    first = get_vector_store()
    second = get_vector_store()
    assert first is second
    assert len(loads) == 1

    (index_dir / "index.faiss").write_text("index-2-changed", encoding="utf-8")
    third = get_vector_store()
    assert third is not first
    assert len(loads) == 2


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

    (
        case_results,
        retrieval_only_results,
        code_output_results,
        pdf_structure_results,
        provider_results,
        recorded_results,
    ) = evaluate_config(config)
    gate_results = evaluate_regression_gates(
        config,
        case_results=case_results,
        retrieval_only_results=retrieval_only_results,
        code_output_results=code_output_results,
        pdf_structure_results=pdf_structure_results,
        provider_results=provider_results,
        recorded_results=recorded_results,
    )

    assert case_results
    assert provider_results
    assert recorded_results
    assert code_output_results
    assert pdf_structure_results
    assert gate_results
    assert all(result.ok for result in case_results)
    assert all(result.ok for result in provider_results)
    assert all(result.ok for result in recorded_results)
    assert all(result.ok for result in code_output_results)
    assert all(result.ok for result in pdf_structure_results)
    assert all(result.ok for result in gate_results)


def test_evaluation_report_summarizes_results_without_secrets():
    config = load_eval_config(Path("eval/rag_baseline.json"))
    (
        case_results,
        retrieval_only_results,
        code_output_results,
        pdf_structure_results,
        provider_results,
        recorded_results,
    ) = evaluate_config(config)
    gate_results = evaluate_regression_gates(
        config,
        case_results=case_results,
        retrieval_only_results=retrieval_only_results,
        code_output_results=code_output_results,
        pdf_structure_results=pdf_structure_results,
        provider_results=provider_results,
        recorded_results=recorded_results,
    )

    report = build_evaluation_report(
        config,
        case_results=case_results,
        retrieval_only_results=retrieval_only_results,
        code_output_results=code_output_results,
        pdf_structure_results=pdf_structure_results,
        provider_results=provider_results,
        recorded_results=recorded_results,
        regression_gate_results=gate_results,
        eval_file=Path("eval/rag_baseline.json"),
    )

    assert report["schema_version"] == 1
    assert report["eval_file"] == "eval/rag_baseline.json"
    assert report["summary"]["offline_cases"]["failed"] == 0
    assert report["summary"]["retrieval_only_cases"]["failed"] == 0
    assert report["summary"]["code_output_cases"]["failed"] == 0
    assert report["summary"]["pdf_structure_cases"]["failed"] == 0
    assert report["summary"]["provider_fixtures"]["failed"] == 0
    assert report["summary"]["recorded_answers"]["failed"] == 0
    assert report["summary"]["live_retrieval"] == {"total": 0, "ok": 0, "failed": 0}
    assert report["summary"]["live_answers"] == {"total": 0, "ok": 0, "failed": 0}
    assert report["summary"]["regression_gates"]["failed"] == 0
    assert "PMSM" in report["coverage"]["topic_tags"]
    assert "answer_quality" in report["coverage"]["eval_lanes"]
    assert "pmsm_motor_control" in report["coverage"]["topic_groups"]
    assert report["quality_maturity"]["metrics"]["seed_paper_count"] >= 11
    assert {target["id"] for target in report["quality_maturity"]["targets"]} >= {
        "self_use",
        "small_group",
        "community",
    }
    assert report["results"]["offline_cases"][0]["case_id"]
    assert report["results"]["pdf_structure_cases"][0]["case_id"]
    assert report["results"]["regression_gates"][0]["gate_id"]
    assert "api_key" not in str(report).lower()


def test_default_rag_baseline_reaches_twenty_case_domain_gate():
    config = load_eval_config(Path(__file__).resolve().parents[1] / "eval/rag_baseline.json")
    (
        case_results,
        retrieval_only_results,
        code_output_results,
        pdf_structure_results,
        provider_results,
        recorded_results,
    ) = evaluate_config(config)
    gate_results = evaluate_regression_gates(
        config,
        case_results=case_results,
        retrieval_only_results=retrieval_only_results,
        code_output_results=code_output_results,
        pdf_structure_results=pdf_structure_results,
        provider_results=provider_results,
        recorded_results=recorded_results,
    )
    report = build_evaluation_report(
        config,
        case_results=case_results,
        retrieval_only_results=retrieval_only_results,
        code_output_results=code_output_results,
        pdf_structure_results=pdf_structure_results,
        provider_results=provider_results,
        recorded_results=recorded_results,
        regression_gate_results=gate_results,
    )

    assert report["case_count"] >= 20
    assert report["retrieval_only_case_count"] >= 30
    assert report["retrieval_eval_question_count"] >= 50
    assert report["code_output_case_count"] >= 3
    assert report["pdf_structure_case_count"] >= 6
    assert len(report["coverage"]["topic_tags"]) >= 50
    assert "zero speed" in report["coverage"]["topic_tags"]
    assert "MATLAB export" in report["coverage"]["topic_tags"]
    assert report["summary"]["offline_cases"]["failed"] == 0
    assert report["summary"]["retrieval_only_cases"]["ok"] >= 30
    assert report["summary"]["code_output_cases"]["ok"] >= 3
    assert report["summary"]["pdf_structure_cases"]["ok"] >= 6
    assert report["summary"]["recorded_answers"]["ok"] >= 20
    assert report["summary"]["regression_gates"]["failed"] == 0
    assert all(
        case.get("topic_tags") and case.get("eval_lanes") and case.get("expected_refs")
        for case in config["cases"] + config["retrieval_only_cases"]
    )


def test_default_quality_maturity_targets_mark_self_use_met_and_future_gaps():
    config = load_eval_config(Path(__file__).resolve().parents[1] / "eval/rag_baseline.json")
    metrics = quality_metric_values(config)
    targets = evaluate_quality_maturity_targets(config, metrics)

    by_id = {target["id"]: target for target in targets}
    assert by_id["self_use"]["ok"]
    assert by_id["small_group"]["status"] == "gap"
    assert "live_retrieval_result_count" in by_id["small_group"]["missing_metrics"]
    assert "seed_paper_count" not in by_id["small_group"]["missing_metrics"]
    assert by_id["community"]["status"] == "gap"
    assert "live_answer_result_count" in by_id["community"]["missing_metrics"]


def test_quality_metric_values_handles_manifest_edges(tmp_path):
    config = {
        "domain_ontology": {
            "topic_groups": {
                "motor_control": ["PMSM"],
            },
        },
        "cases": [
            {
                "id": "answer",
                "topic_tags": ["PMSM"],
                "eval_lanes": ["answer_quality"],
                "recorded_answer": "PMSM answer [1].",
            }
        ],
        "retrieval_only_cases": [
            {
                "id": "retrieval",
                "topic_tags": ["FOC"],
                "eval_lanes": ["retrieval"],
            }
        ],
    }

    assert quality_metric_values(config, project_root=tmp_path)["seed_paper_count"] == 0

    manifest_path = tmp_path / "papers" / "library" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text("[", encoding="utf-8")
    assert quality_metric_values(config, project_root=tmp_path)["seed_paper_count"] == 0

    manifest_path.write_text("[]", encoding="utf-8")
    assert quality_metric_values(config, project_root=tmp_path)["seed_paper_count"] == 0

    manifest_path.write_text('{"paper.pdf": {}}', encoding="utf-8")
    metrics = quality_metric_values(
        config,
        live_results=[object()],
        live_retrieval_results=[object(), object()],
        project_root=tmp_path,
    )
    assert metrics["seed_paper_count"] == 1
    assert metrics["retrieval_eval_question_count"] == 2
    assert metrics["recorded_answer_count"] == 1
    assert metrics["topic_group_count"] == 1
    assert metrics["live_answer_result_count"] == 1
    assert metrics["live_retrieval_result_count"] == 2


def test_quality_maturity_targets_ignore_empty_ids_and_bad_required_metrics():
    targets = evaluate_quality_maturity_targets(
        {
            "quality_maturity_targets": [
                {"label": "missing id"},
                {"id": "bad-required", "required_metrics": []},
                {"id": "gap", "required_metrics": {"paper_count": 3, "case_count": 1}},
                {"id": "met", "required_metrics": {"case_count": 1}},
            ]
        },
        {"case_count": 1, "paper_count": 2},
    )

    by_id = {target["id"]: target for target in targets}
    assert set(by_id) == {"bad-required", "gap", "met"}
    assert by_id["bad-required"]["ok"]
    assert by_id["gap"]["status"] == "gap"
    assert by_id["gap"]["missing_metrics"] == ["paper_count"]
    assert by_id["gap"]["checks"][0]["gap"] == 0
    assert by_id["gap"]["checks"][1]["gap"] == 1
    assert by_id["met"]["status"] == "met"


def test_retrieval_only_cases_are_reported_and_gated(tmp_path):
    paper = tmp_path / "paper.pdf"
    from fitz import open as open_pdf

    doc = open_pdf()
    page = doc.new_page()
    page.insert_text((72, 72), "PMSM retrieval-only source text")
    doc.save(paper)
    doc.close()

    config = {
        "quality_gates": {
            "minimum_case_count": 1,
            "minimum_retrieval_only_case_count": 1,
            "minimum_retrieval_eval_question_count": 2,
            "minimum_retrieval_only_pass_rate": 1.0,
            "minimum_expected_source_ref_count": 2,
        },
        "cases": [
            {
                "id": "answer-case",
                "question": "What is the answer case?",
                "fixture_answer": "Use source [1].",
                "expected_refs": [
                    {
                        "source": "paper.pdf",
                        "source_path": "paper.pdf",
                        "page": 1,
                        "snippet": "PMSM retrieval-only source text",
                    }
                ],
            }
        ],
        "retrieval_only_cases": [
            {
                "id": "retrieval-case",
                "question": "Find the retrieval-only source.",
                "expected_refs": [
                    {
                        "source": "paper.pdf",
                        "source_path": "paper.pdf",
                        "page": 1,
                        "snippet": "PMSM retrieval-only source text",
                    }
                ],
            }
        ],
    }

    retrieval_result = evaluate_retrieval_only_case(
        config["retrieval_only_cases"][0],
        project_root=tmp_path,
    )
    (
        case_results,
        retrieval_only_results,
        code_output_results,
        pdf_structure_results,
        provider_results,
        recorded_results,
    ) = evaluate_config(config, project_root=tmp_path)
    gate_results = evaluate_regression_gates(
        config,
        case_results=case_results,
        retrieval_only_results=retrieval_only_results,
        code_output_results=code_output_results,
        pdf_structure_results=pdf_structure_results,
        provider_results=provider_results,
        recorded_results=recorded_results,
    )
    report = build_evaluation_report(
        config,
        case_results=case_results,
        retrieval_only_results=retrieval_only_results,
        code_output_results=code_output_results,
        pdf_structure_results=pdf_structure_results,
        provider_results=provider_results,
        recorded_results=recorded_results,
        regression_gate_results=gate_results,
    )

    assert retrieval_result.ok
    assert report["retrieval_only_case_count"] == 1
    assert report["retrieval_eval_question_count"] == 2
    assert report["summary"]["retrieval_only_cases"] == {"total": 1, "ok": 1, "failed": 0}
    assert {result.gate_id for result in gate_results} >= {
        "minimum_retrieval_only_case_count",
        "minimum_retrieval_eval_question_count",
        "minimum_retrieval_only_pass_rate",
    }
    assert all(result.ok for result in gate_results)


def test_retrieval_only_case_rejects_missing_expected_refs():
    result = evaluate_retrieval_only_case(
        {
            "id": "missing-refs",
            "question": "Which source supports this?",
            "expected_refs": [],
        }
    )

    assert not result.ok
    assert result.message == "expected_refs_missing"


def test_pdf_structure_case_verifies_equation_marker(tmp_path):
    import fitz

    paper = tmp_path / "papers" / "library" / "structure.pdf"
    paper.parent.mkdir(parents=True)
    document = fitz.open()
    page = document.new_page()
    page.insert_text(
        (72, 72),
        "Voltage equation\nud = Rsid + Ld did/dt\nTable 1. Parameter summary",
    )
    document.save(paper)
    document.close()

    result = evaluate_pdf_structure_case(
        {
            "id": "structure-equation",
            "source_path": "papers/library/structure.pdf",
            "page": 1,
            "kind": "equation",
            "contains": "ud = Rsid",
        },
        project_root=tmp_path,
    )

    assert result.ok
    assert result.kind == "equation"
    assert result.marker_count >= 1
    assert "ud = Rsid" in result.matched_text


def test_code_output_case_runs_python_and_verifies_artifacts():
    result = evaluate_code_output_case(
        {
            "id": "code-output-ok",
            "language": "python",
            "entrypoint": "main.py",
            "files": {
                "main.py": (
                    "from pathlib import Path\n"
                    "Path('plot.png').write_bytes(b'\\x89PNG\\r\\n\\x1a\\n')\n"
                    "Path('summary.txt').write_text('sliding mode code-output eval', encoding='utf-8')\n"
                    "print('fluxmind-code-output-ok plot.png summary.txt')\n"
                )
            },
            "required_stdout_terms": ["fluxmind-code-output-ok", "plot.png"],
            "expected_artifacts": [
                {
                    "title": "plot.png",
                    "kind": "plot",
                    "mime_type": "image/png",
                    "minimum_byte_count": 8,
                },
                {
                    "title": "summary.txt",
                    "kind": "text",
                    "mime_type": "text/plain",
                    "contains": "sliding mode code-output eval",
                },
            ],
            "expected_runtime_metadata": {
                "provider_runtime": "python-local",
                "filesystem_isolation": "temporary_workdir",
            },
        }
    )

    assert result.ok
    assert result.language == "python"
    assert result.exit_code == 0
    assert [artifact.title for artifact in result.artifact_results] == ["plot.png", "summary.txt"]
    assert all(artifact.ok for artifact in result.artifact_results)


def test_code_output_case_runs_python_template_and_verifies_plot_artifact():
    result = evaluate_code_output_case(
        {
            "id": "code-output-template",
            "language": "python",
            "template_id": "smc_reaching_law",
            "entrypoint": "main.py",
            "required_stdout_terms": ["wrote smc_reaching_law.csv and smc_reaching_law.svg"],
            "expected_artifacts": [
                {
                    "title": "smc_reaching_law.csv",
                    "kind": "text",
                    "mime_type": "text/csv",
                    "contains": "time_s,sliding_surface",
                },
                {
                    "title": "smc_reaching_law.svg",
                    "kind": "plot",
                    "mime_type": "image/svg+xml",
                    "contains": "SMC reaching-law response",
                },
            ],
            "expected_runtime_metadata": {
                "provider_runtime": "python-local",
                "filesystem_isolation": "temporary_workdir",
            },
        }
    )

    assert result.ok
    assert [artifact.title for artifact in result.artifact_results] == [
        "smc_reaching_law.csv",
        "smc_reaching_law.svg",
    ]


def test_code_output_case_accepts_expected_octave_runtime_unavailable(monkeypatch):
    monkeypatch.setattr("src.providers.shutil.which", lambda _name: None)

    result = evaluate_code_output_case(
        {
            "id": "octave-runtime-unavailable",
            "language": "octave",
            "template_id": "pmsm_current_decay",
            "entrypoint": "main.m",
            "expected_artifacts": [
                {
                    "title": "pmsm_current_decay.csv",
                    "kind": "text",
                    "mime_type": "text/csv",
                },
            ],
            "expected_runtime_unavailable": {
                "exit_code": 127,
                "stderr_contains": ["GNU Octave executable not found"],
                "runtime_metadata": {
                    "provider_runtime": "gnu-octave-local",
                    "runtime_available": "false",
                    "octave_available": "false",
                },
            },
        }
    )

    assert result.ok
    assert result.exit_code == 127
    assert result.artifact_results == []
    assert result.message == "ok runtime_unavailable mode=provider"


def test_code_output_case_reports_octave_runtime_unavailable_mismatch(monkeypatch):
    monkeypatch.setattr("src.providers.shutil.which", lambda _name: None)

    result = evaluate_code_output_case(
        {
            "id": "octave-runtime-unavailable-mismatch",
            "language": "octave",
            "template_id": "pmsm_current_decay",
            "entrypoint": "main.m",
            "expected_artifacts": [
                {
                    "title": "pmsm_current_decay.csv",
                    "kind": "text",
                    "mime_type": "text/csv",
                },
            ],
            "expected_runtime_unavailable": {
                "exit_code": 127,
                "runtime_metadata": {"runtime_available": "true"},
            },
        }
    )

    assert not result.ok
    assert result.exit_code == 127
    assert result.artifact_results == []
    assert result.missing_runtime_metadata == ["runtime_available=true"]
    assert result.message == "runtime_unavailable_mismatch=['runtime_available=true']"


def test_code_output_case_runs_octave_template_when_runtime_exists(tmp_path, monkeypatch):
    fake_octave = tmp_path / "octave"
    fake_octave.write_text(
        "#!/bin/sh\n"
        "echo wrote pmsm_current_decay.csv\n"
        "printf '0,1,0,1\\n0.1,1,0.2,0.8\\n' > pmsm_current_decay.csv\n",
        encoding="utf-8",
    )
    fake_octave.chmod(0o755)
    monkeypatch.setattr("src.providers.shutil.which", lambda _name: str(fake_octave))

    result = evaluate_code_output_case(
        {
            "id": "octave-template",
            "language": "octave",
            "template_id": "pmsm_current_decay",
            "entrypoint": "main.m",
            "required_stdout_terms": ["wrote pmsm_current_decay.csv"],
            "expected_artifacts": [
                {
                    "title": "pmsm_current_decay.csv",
                    "kind": "text",
                    "mime_type": "text/csv",
                    "minimum_byte_count": 16,
                    "contains": "0.1,1,0.2,0.8",
                },
            ],
            "expected_runtime_metadata": {
                "provider_runtime": "gnu-octave-local",
                "runtime_available": "true",
                "octave_available": "true",
                "filesystem_isolation": "temporary_workdir",
            },
            "expected_runtime_unavailable": {
                "exit_code": 127,
                "runtime_metadata": {"runtime_available": "false"},
            },
        }
    )

    assert result.ok
    assert result.exit_code == 0
    assert result.artifact_results[0].ok
    assert result.message == "ok mode=provider artifacts=1"


def test_code_output_case_runs_python_through_local_job_runner():
    result = evaluate_code_output_case(
        {
            "id": "code-output-local-job",
            "language": "python",
            "execution_mode": "local_job",
            "entrypoint": "main.py",
            "files": {
                "main.py": (
                    "from pathlib import Path\n"
                    "Path('job-summary.txt').write_text('job-backed code-output eval', encoding='utf-8')\n"
                    "Path('job-plot.svg').write_text('<svg>job-backed plot</svg>', encoding='utf-8')\n"
                    "print('fluxmind-job-code-output job-summary.txt job-plot.svg')\n"
                )
            },
            "required_stdout_terms": ["fluxmind-job-code-output", "job-summary.txt"],
            "expected_artifacts": [
                {
                    "title": "job-summary.txt",
                    "kind": "text",
                    "mime_type": "text/plain",
                    "contains": "job-backed code-output eval",
                },
                {
                    "title": "job-plot.svg",
                    "kind": "plot",
                    "mime_type": "image/svg+xml",
                    "contains": "job-backed plot",
                },
            ],
            "expected_runtime_metadata": {
                "provider_runtime": "python-local",
                "filesystem_isolation": "temporary_workdir",
            },
            "expected_job_status": "succeeded",
            "expected_job_metadata": {
                "kind": "code_execution",
                "attempts": "1",
            },
            "expected_job_log_statuses": ["running", "succeeded"],
        }
    )

    assert result.ok
    assert result.execution_mode == "local_job"
    assert result.job_status == "succeeded"
    assert result.job_metadata_ok
    assert [artifact.title for artifact in result.artifact_results] == [
        "job-summary.txt",
        "job-plot.svg",
    ]


def test_code_output_case_rejects_unknown_template():
    result = evaluate_code_output_case(
        {
            "id": "code-output-unknown-template",
            "language": "python",
            "template_id": "missing-template",
            "expected_artifacts": [
                {
                    "title": "plot.png",
                    "kind": "plot",
                    "mime_type": "image/png",
                },
            ],
        }
    )

    assert not result.ok
    assert result.exit_code == 2
    assert result.message == "unknown_code_output_template:python:missing-template"


def test_code_output_case_rejects_missing_expected_artifacts():
    result = evaluate_code_output_case(
        {
            "id": "code-output-missing-artifact",
            "language": "python",
            "entrypoint": "main.py",
            "files": {"main.py": "print('no artifact')"},
            "required_stdout_terms": ["no artifact"],
            "expected_artifacts": [
                {
                    "title": "plot.png",
                    "kind": "plot",
                    "mime_type": "image/png",
                },
            ],
        }
    )

    assert not result.ok
    assert result.stdout_ok
    assert result.artifact_results[0].message == "artifact_missing"


def test_regression_gates_reject_narrow_eval_set():
    config = {
        "quality_gates": {
            "minimum_case_count": 2,
            "required_answer_modes": ["explanation", "derivation"],
            "minimum_provider_fixture_count": 1,
            "minimum_recorded_answer_count": 1,
            "minimum_recorded_answer_pass_rate": 1.0,
            "minimum_average_recorded_answer_term_coverage": 0.8,
            "minimum_topic_tag_count": 2,
            "required_topic_tags": ["SMC", "PMSM"],
            "minimum_eval_lane_count": 2,
            "required_eval_lanes": ["retrieval", "answer_quality"],
            "required_topic_groups": ["motor_control"],
            "minimum_code_output_case_count": 1,
            "required_code_output_languages": ["python"],
            "required_code_output_template_ids": ["smc_reaching_law"],
            "required_code_output_execution_modes": ["provider", "local_job"],
            "minimum_code_output_pass_rate": 1.0,
            "minimum_pdf_structure_case_count": 1,
            "required_pdf_structure_kinds": ["equation", "table", "figure"],
            "minimum_pdf_structure_pass_rate": 1.0,
        },
        "domain_ontology": {
            "topic_groups": {
                "motor_control": ["PMSM"],
            },
        },
        "cases": [
            {
                "id": "only-explanation",
                "answer_mode": "explanation",
                "topic_tags": ["SMC"],
                "eval_lanes": ["retrieval"],
                "expected_refs": [],
            }
        ],
    }

    results = evaluate_regression_gates(
        config,
        case_results=[],
        provider_results=[],
        recorded_results=[],
    )

    failed_gate_ids = {result.gate_id for result in results if not result.ok}
    assert "minimum_case_count" in failed_gate_ids
    assert "required_answer_modes" in failed_gate_ids
    assert "minimum_recorded_answer_count" in failed_gate_ids
    assert "minimum_average_recorded_answer_term_coverage" in failed_gate_ids
    assert "minimum_topic_tag_count" in failed_gate_ids
    assert "required_topic_tags" in failed_gate_ids
    assert "minimum_eval_lane_count" in failed_gate_ids
    assert "required_eval_lanes" in failed_gate_ids
    assert "required_topic_groups" in failed_gate_ids
    assert "minimum_code_output_case_count" in failed_gate_ids
    assert "required_code_output_languages" in failed_gate_ids
    assert "required_code_output_template_ids" in failed_gate_ids
    assert "required_code_output_execution_modes" in failed_gate_ids
    assert "minimum_code_output_pass_rate" in failed_gate_ids
    assert "minimum_pdf_structure_case_count" in failed_gate_ids
    assert "required_pdf_structure_kinds" in failed_gate_ids
    assert "minimum_pdf_structure_pass_rate" in failed_gate_ids


def test_regression_gates_accept_domain_coverage_metadata():
    config = {
        "domain_ontology": {
            "topic_groups": {
                "motor_control": ["PMSM", "FOC"],
                "robust_control": ["SMC"],
            },
        },
        "quality_gates": {
            "minimum_topic_tag_count": 3,
            "required_topic_tags": ["SMC", "PMSM", "FOC"],
            "minimum_eval_lane_count": 3,
            "required_eval_lanes": ["retrieval", "answer_quality", "forum_style"],
            "required_topic_groups": ["motor_control", "robust_control"],
        },
        "cases": [
            {
                "id": "smc",
                "topic_tags": ["SMC", "PMSM"],
                "eval_lanes": ["retrieval", "answer_quality"],
            },
            {
                "id": "foc",
                "topic_tags": ["FOC"],
                "eval_lanes": ["forum_style"],
            },
        ],
    }

    results = evaluate_regression_gates(
        config,
        case_results=[],
        provider_results=[],
        recorded_results=[],
    )

    assert results
    assert all(result.ok for result in results)


def test_regression_gates_score_live_results_when_supplied():
    config = {
        "quality_gates": {
            "minimum_live_answer_pass_rate": 1.0,
            "minimum_average_live_answer_term_coverage": 0.9,
        },
        "cases": [],
    }
    case = {
        "id": "live-bad",
        "expected_refs": [
            {"source": "a.pdf", "source_path": "papers/library/a.pdf", "page": 1},
        ],
        "live_required_answer_terms": ["observer"],
        "minimum_live_answer_term_coverage": 1.0,
    }
    live_result = evaluate_live_query_payload(
        case,
        {
            "request_id": "req-live",
            "result": {
                "answer": "Ungrounded answer.",
                "citation_validation": {"ok": False},
                "context_refs": [],
            },
        },
    )

    results = evaluate_regression_gates(
        config,
        case_results=[],
        provider_results=[],
        recorded_results=[],
        live_results=[live_result],
    )

    assert {result.gate_id for result in results if not result.ok} == {
        "minimum_live_answer_pass_rate",
        "minimum_average_live_answer_term_coverage",
    }


def test_live_scoring_can_use_explicit_live_expected_refs():
    case = {
        "id": "live-alternative-source",
        "expected_refs": [
            {"source": "offline.pdf", "source_path": "papers/library/offline.pdf", "page": 1},
        ],
        "live_expected_refs": [
            {"source": "offline.pdf", "source_path": "papers/library/offline.pdf", "page": 1},
            {"source": "live.pdf", "source_path": "papers/library/live.pdf", "page": 2},
        ],
        "live_required_answer_terms": ["observer"],
        "minimum_live_answer_term_coverage": 1.0,
    }

    answer_result = evaluate_live_query_payload(
        case,
        {
            "request_id": "req-live-alternative",
            "result": {
                "answer": "Observer answer [1].",
                "citation_validation": {"ok": True},
                "context_refs": [
                    {"source_path": "papers/library/live.pdf", "page": 2},
                ],
            },
        },
    )
    retrieval_result = evaluate_live_retrieval_payload(
        case,
        {
            "request_id": "req-live-retrieval-alternative",
            "retrieval": {
                "ok": True,
                "context_count": 1,
                "context_refs": [
                    {"source_path": "papers/library/live.pdf", "page": 2},
                ],
                "missing_source_page_refs": [],
            },
        },
    )

    assert answer_result.ok
    assert answer_result.expected_context_coverage == 0.5
    assert answer_result.matched_expected_source_paths == ["papers/library/live.pdf"]
    assert retrieval_result.ok
    assert retrieval_result.expected_context_coverage == 0.5
    assert retrieval_result.missing_expected_source_paths == ["papers/library/offline.pdf"]


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
