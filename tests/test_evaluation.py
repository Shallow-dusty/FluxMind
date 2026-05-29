from pathlib import Path

from langchain_core.documents import Document

from src.chain import (
    ANSWER_MODE_INSTRUCTIONS,
    format_context,
    hybrid_retrieve,
    normalize_answer_mode,
    tokenize_query,
    validate_numbered_citations,
)
from src.evaluation import evaluate_config, load_eval_config


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


def test_hybrid_retrieve_adds_keyword_hits_after_vector_hits(monkeypatch):
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
            return [vector_doc][:k]

    monkeypatch.setattr("src.chain.get_vector_store", lambda: FakeStore())

    docs = hybrid_retrieve("flux observer", k=2)

    assert docs == [vector_doc, keyword_doc]


def test_tokenize_query_supports_cjk_terms():
    assert "磁链" in tokenize_query("磁链 observer")
    assert "observer" in tokenize_query("磁链 observer")


def test_offline_eval_config_passes():
    config = load_eval_config(Path("eval/rag_baseline.json"))

    case_results, provider_results = evaluate_config(config)

    assert case_results
    assert provider_results
    assert all(result.ok for result in case_results)
    assert all(result.ok for result in provider_results)
