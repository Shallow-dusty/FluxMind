"""Offline RAG evaluation helpers for citation and fixture checks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from src.chain import CitationValidation, validate_numbered_citations
from src.runtime import normalize_exception


@dataclass(frozen=True)
class EvaluationCaseResult:
    """Result for one offline evaluation case."""

    case_id: str
    ok: bool
    citation_validation: CitationValidation
    message: str


@dataclass(frozen=True)
class ProviderFixtureResult:
    """Result for one provider-error fixture."""

    fixture_id: str
    ok: bool
    expected_code: str
    actual_code: str


@dataclass(frozen=True)
class RecordedAnswerResult:
    """Result for one recorded answer quality/citation gate."""

    case_id: str
    ok: bool
    citation_validation: CitationValidation
    coverage: float
    matched_terms: list[str]
    missing_terms: list[str]
    message: str


def load_eval_config(path: Path) -> dict[str, Any]:
    """Load an offline evaluation JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def docs_from_expected_refs(case: dict[str, Any]) -> list[Document]:
    """Build minimal retrieved docs from expected references."""
    docs: list[Document] = []
    for ref in case.get("expected_refs", []):
        docs.append(
            Document(
                page_content=ref.get("snippet", ""),
                metadata={
                    "source": ref["source"],
                    "source_path": ref.get("source_path", ref["source"]),
                    "page": ref["page"],
                },
            )
        )
    return docs


def evaluate_case(case: dict[str, Any]) -> EvaluationCaseResult:
    """Evaluate one fixture answer against its expected retrieved refs."""
    docs = docs_from_expected_refs(case)
    required_refs = list(range(1, len(docs) + 1)) if case.get("require_all_refs", True) else []
    validation = validate_numbered_citations(
        case.get("fixture_answer", ""),
        docs,
        required_refs=required_refs,
    )
    message = "ok" if validation.ok else (
        f"invalid_refs={validation.invalid_refs} "
        f"missing_required_refs={validation.missing_required_refs}"
    )
    return EvaluationCaseResult(
        case_id=case["id"],
        ok=validation.ok,
        citation_validation=validation,
        message=message,
    )


def answer_term_coverage(answer: str, required_terms: list[str]) -> tuple[float, list[str], list[str]]:
    """Return simple deterministic coverage for recorded answer key terms."""
    if not required_terms:
        return 1.0, [], []
    normalized = answer.lower()
    matched = [term for term in required_terms if term.lower() in normalized]
    missing = [term for term in required_terms if term.lower() not in normalized]
    return len(matched) / len(required_terms), matched, missing


def evaluate_recorded_answer(case: dict[str, Any]) -> RecordedAnswerResult | None:
    """Evaluate a recorded model answer without calling the provider."""
    recorded = case.get("recorded_answer")
    if not recorded:
        return None

    docs = docs_from_expected_refs(case)
    required_refs = list(range(1, len(docs) + 1)) if case.get("require_all_refs", True) else []
    validation = validate_numbered_citations(recorded, docs, required_refs=required_refs)
    required_terms = case.get("required_answer_terms", [])
    minimum = float(case.get("minimum_answer_term_coverage", 1.0))
    coverage, matched_terms, missing_terms = answer_term_coverage(recorded, required_terms)
    ok = validation.ok and coverage >= minimum
    if ok:
        message = f"ok coverage={coverage:.2f}"
    else:
        message = (
            f"coverage={coverage:.2f} minimum={minimum:.2f} "
            f"invalid_refs={validation.invalid_refs} "
            f"missing_required_refs={validation.missing_required_refs} "
            f"missing_terms={missing_terms}"
        )
    return RecordedAnswerResult(
        case_id=case["id"],
        ok=ok,
        citation_validation=validation,
        coverage=coverage,
        matched_terms=matched_terms,
        missing_terms=missing_terms,
        message=message,
    )


def evaluate_provider_fixture(fixture: dict[str, Any]) -> ProviderFixtureResult:
    """Evaluate provider error normalization without calling an external provider."""
    error_type = fixture.get("type")
    message = fixture.get("message", "")
    if error_type == "timeout":
        exc: Exception = TimeoutError(message)
    else:
        exc = RuntimeError(message)
    normalized = normalize_exception(exc)
    expected_code = fixture["expected_code"]
    return ProviderFixtureResult(
        fixture_id=fixture["id"],
        ok=normalized.code == expected_code,
        expected_code=expected_code,
        actual_code=normalized.code,
    )


def evaluate_config(
    config: dict[str, Any],
) -> tuple[list[EvaluationCaseResult], list[ProviderFixtureResult], list[RecordedAnswerResult]]:
    """Evaluate all offline cases and provider fixtures from a config."""
    cases = [evaluate_case(case) for case in config.get("cases", [])]
    provider_fixtures = [
        evaluate_provider_fixture(fixture)
        for fixture in config.get("provider_failure_fixtures", [])
    ]
    recorded_answers = [
        result
        for case in config.get("cases", [])
        if (result := evaluate_recorded_answer(case)) is not None
    ]
    return cases, provider_fixtures, recorded_answers
