"""Offline RAG evaluation helpers for citation and fixture checks."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from src.chain import CitationValidation, validate_numbered_citations
from src.config import PROJECT_ROOT
from src.runtime import normalize_exception


@dataclass(frozen=True)
class SourceReferenceResult:
    """Result for one expected source/page/snippet verification."""

    source: str
    page: int
    ok: bool
    message: str


@dataclass(frozen=True)
class EvaluationCaseResult:
    """Result for one offline evaluation case."""

    case_id: str
    ok: bool
    citation_validation: CitationValidation
    source_references: list[SourceReferenceResult]
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


@dataclass(frozen=True)
class LiveAnswerResult:
    """Result for one live /query/inspect regression case."""

    case_id: str
    ok: bool
    request_id: str | None
    citation_ok: bool
    expected_context_coverage: float
    answer_term_coverage: float
    matched_expected_source_paths: list[str]
    missing_expected_source_paths: list[str]
    matched_terms: list[str]
    missing_terms: list[str]
    message: str


@dataclass(frozen=True)
class LiveRetrievalResult:
    """Result for one live /query/retrieve no-LLM regression case."""

    case_id: str
    ok: bool
    request_id: str | None
    retrieval_ok: bool
    context_count: int
    expected_context_coverage: float
    matched_expected_source_paths: list[str]
    missing_expected_source_paths: list[str]
    missing_source_page_refs: list[int]
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


def normalize_page_text(value: str) -> str:
    """Normalize extracted PDF text and snippets for stable offline matching."""
    return " ".join(value.lower().split())


def source_path_for_ref(ref: dict[str, Any], *, project_root: Path = PROJECT_ROOT) -> Path:
    """Resolve an expected source reference inside the local project."""
    source_path = ref.get("source_path")
    if source_path:
        return (project_root / source_path).resolve()
    return (project_root / "papers" / "library" / ref["source"]).resolve()


def verify_source_reference(
    ref: dict[str, Any],
    *,
    project_root: Path = PROJECT_ROOT,
) -> SourceReferenceResult:
    """Verify that an expected source file/page contains its expected snippet."""
    path = source_path_for_ref(ref, project_root=project_root)
    try:
        path.relative_to(project_root.resolve())
    except ValueError:
        return SourceReferenceResult(
            source=ref["source"],
            page=int(ref["page"]),
            ok=False,
            message="source_path_escapes_project_root",
        )
    if not path.exists():
        return SourceReferenceResult(
            source=ref["source"],
            page=int(ref["page"]),
            ok=False,
            message=f"source_missing:{path}",
        )

    page = int(ref["page"])
    try:
        import fitz

        document = fitz.open(path)
        if page < 1 or page > document.page_count:
            return SourceReferenceResult(
                source=ref["source"],
                page=page,
                ok=False,
                message=f"page_out_of_range:{page}/{document.page_count}",
            )
        page_text = normalize_page_text(document[page - 1].get_text())
    except Exception as exc:
        return SourceReferenceResult(
            source=ref["source"],
            page=page,
            ok=False,
            message=f"source_parse_failed:{exc}",
        )

    snippet = normalize_page_text(ref.get("snippet", ""))
    ok = bool(snippet) and snippet in page_text
    return SourceReferenceResult(
        source=ref["source"],
        page=page,
        ok=ok,
        message="ok" if ok else "snippet_not_found",
    )


def evaluate_case(case: dict[str, Any], *, project_root: Path = PROJECT_ROOT) -> EvaluationCaseResult:
    """Evaluate one fixture answer against its expected retrieved refs."""
    docs = docs_from_expected_refs(case)
    required_refs = list(range(1, len(docs) + 1)) if case.get("require_all_refs", True) else []
    validation = validate_numbered_citations(
        case.get("fixture_answer", ""),
        docs,
        required_refs=required_refs,
    )
    source_references = [
        verify_source_reference(ref, project_root=project_root)
        for ref in case.get("expected_refs", [])
    ]
    source_ok = all(result.ok for result in source_references)
    if validation.ok and source_ok:
        message = "ok"
    else:
        source_failures = [
            f"{result.source}:p{result.page}:{result.message}"
            for result in source_references
            if not result.ok
        ]
        message = (
            f"invalid_refs={validation.invalid_refs} "
            f"missing_required_refs={validation.missing_required_refs} "
            f"source_failures={source_failures}"
        )
    return EvaluationCaseResult(
        case_id=case["id"],
        ok=validation.ok and source_ok,
        citation_validation=validation,
        source_references=source_references,
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


def required_terms_for_case(case: dict[str, Any], *, live: bool = False) -> list[str]:
    """Return deterministic answer terms for recorded or live scoring."""
    if live and case.get("live_required_answer_terms"):
        return case["live_required_answer_terms"]
    return case.get("required_answer_terms", [])


def minimum_term_coverage_for_case(case: dict[str, Any], *, live: bool = False) -> float:
    """Return deterministic answer-term coverage threshold."""
    if live and "minimum_live_answer_term_coverage" in case:
        return float(case["minimum_live_answer_term_coverage"])
    return float(case.get("minimum_answer_term_coverage", 1.0))


def expected_source_paths(case: dict[str, Any]) -> list[str]:
    """Return expected source paths from an eval case."""
    paths = []
    for ref in case.get("expected_refs", []):
        paths.append(ref.get("source_path", f"papers/library/{ref['source']}"))
    return paths


def live_eval_settings(config: dict[str, Any]) -> dict[str, Any]:
    """Return optional live-eval settings with conservative local defaults."""
    settings = config.get("live_eval", {})
    return {
        "minimum_expected_context_ref_coverage": float(
            settings.get("minimum_expected_context_ref_coverage", 0.5)
        ),
        "timeout_s": float(settings.get("timeout_s", 120.0)),
    }


def evaluate_live_query_payload(
    case: dict[str, Any],
    payload: dict[str, Any],
    *,
    minimum_expected_context_ref_coverage: float = 0.5,
) -> LiveAnswerResult:
    """Score one /query/inspect response against retrieval/citation/term gates."""
    result = payload.get("result", {})
    answer = result.get("answer", "")
    citation = result.get("citation_validation", {})
    citation_ok = bool(citation.get("ok"))
    context_paths = {
        ref.get("source_path")
        for ref in result.get("context_refs", [])
        if ref.get("source_path")
    }
    expected_paths = expected_source_paths(case)
    matched_paths = sorted(set(expected_paths) & context_paths)
    missing_paths = sorted(set(expected_paths) - context_paths)
    expected_coverage = len(matched_paths) / len(expected_paths) if expected_paths else 1.0

    required_terms = required_terms_for_case(case, live=True)
    minimum_terms = minimum_term_coverage_for_case(case, live=True)
    term_coverage, matched_terms, missing_terms = answer_term_coverage(answer, required_terms)

    ok = (
        citation_ok
        and expected_coverage >= minimum_expected_context_ref_coverage
        and term_coverage >= minimum_terms
    )
    if ok:
        message = (
            f"ok context_coverage={expected_coverage:.2f} "
            f"answer_coverage={term_coverage:.2f}"
        )
    else:
        message = (
            f"citation_ok={citation_ok} "
            f"context_coverage={expected_coverage:.2f} "
            f"minimum_context={minimum_expected_context_ref_coverage:.2f} "
            f"answer_coverage={term_coverage:.2f} "
            f"minimum_answer={minimum_terms:.2f} "
            f"invalid_refs={citation.get('invalid_refs', [])} "
            f"missing_source_page_refs={citation.get('missing_source_page_refs', [])} "
            f"missing_expected_sources={missing_paths} "
            f"missing_terms={missing_terms}"
        )
    return LiveAnswerResult(
        case_id=case["id"],
        ok=ok,
        request_id=payload.get("request_id"),
        citation_ok=citation_ok,
        expected_context_coverage=expected_coverage,
        answer_term_coverage=term_coverage,
        matched_expected_source_paths=matched_paths,
        missing_expected_source_paths=missing_paths,
        matched_terms=matched_terms,
        missing_terms=missing_terms,
        message=message,
    )


def evaluate_live_retrieval_payload(
    case: dict[str, Any],
    payload: dict[str, Any],
    *,
    minimum_expected_context_ref_coverage: float = 0.5,
) -> LiveRetrievalResult:
    """Score one /query/retrieve response against retrieval source/page gates."""
    retrieval = payload.get("retrieval", {})
    context_refs = retrieval.get("context_refs", [])
    context_paths = {
        ref.get("source_path")
        for ref in context_refs
        if ref.get("source_path")
    }
    expected_paths = expected_source_paths(case)
    matched_paths = sorted(set(expected_paths) & context_paths)
    missing_paths = sorted(set(expected_paths) - context_paths)
    expected_coverage = len(matched_paths) / len(expected_paths) if expected_paths else 1.0
    missing_source_page_refs = [
        int(ref)
        for ref in retrieval.get("missing_source_page_refs", [])
    ]
    retrieval_ok = bool(retrieval.get("ok"))
    context_count = int(retrieval.get("context_count") or len(context_refs))
    ok = (
        retrieval_ok
        and context_count > 0
        and not missing_source_page_refs
        and expected_coverage >= minimum_expected_context_ref_coverage
    )
    if ok:
        message = f"ok context_coverage={expected_coverage:.2f} context_count={context_count}"
    else:
        message = (
            f"retrieval_ok={retrieval_ok} "
            f"context_count={context_count} "
            f"context_coverage={expected_coverage:.2f} "
            f"minimum_context={minimum_expected_context_ref_coverage:.2f} "
            f"missing_source_page_refs={missing_source_page_refs} "
            f"missing_expected_sources={missing_paths}"
        )
    return LiveRetrievalResult(
        case_id=case["id"],
        ok=ok,
        request_id=payload.get("request_id"),
        retrieval_ok=retrieval_ok,
        context_count=context_count,
        expected_context_coverage=expected_coverage,
        matched_expected_source_paths=matched_paths,
        missing_expected_source_paths=missing_paths,
        missing_source_page_refs=missing_source_page_refs,
        message=message,
    )


def query_inspect_payload(
    base_url: str,
    api_token: str,
    case: dict[str, Any],
    *,
    timeout_s: float,
) -> dict[str, Any]:
    """Call a deployed /query/inspect endpoint for one eval case."""
    url = f"{base_url.rstrip('/')}/query/inspect"
    data = json.dumps(
        {
            "question": case["question"],
            "answer_mode": case.get("answer_mode", "explanation"),
        }
    ).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "X-Request-ID": f"live-eval-{case['id']}"[:64],
    }
    if api_token:
        headers["X-API-Key"] = api_token
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc


def query_retrieve_payload(
    base_url: str,
    api_token: str,
    case: dict[str, Any],
    *,
    timeout_s: float,
) -> dict[str, Any]:
    """Call a deployed /query/retrieve endpoint for one eval case."""
    url = f"{base_url.rstrip('/')}/query/retrieve"
    data = json.dumps(
        {
            "question": case["question"],
            "answer_mode": case.get("answer_mode", "explanation"),
        }
    ).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "X-Request-ID": f"retrieval-eval-{case['id']}"[:64],
    }
    if api_token:
        headers["X-API-Key"] = api_token
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc


def evaluate_live_config(
    config: dict[str, Any],
    *,
    base_url: str,
    api_token: str = "",
    timeout_s: float | None = None,
    inspect_client=query_inspect_payload,
) -> list[LiveAnswerResult]:
    """Evaluate configured cases against a live /query/inspect endpoint."""
    settings = live_eval_settings(config)
    minimum_context = settings["minimum_expected_context_ref_coverage"]
    timeout = timeout_s if timeout_s is not None else settings["timeout_s"]
    results: list[LiveAnswerResult] = []
    for case in config.get("cases", []):
        try:
            payload = inspect_client(base_url, api_token, case, timeout_s=timeout)
        except Exception as exc:
            error = normalize_exception(exc)
            results.append(
                LiveAnswerResult(
                    case_id=case["id"],
                    ok=False,
                    request_id=None,
                    citation_ok=False,
                    expected_context_coverage=0.0,
                    answer_term_coverage=0.0,
                    matched_expected_source_paths=[],
                    missing_expected_source_paths=expected_source_paths(case),
                    matched_terms=[],
                    missing_terms=required_terms_for_case(case, live=True),
                    message=f"request_failed code={error.code} message={error.message}",
                )
            )
            continue
        results.append(
            evaluate_live_query_payload(
                case,
                payload,
                minimum_expected_context_ref_coverage=minimum_context,
            )
        )
    return results


def evaluate_live_retrieval_config(
    config: dict[str, Any],
    *,
    base_url: str,
    api_token: str = "",
    timeout_s: float | None = None,
    retrieve_client=query_retrieve_payload,
) -> list[LiveRetrievalResult]:
    """Evaluate configured cases against a live /query/retrieve endpoint."""
    settings = live_eval_settings(config)
    minimum_context = settings["minimum_expected_context_ref_coverage"]
    timeout = timeout_s if timeout_s is not None else settings["timeout_s"]
    results: list[LiveRetrievalResult] = []
    for case in config.get("cases", []):
        try:
            payload = retrieve_client(base_url, api_token, case, timeout_s=timeout)
        except Exception as exc:
            error = normalize_exception(exc)
            results.append(
                LiveRetrievalResult(
                    case_id=case["id"],
                    ok=False,
                    request_id=None,
                    retrieval_ok=False,
                    context_count=0,
                    expected_context_coverage=0.0,
                    matched_expected_source_paths=[],
                    missing_expected_source_paths=expected_source_paths(case),
                    missing_source_page_refs=[],
                    message=f"request_failed code={error.code} message={error.message}",
                )
            )
            continue
        results.append(
            evaluate_live_retrieval_payload(
                case,
                payload,
                minimum_expected_context_ref_coverage=minimum_context,
            )
        )
    return results


def evaluate_recorded_answer(case: dict[str, Any]) -> RecordedAnswerResult | None:
    """Evaluate a recorded model answer without calling the provider."""
    recorded = case.get("recorded_answer")
    if not recorded:
        return None

    docs = docs_from_expected_refs(case)
    required_refs = list(range(1, len(docs) + 1)) if case.get("require_all_refs", True) else []
    validation = validate_numbered_citations(recorded, docs, required_refs=required_refs)
    required_terms = required_terms_for_case(case)
    minimum = minimum_term_coverage_for_case(case)
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
    *,
    project_root: Path = PROJECT_ROOT,
) -> tuple[list[EvaluationCaseResult], list[ProviderFixtureResult], list[RecordedAnswerResult]]:
    """Evaluate all offline cases and provider fixtures from a config."""
    cases = [evaluate_case(case, project_root=project_root) for case in config.get("cases", [])]
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


def _result_summary(results: list[Any]) -> dict[str, int]:
    ok = sum(1 for result in results if result.ok)
    return {
        "total": len(results),
        "ok": ok,
        "failed": len(results) - ok,
    }


def build_evaluation_report(
    config: dict[str, Any],
    *,
    case_results: list[EvaluationCaseResult],
    provider_results: list[ProviderFixtureResult],
    recorded_results: list[RecordedAnswerResult],
    live_results: list[LiveAnswerResult] | None = None,
    live_retrieval_results: list[LiveRetrievalResult] | None = None,
    eval_file: Path | None = None,
) -> dict[str, Any]:
    """Build a no-secret machine-readable evaluation report."""
    live_results = live_results or []
    live_retrieval_results = live_retrieval_results or []
    return {
        "schema_version": 1,
        "eval_file": str(eval_file) if eval_file else None,
        "case_count": len(config.get("cases", [])),
        "provider_fixture_count": len(config.get("provider_failure_fixtures", [])),
        "summary": {
            "offline_cases": _result_summary(case_results),
            "provider_fixtures": _result_summary(provider_results),
            "recorded_answers": _result_summary(recorded_results),
            "live_retrieval": _result_summary(live_retrieval_results),
            "live_answers": _result_summary(live_results),
        },
        "results": {
            "offline_cases": [asdict(result) for result in case_results],
            "provider_fixtures": [asdict(result) for result in provider_results],
            "recorded_answers": [asdict(result) for result in recorded_results],
            "live_retrieval": [asdict(result) for result in live_retrieval_results],
            "live_answers": [asdict(result) for result in live_results],
        },
    }
