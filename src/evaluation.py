"""Offline RAG evaluation helpers for citation and fixture checks."""

from __future__ import annotations

import json
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from src.capabilities import CodeExecutionRequest, CodeExecutionResult, GeneratedArtifact
from src.chain import CitationValidation, validate_numbered_citations
from src.config import PROJECT_ROOT
from src.execution_templates import OCTAVE_EXECUTION_TEMPLATES, PYTHON_EXECUTION_TEMPLATES
from src.ingestion import extract_pdf_structure_markers
from src.jobs import JobRecord, LocalJobRunner, LocalJobStore
from src.providers import LocalArtifactStore, LocalOctaveExecutionProvider, LocalPythonExecutionProvider
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
class RetrievalOnlyCaseResult:
    """Result for one no-LLM retrieval-only source/page case."""

    case_id: str
    ok: bool
    source_references: list[SourceReferenceResult]
    message: str


@dataclass(frozen=True)
class PdfStructureCaseResult:
    """Result for one no-key PDF layout extraction acceptance case."""

    case_id: str
    ok: bool
    kind: str
    source_path: str
    page: int | None
    matched_text: str | None
    marker_count: int
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


@dataclass(frozen=True)
class CodeOutputArtifactResult:
    """Result for one expected artifact produced by a code-output eval."""

    title: str
    kind: str
    ok: bool
    message: str


@dataclass(frozen=True)
class CodeOutputCaseResult:
    """Result for one local no-key code-output execution eval case."""

    case_id: str
    ok: bool
    language: str
    execution_mode: str
    exit_code: int
    stdout_ok: bool
    missing_stdout_terms: list[str]
    runtime_metadata_ok: bool
    missing_runtime_metadata: list[str]
    job_status: str | None
    job_metadata_ok: bool
    missing_job_metadata: list[str]
    artifact_results: list[CodeOutputArtifactResult]
    message: str


@dataclass(frozen=True)
class RegressionGateResult:
    """Result for one config-level RAG regression gate."""

    gate_id: str
    ok: bool
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


def evaluate_retrieval_only_case(
    case: dict[str, Any],
    *,
    project_root: Path = PROJECT_ROOT,
) -> RetrievalOnlyCaseResult:
    """Evaluate one retrieval-only case by verifying expected source/page refs."""
    expected_refs = case.get("expected_refs", [])
    if not expected_refs:
        return RetrievalOnlyCaseResult(
            case_id=case["id"],
            ok=False,
            source_references=[],
            message="expected_refs_missing",
        )
    source_references = [
        verify_source_reference(ref, project_root=project_root)
        for ref in expected_refs
    ]
    source_ok = all(result.ok for result in source_references)
    if source_ok:
        message = "ok"
    else:
        source_failures = [
            f"{result.source}:p{result.page}:{result.message}"
            for result in source_references
            if not result.ok
        ]
        message = f"source_failures={source_failures}"
    return RetrievalOnlyCaseResult(
        case_id=case["id"],
        ok=source_ok,
        source_references=source_references,
        message=message,
    )


def evaluate_pdf_structure_case(
    case: dict[str, Any],
    *,
    project_root: Path = PROJECT_ROOT,
) -> PdfStructureCaseResult:
    """Evaluate one no-key PDF layout extraction acceptance case."""
    source_path = str(case["source_path"])
    path = (project_root / source_path).resolve()
    kind = str(case.get("kind", "")).casefold()
    page = int(case["page"]) if case.get("page") is not None else None
    contains = normalize_page_text(str(case.get("contains", "")))
    try:
        path.relative_to(project_root.resolve())
    except ValueError:
        return PdfStructureCaseResult(
            case_id=case["id"],
            ok=False,
            kind=kind,
            source_path=source_path,
            page=page,
            matched_text=None,
            marker_count=0,
            message="source_path_escapes_project_root",
        )
    if not path.exists():
        return PdfStructureCaseResult(
            case_id=case["id"],
            ok=False,
            kind=kind,
            source_path=source_path,
            page=page,
            matched_text=None,
            marker_count=0,
            message=f"source_missing:{path}",
        )
    markers = extract_pdf_structure_markers(
        path,
        kinds={kind} if kind else None,
        page=page,
        max_markers=int(case.get("max_markers", 100)),
    )
    matched_text = None
    if contains:
        for marker in markers:
            if contains in normalize_page_text(marker.text):
                matched_text = marker.text
                break
    elif markers:
        matched_text = markers[0].text
    ok = matched_text is not None
    if ok:
        message = f"ok markers={len(markers)}"
    else:
        message = f"markers={len(markers)} contains_missing={case.get('contains', '')}"
    return PdfStructureCaseResult(
        case_id=case["id"],
        ok=ok,
        kind=kind,
        source_path=source_path,
        page=page,
        matched_text=matched_text,
        marker_count=len(markers),
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


def retrieval_eval_cases(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return all cases that should participate in no-LLM retrieval checks."""
    return list(config.get("cases", [])) + list(config.get("retrieval_only_cases", []))


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
    for case in retrieval_eval_cases(config):
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


def _execution_provider(language: str, artifact_root: Path):
    store = LocalArtifactStore(artifact_root)
    if language == "python":
        return LocalPythonExecutionProvider(store)
    if language == "octave":
        return LocalOctaveExecutionProvider(store)
    raise ValueError(f"unsupported_code_output_language:{language}")


def _code_output_result_from_job(job: JobRecord) -> CodeExecutionResult:
    payload = job.result or {}
    artifacts = [
        GeneratedArtifact(**artifact)
        for artifact in job.artifacts
    ]
    return CodeExecutionResult(
        exit_code=int(payload.get("exit_code", 1)),
        stdout=str(payload.get("stdout", "")),
        stderr=str(payload.get("stderr", "")),
        artifacts=artifacts,
        runtime_metadata={
            str(key): str(value)
            for key, value in payload.get("runtime_metadata", {}).items()
        },
    )


def _missing_code_output_job_metadata(case: dict[str, Any], job: JobRecord) -> list[str]:
    missing: list[str] = []
    expected_status = case.get("expected_job_status")
    if expected_status and job.status != str(expected_status):
        missing.append(f"status={expected_status}")

    expected_metadata = {
        str(key): str(value)
        for key, value in case.get("expected_job_metadata", {}).items()
    }
    for key, value in expected_metadata.items():
        if str(getattr(job, key, "")) != value:
            missing.append(f"{key}={value}")

    expected_log_statuses = [
        str(status)
        for status in case.get("expected_job_log_statuses", [])
    ]
    if expected_log_statuses:
        actual_log_statuses = [
            str(entry.get("status", ""))
            for entry in job.logs
        ]
        if actual_log_statuses != expected_log_statuses:
            missing.append(
                f"job_log_statuses={expected_log_statuses}"
            )
    return missing


def _code_output_request_fields(case: dict[str, Any]) -> tuple[str, dict[str, str]]:
    language = str(case.get("language", "python"))
    entrypoint = str(case.get("entrypoint") or ("main.m" if language == "octave" else "main.py"))
    files = {
        str(path): str(content)
        for path, content in case.get("files", {}).items()
    }
    template_id = case.get("template_id")
    if not template_id:
        return entrypoint, files

    if language == "python":
        templates = PYTHON_EXECUTION_TEMPLATES
    elif language == "octave":
        templates = OCTAVE_EXECUTION_TEMPLATES
    else:
        raise ValueError(f"unsupported_code_output_language:{language}")

    template_id = str(template_id)
    if template_id not in templates:
        raise ValueError(f"unknown_code_output_template:{language}:{template_id}")
    files.setdefault(entrypoint, templates[template_id])
    return entrypoint, files


def _artifact_path_from_uri(uri: str) -> Path | None:
    parsed = urllib.parse.urlparse(uri)
    if parsed.scheme != "file":
        return None
    return Path(urllib.request.url2pathname(parsed.path))


def _evaluate_code_output_artifact(
    expected: dict[str, Any],
    artifacts: list[Any],
) -> CodeOutputArtifactResult:
    title = str(expected.get("title", ""))
    candidates = [
        artifact
        for artifact in artifacts
        if (not title or artifact.title == title)
        and (not expected.get("kind") or artifact.kind == expected["kind"])
        and (not expected.get("mime_type") or artifact.mime_type == expected["mime_type"])
    ]
    if not candidates:
        return CodeOutputArtifactResult(
            title=title,
            kind=str(expected.get("kind", "")),
            ok=False,
            message="artifact_missing",
        )

    artifact = candidates[0]
    minimum_bytes = int(expected.get("minimum_byte_count", 0))
    byte_count = int(artifact.metadata.get("byte_count", "0"))
    if byte_count < minimum_bytes:
        return CodeOutputArtifactResult(
            title=artifact.title or title,
            kind=artifact.kind,
            ok=False,
            message=f"byte_count={byte_count} minimum={minimum_bytes}",
        )

    contains = expected.get("contains")
    if contains:
        path = _artifact_path_from_uri(artifact.uri)
        if path is None or not path.exists():
            return CodeOutputArtifactResult(
                title=artifact.title or title,
                kind=artifact.kind,
                ok=False,
                message="artifact_file_missing",
            )
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return CodeOutputArtifactResult(
                title=artifact.title or title,
                kind=artifact.kind,
                ok=False,
                message="artifact_not_text",
            )
        if str(contains) not in text:
            return CodeOutputArtifactResult(
                title=artifact.title or title,
                kind=artifact.kind,
                ok=False,
                message="artifact_text_missing",
            )

    return CodeOutputArtifactResult(
        title=artifact.title or title,
        kind=artifact.kind,
        ok=True,
        message="ok",
    )


def evaluate_code_output_case(case: dict[str, Any]) -> CodeOutputCaseResult:
    """Run one no-key local code-output eval and verify stdout/artifacts."""
    language = str(case.get("language", "python"))
    execution_mode = str(case.get("execution_mode", "provider"))
    expected_artifacts = case.get("expected_artifacts", [])
    try:
        entrypoint, files = _code_output_request_fields(case)
    except ValueError as exc:
        return CodeOutputCaseResult(
            case_id=case["id"],
            ok=False,
            language=language,
            execution_mode=execution_mode,
            exit_code=2,
            stdout_ok=False,
            missing_stdout_terms=[],
            runtime_metadata_ok=False,
            missing_runtime_metadata=[],
            job_status=None,
            job_metadata_ok=False,
            missing_job_metadata=[],
            artifact_results=[],
            message=str(exc),
        )
    if not expected_artifacts:
        return CodeOutputCaseResult(
            case_id=case["id"],
            ok=False,
            language=language,
            execution_mode=execution_mode,
            exit_code=0,
            stdout_ok=False,
            missing_stdout_terms=[],
            runtime_metadata_ok=False,
            missing_runtime_metadata=[],
            job_status=None,
            job_metadata_ok=False,
            missing_job_metadata=[],
            artifact_results=[],
            message="expected_artifacts_missing",
        )

    with tempfile.TemporaryDirectory(prefix="fluxmind-eval-artifacts-") as tmp:
        request = CodeExecutionRequest(
            language=language,
            entrypoint=entrypoint,
            files=files,
            timeout_s=int(case.get("timeout_s", 30)),
            memory_mb=int(case.get("memory_mb", 512)),
        )
        job_status: str | None = None
        missing_job_metadata: list[str] = []
        if execution_mode == "provider":
            provider = _execution_provider(language, Path(tmp) / "artifacts")
            result = provider.run(request)
        elif execution_mode == "local_job":
            store = LocalJobStore(Path(tmp) / "jobs.jsonl")
            runner = LocalJobRunner(
                store,
                artifact_root=Path(tmp) / "artifacts",
                record_runtime_events=False,
            )
            job = runner.run_local_code(
                request,
                request_id=f"eval-{case['id']}",
            )
            job_status = job.status
            missing_job_metadata = _missing_code_output_job_metadata(case, job)
            result = _code_output_result_from_job(job)
        else:
            return CodeOutputCaseResult(
                case_id=case["id"],
                ok=False,
                language=language,
                execution_mode=execution_mode,
                exit_code=2,
                stdout_ok=False,
                missing_stdout_terms=[],
                runtime_metadata_ok=False,
                missing_runtime_metadata=[],
                job_status=None,
                job_metadata_ok=False,
                missing_job_metadata=[],
                artifact_results=[],
                message=f"unsupported_code_output_execution_mode:{execution_mode}",
            )
        required_stdout_terms = [str(term) for term in case.get("required_stdout_terms", [])]
        missing_stdout_terms = [
            term
            for term in required_stdout_terms
            if term not in result.stdout
        ]
        artifact_results = [
            _evaluate_code_output_artifact(expected, result.artifacts)
            for expected in expected_artifacts
        ]
        expected_metadata = {
            str(key): str(value)
            for key, value in case.get("expected_runtime_metadata", {}).items()
        }
        missing_runtime_metadata = [
            f"{key}={value}"
            for key, value in expected_metadata.items()
            if result.runtime_metadata.get(key) != value
        ]

    stdout_ok = not missing_stdout_terms
    runtime_metadata_ok = not missing_runtime_metadata
    job_metadata_ok = not missing_job_metadata
    artifact_ok = all(artifact.ok for artifact in artifact_results)
    ok = result.success and stdout_ok and runtime_metadata_ok and job_metadata_ok and artifact_ok
    if ok:
        message = f"ok mode={execution_mode} artifacts={len(artifact_results)}"
    else:
        artifact_failures = [
            f"{artifact.title}:{artifact.message}"
            for artifact in artifact_results
            if not artifact.ok
        ]
        message = (
            f"exit_code={result.exit_code} "
            f"missing_stdout_terms={missing_stdout_terms} "
            f"missing_runtime_metadata={missing_runtime_metadata} "
            f"missing_job_metadata={missing_job_metadata} "
            f"artifact_failures={artifact_failures}"
        )
    return CodeOutputCaseResult(
        case_id=case["id"],
        ok=ok,
        language=language,
        execution_mode=execution_mode,
        exit_code=result.exit_code,
        stdout_ok=stdout_ok,
        missing_stdout_terms=missing_stdout_terms,
        runtime_metadata_ok=runtime_metadata_ok,
        missing_runtime_metadata=missing_runtime_metadata,
        job_status=job_status,
        job_metadata_ok=job_metadata_ok,
        missing_job_metadata=missing_job_metadata,
        artifact_results=artifact_results,
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
) -> tuple[
    list[EvaluationCaseResult],
    list[RetrievalOnlyCaseResult],
    list[CodeOutputCaseResult],
    list[PdfStructureCaseResult],
    list[ProviderFixtureResult],
    list[RecordedAnswerResult],
]:
    """Evaluate all offline cases and provider fixtures from a config."""
    cases = [evaluate_case(case, project_root=project_root) for case in config.get("cases", [])]
    retrieval_only_cases = [
        evaluate_retrieval_only_case(case, project_root=project_root)
        for case in config.get("retrieval_only_cases", [])
    ]
    code_output_cases = [
        evaluate_code_output_case(case)
        for case in config.get("code_output_cases", [])
    ]
    pdf_structure_cases = [
        evaluate_pdf_structure_case(case, project_root=project_root)
        for case in config.get("pdf_structure_cases", [])
    ]
    provider_fixtures = [
        evaluate_provider_fixture(fixture)
        for fixture in config.get("provider_failure_fixtures", [])
    ]
    recorded_answers = [
        result
        for case in config.get("cases", [])
        if (result := evaluate_recorded_answer(case)) is not None
    ]
    return cases, retrieval_only_cases, code_output_cases, pdf_structure_cases, provider_fixtures, recorded_answers


def _pass_rate(results: list[Any]) -> float:
    if not results:
        return 0.0
    return sum(1 for result in results if result.ok) / len(results)


def _average_coverage(results: list[Any], attribute: str) -> float:
    if not results:
        return 0.0
    return sum(float(getattr(result, attribute)) for result in results) / len(results)


def _case_values(cases: list[dict[str, Any]], field: str) -> list[str]:
    values: list[str] = []
    for case in cases:
        raw_value = case.get(field, [])
        if isinstance(raw_value, str):
            candidates = [raw_value]
        else:
            candidates = list(raw_value)
        for candidate in candidates:
            value = str(candidate).strip()
            if value:
                values.append(value)
    return sorted(set(values))


def _topic_groups(config: dict[str, Any]) -> dict[str, list[str]]:
    groups = config.get("domain_ontology", {}).get("topic_groups", {})
    normalized: dict[str, list[str]] = {}
    for group_id, tags in groups.items():
        normalized[str(group_id)] = [str(tag).strip() for tag in tags if str(tag).strip()]
    return normalized


def _covered_topic_groups(config: dict[str, Any]) -> list[str]:
    case_tags = set(_case_values(retrieval_eval_cases(config), "topic_tags"))
    covered: list[str] = []
    for group_id, tags in _topic_groups(config).items():
        if case_tags & set(tags):
            covered.append(group_id)
    return sorted(covered)


def _gate_result(
    gate_id: str,
    *,
    ok: bool,
    actual: Any,
    expected: Any,
    missing: list[str] | None = None,
) -> RegressionGateResult:
    message = f"actual={actual} expected={expected}"
    if missing is not None:
        message = f"{message} missing={missing}"
    return RegressionGateResult(
        gate_id=gate_id,
        ok=ok,
        message=f"ok {message}" if ok else message,
    )


def evaluate_regression_gates(
    config: dict[str, Any],
    *,
    case_results: list[EvaluationCaseResult],
    provider_results: list[ProviderFixtureResult],
    recorded_results: list[RecordedAnswerResult],
    retrieval_only_results: list[RetrievalOnlyCaseResult] | None = None,
    code_output_results: list[CodeOutputCaseResult] | None = None,
    pdf_structure_results: list[PdfStructureCaseResult] | None = None,
    live_results: list[LiveAnswerResult] | None = None,
    live_retrieval_results: list[LiveRetrievalResult] | None = None,
) -> list[RegressionGateResult]:
    """Evaluate config-level breadth and quality gates for the RAG baseline."""
    gates = config.get("quality_gates", {})
    if not gates:
        return []

    cases = config.get("cases", [])
    retrieval_only_cases = config.get("retrieval_only_cases", [])
    code_output_cases = config.get("code_output_cases", [])
    pdf_structure_cases = config.get("pdf_structure_cases", [])
    all_retrieval_cases = retrieval_eval_cases(config)
    retrieval_only_results = retrieval_only_results or []
    code_output_results = code_output_results or []
    pdf_structure_results = pdf_structure_results or []
    results: list[RegressionGateResult] = []

    if "minimum_case_count" in gates:
        minimum = int(gates["minimum_case_count"])
        actual = len(cases)
        results.append(
            _gate_result(
                "minimum_case_count",
                ok=actual >= minimum,
                actual=actual,
                expected=f">={minimum}",
            )
        )

    required_modes = list(gates.get("required_answer_modes", []))
    if required_modes:
        actual_modes = sorted({case.get("answer_mode", "explanation") for case in cases})
        missing_modes = sorted(set(required_modes) - set(actual_modes))
        results.append(
            _gate_result(
                "required_answer_modes",
                ok=not missing_modes,
                actual=actual_modes,
                expected=required_modes,
                missing=missing_modes,
            )
        )

    if "minimum_topic_tag_count" in gates:
        minimum = int(gates["minimum_topic_tag_count"])
        actual_tags = _case_values(all_retrieval_cases, "topic_tags")
        results.append(
            _gate_result(
                "minimum_topic_tag_count",
                ok=len(actual_tags) >= minimum,
                actual=len(actual_tags),
                expected=f">={minimum}",
            )
        )

    required_topic_tags = list(gates.get("required_topic_tags", []))
    if required_topic_tags:
        actual_tags = _case_values(all_retrieval_cases, "topic_tags")
        missing_tags = sorted(set(required_topic_tags) - set(actual_tags))
        results.append(
            _gate_result(
                "required_topic_tags",
                ok=not missing_tags,
                actual=actual_tags,
                expected=required_topic_tags,
                missing=missing_tags,
            )
        )

    if "minimum_eval_lane_count" in gates:
        minimum = int(gates["minimum_eval_lane_count"])
        actual_lanes = _case_values(all_retrieval_cases, "eval_lanes")
        results.append(
            _gate_result(
                "minimum_eval_lane_count",
                ok=len(actual_lanes) >= minimum,
                actual=len(actual_lanes),
                expected=f">={minimum}",
            )
        )

    required_eval_lanes = list(gates.get("required_eval_lanes", []))
    if required_eval_lanes:
        actual_lanes = _case_values(all_retrieval_cases, "eval_lanes")
        missing_lanes = sorted(set(required_eval_lanes) - set(actual_lanes))
        results.append(
            _gate_result(
                "required_eval_lanes",
                ok=not missing_lanes,
                actual=actual_lanes,
                expected=required_eval_lanes,
                missing=missing_lanes,
            )
        )

    required_topic_groups = list(gates.get("required_topic_groups", []))
    if required_topic_groups:
        actual_groups = _covered_topic_groups(config)
        missing_groups = sorted(set(required_topic_groups) - set(actual_groups))
        results.append(
            _gate_result(
                "required_topic_groups",
                ok=not missing_groups,
                actual=actual_groups,
                expected=required_topic_groups,
                missing=missing_groups,
            )
        )

    if "minimum_expected_source_ref_count" in gates:
        minimum = int(gates["minimum_expected_source_ref_count"])
        actual = sum(len(case.get("expected_refs", [])) for case in all_retrieval_cases)
        results.append(
            _gate_result(
                "minimum_expected_source_ref_count",
                ok=actual >= minimum,
                actual=actual,
                expected=f">={minimum}",
            )
        )

    if "minimum_retrieval_only_case_count" in gates:
        minimum = int(gates["minimum_retrieval_only_case_count"])
        actual = len(retrieval_only_cases)
        results.append(
            _gate_result(
                "minimum_retrieval_only_case_count",
                ok=actual >= minimum,
                actual=actual,
                expected=f">={minimum}",
            )
        )

    if "minimum_retrieval_eval_question_count" in gates:
        minimum = int(gates["minimum_retrieval_eval_question_count"])
        actual = len(all_retrieval_cases)
        results.append(
            _gate_result(
                "minimum_retrieval_eval_question_count",
                ok=actual >= minimum,
                actual=actual,
                expected=f">={minimum}",
            )
        )

    if "minimum_retrieval_only_pass_rate" in gates:
        minimum = float(gates["minimum_retrieval_only_pass_rate"])
        actual = _pass_rate(retrieval_only_results)
        results.append(
            _gate_result(
                "minimum_retrieval_only_pass_rate",
                ok=actual >= minimum,
                actual=f"{actual:.2f}",
                expected=f">={minimum:.2f}",
            )
        )

    if "minimum_code_output_case_count" in gates:
        minimum = int(gates["minimum_code_output_case_count"])
        actual = len(code_output_cases)
        results.append(
            _gate_result(
                "minimum_code_output_case_count",
                ok=actual >= minimum,
                actual=actual,
                expected=f">={minimum}",
            )
        )

    required_code_languages = list(gates.get("required_code_output_languages", []))
    if required_code_languages:
        actual_languages = sorted({
            str(case.get("language", "python"))
            for case in code_output_cases
        })
        missing_languages = sorted(set(required_code_languages) - set(actual_languages))
        results.append(
            _gate_result(
                "required_code_output_languages",
                ok=not missing_languages,
                actual=actual_languages,
                expected=required_code_languages,
                missing=missing_languages,
            )
        )

    required_code_templates = list(gates.get("required_code_output_template_ids", []))
    if required_code_templates:
        actual_templates = sorted({
            str(case.get("template_id"))
            for case in code_output_cases
            if case.get("template_id")
        })
        missing_templates = sorted(set(required_code_templates) - set(actual_templates))
        results.append(
            _gate_result(
                "required_code_output_template_ids",
                ok=not missing_templates,
                actual=actual_templates,
                expected=required_code_templates,
                missing=missing_templates,
            )
        )

    required_code_execution_modes = list(gates.get("required_code_output_execution_modes", []))
    if required_code_execution_modes:
        actual_execution_modes = sorted({
            str(case.get("execution_mode", "provider"))
            for case in code_output_cases
        })
        missing_modes = sorted(set(required_code_execution_modes) - set(actual_execution_modes))
        results.append(
            _gate_result(
                "required_code_output_execution_modes",
                ok=not missing_modes,
                actual=actual_execution_modes,
                expected=required_code_execution_modes,
                missing=missing_modes,
            )
        )

    if "minimum_code_output_pass_rate" in gates:
        minimum = float(gates["minimum_code_output_pass_rate"])
        actual = _pass_rate(code_output_results)
        results.append(
            _gate_result(
                "minimum_code_output_pass_rate",
                ok=actual >= minimum,
                actual=f"{actual:.2f}",
                expected=f">={minimum:.2f}",
            )
        )

    if "minimum_pdf_structure_case_count" in gates:
        minimum = int(gates["minimum_pdf_structure_case_count"])
        actual = len(pdf_structure_cases)
        results.append(
            _gate_result(
                "minimum_pdf_structure_case_count",
                ok=actual >= minimum,
                actual=actual,
                expected=f">={minimum}",
            )
        )

    required_pdf_structure_kinds = list(gates.get("required_pdf_structure_kinds", []))
    if required_pdf_structure_kinds:
        actual_kinds = sorted({
            str(case.get("kind", "")).casefold()
            for case in pdf_structure_cases
            if case.get("kind")
        })
        missing_kinds = sorted(set(required_pdf_structure_kinds) - set(actual_kinds))
        results.append(
            _gate_result(
                "required_pdf_structure_kinds",
                ok=not missing_kinds,
                actual=actual_kinds,
                expected=required_pdf_structure_kinds,
                missing=missing_kinds,
            )
        )

    if "minimum_pdf_structure_pass_rate" in gates:
        minimum = float(gates["minimum_pdf_structure_pass_rate"])
        actual = _pass_rate(pdf_structure_results)
        results.append(
            _gate_result(
                "minimum_pdf_structure_pass_rate",
                ok=actual >= minimum,
                actual=f"{actual:.2f}",
                expected=f">={minimum:.2f}",
            )
        )

    if "minimum_provider_fixture_count" in gates:
        minimum = int(gates["minimum_provider_fixture_count"])
        actual = len(provider_results)
        results.append(
            _gate_result(
                "minimum_provider_fixture_count",
                ok=actual >= minimum,
                actual=actual,
                expected=f">={minimum}",
            )
        )

    if "minimum_recorded_answer_count" in gates:
        minimum = int(gates["minimum_recorded_answer_count"])
        actual = len(recorded_results)
        results.append(
            _gate_result(
                "minimum_recorded_answer_count",
                ok=actual >= minimum,
                actual=actual,
                expected=f">={minimum}",
            )
        )

    if "minimum_recorded_answer_pass_rate" in gates:
        minimum = float(gates["minimum_recorded_answer_pass_rate"])
        actual = _pass_rate(recorded_results)
        results.append(
            _gate_result(
                "minimum_recorded_answer_pass_rate",
                ok=actual >= minimum,
                actual=f"{actual:.2f}",
                expected=f">={minimum:.2f}",
            )
        )

    if "minimum_average_recorded_answer_term_coverage" in gates:
        minimum = float(gates["minimum_average_recorded_answer_term_coverage"])
        actual = _average_coverage(recorded_results, "coverage")
        results.append(
            _gate_result(
                "minimum_average_recorded_answer_term_coverage",
                ok=actual >= minimum,
                actual=f"{actual:.2f}",
                expected=f">={minimum:.2f}",
            )
        )

    if live_results is not None and "minimum_live_answer_pass_rate" in gates:
        minimum = float(gates["minimum_live_answer_pass_rate"])
        actual = _pass_rate(live_results)
        results.append(
            _gate_result(
                "minimum_live_answer_pass_rate",
                ok=actual >= minimum,
                actual=f"{actual:.2f}",
                expected=f">={minimum:.2f}",
            )
        )

    if (
        live_results is not None
        and "minimum_average_live_answer_term_coverage" in gates
    ):
        minimum = float(gates["minimum_average_live_answer_term_coverage"])
        actual = _average_coverage(live_results, "answer_term_coverage")
        results.append(
            _gate_result(
                "minimum_average_live_answer_term_coverage",
                ok=actual >= minimum,
                actual=f"{actual:.2f}",
                expected=f">={minimum:.2f}",
            )
        )

    if (
        live_retrieval_results is not None
        and "minimum_live_retrieval_pass_rate" in gates
    ):
        minimum = float(gates["minimum_live_retrieval_pass_rate"])
        actual = _pass_rate(live_retrieval_results)
        results.append(
            _gate_result(
                "minimum_live_retrieval_pass_rate",
                ok=actual >= minimum,
                actual=f"{actual:.2f}",
                expected=f">={minimum:.2f}",
            )
        )

    return results


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
    retrieval_only_results: list[RetrievalOnlyCaseResult] | None = None,
    code_output_results: list[CodeOutputCaseResult] | None = None,
    pdf_structure_results: list[PdfStructureCaseResult] | None = None,
    live_results: list[LiveAnswerResult] | None = None,
    live_retrieval_results: list[LiveRetrievalResult] | None = None,
    regression_gate_results: list[RegressionGateResult] | None = None,
    eval_file: Path | None = None,
) -> dict[str, Any]:
    """Build a no-secret machine-readable evaluation report."""
    retrieval_only_results = retrieval_only_results or []
    code_output_results = code_output_results or []
    pdf_structure_results = pdf_structure_results or []
    live_results = live_results or []
    live_retrieval_results = live_retrieval_results or []
    regression_gate_results = regression_gate_results or []
    all_retrieval_cases = retrieval_eval_cases(config)
    return {
        "schema_version": 1,
        "eval_file": str(eval_file) if eval_file else None,
        "case_count": len(config.get("cases", [])),
        "retrieval_only_case_count": len(config.get("retrieval_only_cases", [])),
        "retrieval_eval_question_count": len(all_retrieval_cases),
        "code_output_case_count": len(config.get("code_output_cases", [])),
        "pdf_structure_case_count": len(config.get("pdf_structure_cases", [])),
        "provider_fixture_count": len(config.get("provider_failure_fixtures", [])),
        "coverage": {
            "topic_tags": _case_values(all_retrieval_cases, "topic_tags"),
            "eval_lanes": _case_values(all_retrieval_cases, "eval_lanes"),
            "topic_groups": _covered_topic_groups(config),
        },
        "summary": {
            "offline_cases": _result_summary(case_results),
            "retrieval_only_cases": _result_summary(retrieval_only_results),
            "code_output_cases": _result_summary(code_output_results),
            "pdf_structure_cases": _result_summary(pdf_structure_results),
            "provider_fixtures": _result_summary(provider_results),
            "recorded_answers": _result_summary(recorded_results),
            "live_retrieval": _result_summary(live_retrieval_results),
            "live_answers": _result_summary(live_results),
            "regression_gates": _result_summary(regression_gate_results),
        },
        "results": {
            "offline_cases": [asdict(result) for result in case_results],
            "retrieval_only_cases": [asdict(result) for result in retrieval_only_results],
            "code_output_cases": [asdict(result) for result in code_output_results],
            "pdf_structure_cases": [asdict(result) for result in pdf_structure_results],
            "provider_fixtures": [asdict(result) for result in provider_results],
            "recorded_answers": [asdict(result) for result in recorded_results],
            "live_retrieval": [asdict(result) for result in live_retrieval_results],
            "live_answers": [asdict(result) for result in live_results],
            "regression_gates": [
                asdict(result)
                for result in regression_gate_results
            ],
        },
    }
