"""RAG chain: retrieval + LLM generation."""

from __future__ import annotations

import math
import re
import threading
from collections import Counter
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from src.artifacts import LocalArtifactRegistry, format_artifact_references
from src.config import (
    FAISS_INDEX_DIR,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    PROJECT_ROOT,
    RERANKER_MODEL,
    TOP_K,
)
from src.embeddings import get_embedding_model
from src.runtime import ProviderError, normalize_exception

AnswerMode = Literal["explanation", "derivation", "implementation", "literature_review", "code_generation"]

DEFAULT_ANSWER_MODE: AnswerMode = "explanation"

ANSWER_MODE_INSTRUCTIONS: dict[AnswerMode, str] = {
    "explanation": "Explain the concept clearly, define assumptions, and keep citations close to claims.",
    "derivation": "Prioritize equations, derivation steps, assumptions, and cite the source context for each key step.",
    "implementation": "Focus on implementation guidance, parameters, engineering tradeoffs, and reproducible steps.",
    "literature_review": "Compare papers, methods, evidence, and research gaps with source/page citations.",
    "code_generation": (
        "Generate MATLAB/Simulink-oriented code by default. For observer code, "
        "name measured currents, voltage inputs, observer gains, and the "
        "switching or saturation function, then cite the relevant context."
    ),
}

SYSTEM_PROMPT = """\
You are FluxMind, an expert research copilot specializing in:
- Sliding Mode Control (SMC): reaching law design, chattering reduction, higher-order SMC
- Flux Linkage Estimation: observer design, MRAS, extended Kalman filters for motor drives
- MATLAB/Simulink modeling for control systems

## Rules:
1. When answering theoretical questions, ALWAYS cite the source paper/page from the retrieved context.
2. When generating code, use MATLAB syntax by default. Use Python only if explicitly requested.
3. Structure your answers clearly with sections and equations (use LaTeX notation).
4. If the retrieved context doesn't contain relevant information, say so honestly and provide your best knowledge.
5. Answer in the SAME LANGUAGE as the user's question (Chinese or English).
6. Only cite retrieved context refs listed below. Do not invent numbered refs, bibliography numbers, or page refs.

## Answer Mode:
{answer_mode}

{mode_instruction}

{citation_instruction}

## Retrieved Context:
{context}

## Generated Artifact References:
{artifact_context}

If a generated diagram, plot, or file is relevant, cite it by its stable
`[Artifact:<id>]` reference. Do not invent artifact IDs.
"""

USER_TEMPLATE = "{question}"
_BRACKET_CITATION_RE = re.compile(r"(?<!\!)\[(\d+)\]")
_TOKEN_RE = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)
_VECTOR_STORE_LOCK = threading.RLock()
_VECTOR_STORE_CACHE: dict[str, Any] = {"signature": None, "store": None}


@dataclass(frozen=True)
class CitationValidation:
    """Citation validation result for numbered retrieved-context references."""

    cited_refs: list[int]
    valid_refs: list[int]
    invalid_refs: list[int]
    missing_required_refs: list[int]
    missing_source_page_refs: list[int]

    @property
    def ok(self) -> bool:
        return (
            not self.invalid_refs
            and not self.missing_required_refs
            and not self.missing_source_page_refs
        )

    def to_dict(self) -> dict:
        return asdict(self) | {"ok": self.ok}


@dataclass(frozen=True)
class QueryResult:
    """RAG answer with retrieved-context citation verification metadata."""

    answer: str
    answer_mode: AnswerMode
    citation_validation: CitationValidation
    context_refs: list[dict]
    provider_usage: dict[str, int] | None = None

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "answer_mode": self.answer_mode,
            "citation_validation": self.citation_validation.to_dict(),
            "context_refs": self.context_refs,
            "provider_usage": self.provider_usage,
        }


@dataclass(frozen=True)
class RetrievalDiagnostics:
    """No-LLM retrieval diagnostics for source/page quality gates."""

    answer_mode: AnswerMode
    context_count: int
    citation_instruction: str
    context_refs: list[dict]
    missing_source_page_refs: list[int]

    @property
    def ok(self) -> bool:
        return self.context_count > 0 and not self.missing_source_page_refs

    def to_dict(self) -> dict:
        return {
            "answer_mode": self.answer_mode,
            "context_count": self.context_count,
            "citation_instruction": self.citation_instruction,
            "context_refs": self.context_refs,
            "missing_source_page_refs": self.missing_source_page_refs,
            "ok": self.ok,
        }


def get_llm() -> ChatOpenAI:
    """Get the LLM instance."""
    return ChatOpenAI(
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        model=LLM_MODEL,
        temperature=0.3,
        max_tokens=4096,
    )


def _vector_store_signature(index_path: Path | None = None) -> tuple[tuple[str, int, int], ...] | None:
    """Return a cheap freshness signature for the persisted FAISS store."""
    index_path = index_path or FAISS_INDEX_DIR
    index_file = index_path / "index.faiss"
    if not index_file.exists():
        return None
    signature = []
    for name in ("index.faiss", "index.pkl"):
        path = index_path / name
        if not path.exists():
            return None
        stat = path.stat()
        signature.append((name, stat.st_mtime_ns, stat.st_size))
    return tuple(signature)


def clear_vector_store_cache() -> None:
    """Drop the in-process FAISS cache after a local index mutation."""
    with _VECTOR_STORE_LOCK:
        _VECTOR_STORE_CACHE["signature"] = None
        _VECTOR_STORE_CACHE["store"] = None


def get_vector_store() -> FAISS | None:
    """Load the existing FAISS index and reuse it until the files change."""
    signature = _vector_store_signature()
    if signature is None:
        clear_vector_store_cache()
        return None
    with _VECTOR_STORE_LOCK:
        if _VECTOR_STORE_CACHE["signature"] == signature:
            return _VECTOR_STORE_CACHE["store"]
        from langchain_community.vectorstores import FAISS

        store = FAISS.load_local(
            str(FAISS_INDEX_DIR),
            get_embedding_model(),
            allow_dangerous_deserialization=True,
        )
        _VECTOR_STORE_CACHE["signature"] = signature
        _VECTOR_STORE_CACHE["store"] = store
        return store


def tokenize_query(text: str) -> set[str]:
    """Tokenize Latin/CJK text for lightweight local keyword retrieval."""
    return {
        token.lower()
        for token in _TOKEN_RE.findall(text)
        if len(token.strip()) >= 2
    }


def tokenize_terms(text: str) -> list[str]:
    """Return normalized search terms while preserving term frequency."""
    return [
        token.lower()
        for token in _TOKEN_RE.findall(text)
        if len(token.strip()) >= 2
    ]


def document_search_text(doc: Document) -> str:
    """Combine chunk content and metadata for local lexical scoring."""
    metadata_text = " ".join(str(value) for value in doc.metadata.values())
    return f"{doc.page_content}\n{metadata_text}"


def iter_index_documents(store: FAISS) -> list[Document]:
    """Best-effort document extraction from a LangChain FAISS store."""
    docstore = getattr(store, "docstore", None)
    raw_docs = getattr(docstore, "_dict", None)
    if isinstance(raw_docs, dict):
        return [doc for doc in raw_docs.values() if isinstance(doc, Document)]
    return []


def keyword_search_documents(store: FAISS, question: str, *, k: int) -> list[Document]:
    """Return locally BM25-matched chunks from the FAISS docstore."""
    docs = iter_index_documents(store)
    scores = bm25_relevance_scores(question, docs)
    if not scores:
        return []

    scored = [
        (score, -index, docs[index])
        for index, score in enumerate(scores)
        if score > 0
    ]
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [doc for _score, _index, doc in scored[:k]]


def document_key(doc: Document) -> tuple[str, str, str]:
    """Stable-ish dedupe key for retrieved chunks."""
    return (
        str(doc.metadata.get("source_path") or doc.metadata.get("source") or ""),
        str(doc.metadata.get("page") or ""),
        doc.page_content[:160],
    )


def lexical_relevance_score(question: str, doc: Document) -> float:
    """Score one chunk by local BM25-lite relevance to the query."""
    return bm25_relevance_score(question, doc)


def bm25_relevance_score(question: str, doc: Document, corpus: list[Document] | None = None) -> float:
    """Score one chunk by local BM25-lite relevance to the query."""
    docs = corpus or [doc]
    try:
        index = docs.index(doc)
    except ValueError:
        docs = [doc, *docs]
        index = 0
    scores = bm25_relevance_scores(question, docs)
    return scores[index] if scores else 0.0


def bm25_relevance_scores(question: str, docs: list[Document]) -> list[float]:
    """Return deterministic no-key BM25-lite scores for candidate chunks."""
    query_terms = set(tokenize_terms(question))
    if not query_terms:
        return [0.0 for _doc in docs]

    tokenized_docs = [tokenize_terms(document_search_text(doc)) for doc in docs]
    if not tokenized_docs:
        return []

    lengths = [len(tokens) for tokens in tokenized_docs]
    avg_len = sum(lengths) / len(lengths) if lengths else 0.0
    if avg_len <= 0:
        return [0.0 for _doc in docs]

    document_frequency = Counter(
        term
        for tokens in tokenized_docs
        for term in set(tokens)
        if term in query_terms
    )
    total_docs = len(tokenized_docs)
    k1 = 1.5
    b = 0.75
    scores: list[float] = []
    for tokens, length in zip(tokenized_docs, lengths):
        counts = Counter(tokens)
        score = 0.0
        for term in query_terms:
            tf = counts.get(term, 0)
            if tf <= 0:
                continue
            df = document_frequency.get(term, 0)
            idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
            denominator = tf + k1 * (1 - b + b * (length / avg_len))
            score += idf * ((tf * (k1 + 1)) / denominator)
        scores.append(score)
    return scores


def configured_reranker_path() -> Path | None:
    """Return an existing local reranker model path, or None when disabled."""
    if not RERANKER_MODEL:
        return None
    path = Path(RERANKER_MODEL).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path if path.exists() else None


@lru_cache(maxsize=1)
def get_cross_encoder_reranker(model_path: str):
    """Lazy-load an optional local CrossEncoder reranker."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_path)


def cross_encoder_relevance_scores(question: str, docs: list[Document]) -> list[float] | None:
    """Return optional local learned reranker scores without runtime downloads."""
    model_path = configured_reranker_path()
    if model_path is None or not docs:
        return None

    model = get_cross_encoder_reranker(str(model_path))
    raw_scores = model.predict([(question, document_search_text(doc)) for doc in docs])
    scores = [float(score) for score in raw_scores]
    if len(scores) != len(docs):
        return None
    return scores


def select_source_diverse_documents(
    scored_docs: list[tuple[float, int, Document]],
    *,
    k: int,
    min_score: float | None = None,
) -> list[Document]:
    """Select top scored docs while preserving source diversity on the first pass."""
    scored = sorted(scored_docs, key=lambda item: (item[0], item[1]), reverse=True)
    selected: list[Document] = []
    selected_keys: set[tuple[str, str, str]] = set()
    seen_sources: set[str] = set()

    for score, _index, doc in scored:
        if len(selected) >= k:
            break
        if min_score is not None and score <= min_score:
            continue
        key = document_key(doc)
        source = key[0]
        if not source or source in seen_sources or key in selected_keys:
            continue
        selected.append(doc)
        selected_keys.add(key)
        seen_sources.add(source)

    for score, _index, doc in scored:
        if len(selected) >= k:
            break
        key = document_key(doc)
        if key in selected_keys:
            continue
        selected.append(doc)
        selected_keys.add(key)

    return selected


def rerank_documents(question: str, docs: list[Document], *, k: int = TOP_K) -> list[Document]:
    """Deterministic no-key BM25-lite reranker with first-pass source diversity."""
    scores = bm25_relevance_scores(question, docs)
    scored = [
        (scores[index], -index, doc)
        for index, doc in enumerate(docs)
    ]
    return select_source_diverse_documents(scored, k=k, min_score=0.0)


def learned_rerank_documents(question: str, docs: list[Document], *, k: int = TOP_K) -> list[Document] | None:
    """Optional no-key local CrossEncoder reranker with BM25 fallback outside this function."""
    scores = cross_encoder_relevance_scores(question, docs)
    if scores is None:
        return None
    scored = [
        (scores[index], -index, doc)
        for index, doc in enumerate(docs)
    ]
    return select_source_diverse_documents(scored, k=k)


def hybrid_retrieve(question: str, *, k: int = TOP_K) -> list[Document]:
    """Retrieve context with vector search plus local keyword supplementation."""
    store = get_vector_store()
    if store is None:
        return []

    candidate_k = max(k * 4, k)
    vector_docs = store.similarity_search(question, k=candidate_k)
    keyword_docs = keyword_search_documents(store, question, k=candidate_k)
    merged: list[Document] = []
    seen: set[tuple[str, str, str]] = set()
    for doc in vector_docs + keyword_docs:
        key = document_key(doc)
        if key in seen:
            continue
        seen.add(key)
        merged.append(doc)
    learned_docs = learned_rerank_documents(question, merged, k=k)
    if learned_docs is not None:
        return learned_docs
    return rerank_documents(question, merged, k=k)


def format_context(docs: list[Document]) -> str:
    """Format retrieved documents into context string."""
    if not docs:
        return "(No relevant documents found in the knowledge base.)"

    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        parts.append(f"[{i}] Source: {source}, Page {page}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def citation_instruction(context_count: int) -> str:
    """Return a prompt guard for valid numbered context references."""
    if context_count <= 0:
        return "No numbered source refs are available; do not use numbered citations like [1]."
    if context_count == 1:
        return "Valid numbered source refs for this answer: [1] only."
    return f"Valid numbered source refs for this answer: [1] through [{context_count}] only."


def generated_artifact_context(*, limit: int = 5) -> str:
    """Return recent generated artifact references for optional answer citation."""
    try:
        artifacts = LocalArtifactRegistry().list_artifacts(limit=limit)
    except Exception:
        return "(Generated artifact registry is unavailable.)"
    return format_artifact_references(artifacts, limit=limit)


def normalize_answer_mode(answer_mode: str | None) -> AnswerMode:
    """Return a supported answer mode, defaulting to explanation."""
    if answer_mode in ANSWER_MODE_INSTRUCTIONS:
        return answer_mode  # type: ignore[return-value]
    return DEFAULT_ANSWER_MODE


def validate_numbered_citations(
    answer: str,
    docs: list[Document],
    *,
    required_refs: list[int] | None = None,
) -> CitationValidation:
    """Validate bracket citations like [1] against retrieved document refs."""
    max_ref = len(docs)
    cited_refs = sorted({int(match) for match in _BRACKET_CITATION_RE.findall(answer)})
    valid_refs = [ref for ref in cited_refs if 1 <= ref <= max_ref]
    invalid_refs = [ref for ref in cited_refs if ref < 1 or ref > max_ref]
    required = required_refs or []
    missing_required_refs = [ref for ref in required if ref not in cited_refs]
    missing_source_page_refs = [
        ref
        for ref in valid_refs
        if not has_source_page_metadata(docs[ref - 1])
    ]
    return CitationValidation(
        cited_refs=cited_refs,
        valid_refs=valid_refs,
        invalid_refs=invalid_refs,
        missing_required_refs=missing_required_refs,
        missing_source_page_refs=missing_source_page_refs,
    )


def neutralize_invalid_numbered_citations(answer: str, *, max_ref: int) -> str:
    """Convert out-of-range bracket refs so they cannot masquerade as context refs."""
    if max_ref <= 0:
        return _BRACKET_CITATION_RE.sub(lambda match: f"(source-internal ref {match.group(1)})", answer)

    def replace(match: re.Match[str]) -> str:
        ref = int(match.group(1))
        if 1 <= ref <= max_ref:
            return match.group(0)
        return f"(source-internal ref {ref})"

    return _BRACKET_CITATION_RE.sub(replace, answer)


def has_source_page_metadata(doc: Document) -> bool:
    metadata = doc.metadata or {}
    has_source = bool(metadata.get("source") or metadata.get("source_path"))
    page = metadata.get("page")
    return has_source and page not in {None, "", "?"}


def context_ref_dicts(docs: list[Document]) -> list[dict]:
    """Return source/page metadata for numbered retrieved context refs."""
    refs = []
    for index, doc in enumerate(docs, 1):
        metadata = doc.metadata or {}
        refs.append(
            {
                "ref": index,
                "source": metadata.get("source"),
                "source_path": metadata.get("source_path"),
                "page": metadata.get("page"),
                "preview": " ".join(doc.page_content.split())[:240],
            }
        )
    return refs


def provider_usage_from_response(response: Any) -> dict[str, int] | None:
    """Extract no-secret provider token usage from a LangChain response when available."""
    usage_metadata = getattr(response, "usage_metadata", None) or {}
    response_metadata = getattr(response, "response_metadata", None) or {}
    token_usage = response_metadata.get("token_usage") or response_metadata.get("usage") or {}

    prompt_tokens = (
        usage_metadata.get("input_tokens")
        or usage_metadata.get("prompt_tokens")
        or token_usage.get("prompt_tokens")
        or token_usage.get("input_tokens")
    )
    completion_tokens = (
        usage_metadata.get("output_tokens")
        or usage_metadata.get("completion_tokens")
        or token_usage.get("completion_tokens")
        or token_usage.get("output_tokens")
    )
    total_tokens = (
        usage_metadata.get("total_tokens")
        or token_usage.get("total_tokens")
    )

    usage: dict[str, int] = {}
    if prompt_tokens is not None:
        usage["prompt_tokens"] = int(prompt_tokens)
    if completion_tokens is not None:
        usage["completion_tokens"] = int(completion_tokens)
    if total_tokens is not None:
        usage["total_tokens"] = int(total_tokens)
    elif "prompt_tokens" in usage or "completion_tokens" in usage:
        usage["total_tokens"] = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
    return usage or None


def retrieve_with_metadata(
    question: str,
    *,
    answer_mode: str | None = DEFAULT_ANSWER_MODE,
    k: int = TOP_K,
) -> RetrievalDiagnostics:
    """Retrieve context refs without calling an LLM provider."""
    mode = normalize_answer_mode(answer_mode)
    context_docs = hybrid_retrieve(question, k=k)
    missing_source_page_refs = [
        index
        for index, doc in enumerate(context_docs, 1)
        if not has_source_page_metadata(doc)
    ]
    return RetrievalDiagnostics(
        answer_mode=mode,
        context_count=len(context_docs),
        citation_instruction=citation_instruction(len(context_docs)),
        context_refs=context_ref_dicts(context_docs),
        missing_source_page_refs=missing_source_page_refs,
    )


def query_with_metadata(
    question: str,
    chat_history: list[dict] | None = None,
    *,
    answer_mode: str | None = DEFAULT_ANSWER_MODE,
) -> QueryResult:
    """Run RAG query and validate generated numbered citations."""
    del chat_history
    mode = normalize_answer_mode(answer_mode)

    # Retrieve
    context_docs = hybrid_retrieve(question, k=TOP_K)
    context = format_context(context_docs)
    artifact_context = generated_artifact_context()

    # Build prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", USER_TEMPLATE),
    ])

    # Generate
    llm = get_llm()
    chain = prompt | llm
    try:
        response = chain.invoke(
            {
                "context": context,
                "artifact_context": artifact_context,
                "question": question,
                "answer_mode": mode,
                "mode_instruction": ANSWER_MODE_INSTRUCTIONS[mode],
                "citation_instruction": citation_instruction(len(context_docs)),
            }
        )
    except Exception as exc:
        error = normalize_exception(exc)
        raise ProviderError(error.message, code=error.code, status_code=error.status_code) from exc
    answer = neutralize_invalid_numbered_citations(response.content, max_ref=len(context_docs))
    return QueryResult(
        answer=answer,
        answer_mode=mode,
        citation_validation=validate_numbered_citations(answer, context_docs),
        context_refs=context_ref_dicts(context_docs),
        provider_usage=provider_usage_from_response(response),
    )


def query(
    question: str,
    chat_history: list[dict] | None = None,
    *,
    answer_mode: str | None = DEFAULT_ANSWER_MODE,
) -> str:
    """Run RAG query: retrieve relevant chunks -> generate answer."""
    return query_with_metadata(
        question,
        chat_history,
        answer_mode=answer_mode,
    ).answer


def query_stream(question: str, *, answer_mode: str | None = DEFAULT_ANSWER_MODE):
    """Streaming RAG. Surfaces `reasoning_content` (for reasoning models like MiMo) as a
    blockquoted thinking block before the final answer."""
    mode = normalize_answer_mode(answer_mode)
    context_docs = hybrid_retrieve(question, k=TOP_K)
    context = format_context(context_docs)
    artifact_context = generated_artifact_context()

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT.format(
                context=context,
                artifact_context=artifact_context,
                answer_mode=mode,
                mode_instruction=ANSWER_MODE_INSTRUCTIONS[mode],
                citation_instruction=citation_instruction(len(context_docs)),
            ),
        },
        {"role": "user", "content": question},
    ]

    client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
    try:
        stream = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=4096,
            stream=True,
        )

        reasoning_started = False
        answer_started = False
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            reasoning_piece = getattr(delta, "reasoning_content", None)
            content_piece = getattr(delta, "content", None)

            if reasoning_piece:
                if not reasoning_started:
                    yield "> 💭 "
                    reasoning_started = True
                yield reasoning_piece.replace("\n", "\n> ")

            if content_piece:
                if reasoning_started and not answer_started:
                    yield "\n\n---\n\n"
                answer_started = True
                yield content_piece
    except Exception as exc:
        error = normalize_exception(exc)
        raise ProviderError(error.message, code=error.code, status_code=error.status_code) from exc
