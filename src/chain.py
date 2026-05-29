"""RAG chain: retrieval + LLM generation."""

import re
from dataclasses import dataclass
from typing import Literal

from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from src.config import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, TOP_K, FAISS_INDEX_DIR
from src.embeddings import get_embedding_model
from src.runtime import ProviderError, normalize_exception

AnswerMode = Literal["explanation", "derivation", "implementation", "literature_review", "code_generation"]

DEFAULT_ANSWER_MODE: AnswerMode = "explanation"

ANSWER_MODE_INSTRUCTIONS: dict[AnswerMode, str] = {
    "explanation": "Explain the concept clearly, define assumptions, and keep citations close to claims.",
    "derivation": "Prioritize equations, derivation steps, assumptions, and cite the source context for each key step.",
    "implementation": "Focus on implementation guidance, parameters, engineering tradeoffs, and reproducible steps.",
    "literature_review": "Compare papers, methods, evidence, and research gaps with source/page citations.",
    "code_generation": "Generate MATLAB/Simulink-oriented code by default, explain interfaces, and cite relevant context.",
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

## Answer Mode:
{answer_mode}

{mode_instruction}

## Retrieved Context:
{context}
"""

USER_TEMPLATE = "{question}"
_BRACKET_CITATION_RE = re.compile(r"(?<!\!)\[(\d+)\]")
_TOKEN_RE = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)


@dataclass(frozen=True)
class CitationValidation:
    """Citation validation result for numbered retrieved-context references."""

    cited_refs: list[int]
    valid_refs: list[int]
    invalid_refs: list[int]
    missing_required_refs: list[int]

    @property
    def ok(self) -> bool:
        return not self.invalid_refs and not self.missing_required_refs


def get_llm() -> ChatOpenAI:
    """Get the LLM instance."""
    return ChatOpenAI(
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        model=LLM_MODEL,
        temperature=0.3,
        max_tokens=4096,
    )


def get_vector_store() -> FAISS | None:
    """Load existing FAISS index if available."""
    index_path = FAISS_INDEX_DIR
    if index_path.exists() and (index_path / "index.faiss").exists():
        return FAISS.load_local(
            str(index_path),
            get_embedding_model(),
            allow_dangerous_deserialization=True,
        )
    return None


def tokenize_query(text: str) -> set[str]:
    """Tokenize Latin/CJK text for lightweight local keyword retrieval."""
    return {
        token.lower()
        for token in _TOKEN_RE.findall(text)
        if len(token.strip()) >= 2
    }


def iter_index_documents(store: FAISS) -> list[Document]:
    """Best-effort document extraction from a LangChain FAISS store."""
    docstore = getattr(store, "docstore", None)
    raw_docs = getattr(docstore, "_dict", None)
    if isinstance(raw_docs, dict):
        return [doc for doc in raw_docs.values() if isinstance(doc, Document)]
    return []


def keyword_search_documents(store: FAISS, question: str, *, k: int) -> list[Document]:
    """Return locally keyword-matched chunks from the FAISS docstore."""
    query_terms = tokenize_query(question)
    if not query_terms:
        return []

    scored: list[tuple[int, int, Document]] = []
    for index, doc in enumerate(iter_index_documents(store)):
        content_terms = tokenize_query(doc.page_content)
        metadata_terms = tokenize_query(
            " ".join(str(value) for value in doc.metadata.values())
        )
        score = len(query_terms & content_terms) * 2 + len(query_terms & metadata_terms)
        if score:
            scored.append((score, -index, doc))

    scored.sort(reverse=True)
    return [doc for _score, _index, doc in scored[:k]]


def document_key(doc: Document) -> tuple[str, str, str]:
    """Stable-ish dedupe key for retrieved chunks."""
    return (
        str(doc.metadata.get("source_path") or doc.metadata.get("source") or ""),
        str(doc.metadata.get("page") or ""),
        doc.page_content[:160],
    )


def hybrid_retrieve(question: str, *, k: int = TOP_K) -> list[Document]:
    """Retrieve context with vector search plus local keyword supplementation."""
    store = get_vector_store()
    if store is None:
        return []

    vector_docs = store.similarity_search(question, k=k)
    keyword_docs = keyword_search_documents(store, question, k=k)
    merged: list[Document] = []
    seen: set[tuple[str, str, str]] = set()
    for doc in vector_docs + keyword_docs:
        key = document_key(doc)
        if key in seen:
            continue
        seen.add(key)
        merged.append(doc)
        if len(merged) >= k:
            break
    return merged


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
    return CitationValidation(
        cited_refs=cited_refs,
        valid_refs=valid_refs,
        invalid_refs=invalid_refs,
        missing_required_refs=missing_required_refs,
    )


def query(
    question: str,
    chat_history: list[dict] | None = None,
    *,
    answer_mode: str | None = DEFAULT_ANSWER_MODE,
) -> str:
    """Run RAG query: retrieve relevant chunks -> generate answer."""
    del chat_history
    mode = normalize_answer_mode(answer_mode)

    # Retrieve
    context_docs = hybrid_retrieve(question, k=TOP_K)
    context = format_context(context_docs)

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
                "question": question,
                "answer_mode": mode,
                "mode_instruction": ANSWER_MODE_INSTRUCTIONS[mode],
            }
        )
    except Exception as exc:
        error = normalize_exception(exc)
        raise ProviderError(error.message, code=error.code, status_code=error.status_code) from exc
    return response.content


def query_stream(question: str, *, answer_mode: str | None = DEFAULT_ANSWER_MODE):
    """Streaming RAG. Surfaces `reasoning_content` (for reasoning models like MiMo) as a
    blockquoted thinking block before the final answer."""
    mode = normalize_answer_mode(answer_mode)
    context_docs = hybrid_retrieve(question, k=TOP_K)
    context = format_context(context_docs)

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT.format(
                context=context,
                answer_mode=mode,
                mode_instruction=ANSWER_MODE_INSTRUCTIONS[mode],
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
