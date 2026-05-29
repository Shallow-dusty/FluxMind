"""RAG chain: retrieval + LLM generation."""

from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from src.config import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, TOP_K, FAISS_INDEX_DIR
from src.embeddings import get_embedding_model
from src.runtime import ProviderError

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

## Retrieved Context:
{context}
"""

USER_TEMPLATE = "{question}"


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


def query(question: str, chat_history: list[dict] | None = None) -> str:
    """Run RAG query: retrieve relevant chunks -> generate answer."""
    # Retrieve
    store = get_vector_store()
    context_docs = []
    if store:
        context_docs = store.similarity_search(question, k=TOP_K)

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
        response = chain.invoke({"context": context, "question": question})
    except Exception as exc:
        raise ProviderError(str(exc)) from exc
    return response.content


def query_stream(question: str):
    """Streaming RAG. Surfaces `reasoning_content` (for reasoning models like MiMo) as a
    blockquoted thinking block before the final answer."""
    store = get_vector_store()
    context_docs = store.similarity_search(question, k=TOP_K) if store else []
    context = format_context(context_docs)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
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
        raise ProviderError(str(exc)) from exc
