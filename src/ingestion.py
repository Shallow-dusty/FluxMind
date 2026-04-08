"""PDF ingestion pipeline: load → split → embed → store in FAISS."""

from pathlib import Path

import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

from src.config import PAPERS_DIR, FAISS_INDEX_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from src.embeddings import get_embedding_model


def load_pdf(path: Path) -> list[Document]:
    """Extract text from a single PDF using PyMuPDF."""
    doc = fitz.open(str(path))
    documents = []
    for page_num, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            documents.append(
                Document(
                    page_content=text,
                    metadata={"source": path.name, "page": page_num + 1},
                )
            )
    doc.close()
    return documents


def load_all_pdfs(directory: Path | None = None) -> list[Document]:
    """Load all PDFs from the papers directory."""
    directory = directory or PAPERS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    all_docs = []
    for pdf_path in sorted(directory.glob("*.pdf")):
        all_docs.extend(load_pdf(pdf_path))
    return all_docs


def split_documents(documents: list[Document]) -> list[Document]:
    """Split documents into chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def build_vector_store(documents: list[Document] | None = None) -> FAISS:
    """Build or load FAISS vector store."""
    embeddings = get_embedding_model()
    index_path = FAISS_INDEX_DIR

    # If index exists, load it
    if index_path.exists() and (index_path / "index.faiss").exists():
        store = FAISS.load_local(
            str(index_path), embeddings, allow_dangerous_deserialization=True
        )
        # If new documents provided, add them
        if documents:
            chunks = split_documents(documents)
            if chunks:
                store.add_documents(chunks)
                store.save_local(str(index_path))
        return store

    # Build from scratch
    if documents is None:
        documents = load_all_pdfs()

    if not documents:
        # Create empty store with a placeholder
        store = FAISS.from_documents(
            [Document(page_content="FluxMind knowledge base initialized.", metadata={"source": "system"})],
            embeddings,
        )
        store.save_local(str(index_path))
        return store

    chunks = split_documents(documents)
    store = FAISS.from_documents(chunks, embeddings)
    store.save_local(str(index_path))
    return store


def ingest_uploaded_pdf(pdf_bytes: bytes, filename: str) -> int:
    """Ingest a single uploaded PDF into the vector store. Returns chunk count."""
    # Save PDF to papers dir
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = PAPERS_DIR / filename
    pdf_path.write_bytes(pdf_bytes)

    # Load and add to store
    docs = load_pdf(pdf_path)
    chunks = split_documents(docs)

    embeddings = get_embedding_model()
    FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)

    if (FAISS_INDEX_DIR / "index.faiss").exists():
        store = FAISS.load_local(
            str(FAISS_INDEX_DIR), embeddings, allow_dangerous_deserialization=True
        )
        store.add_documents(chunks)
    else:
        store = FAISS.from_documents(chunks, embeddings)

    store.save_local(str(FAISS_INDEX_DIR))
    return len(chunks)
