"""Local embedding model using sentence-transformers."""

from langchain_community.embeddings import HuggingFaceEmbeddings
from src.config import EMBEDDING_MODEL


def get_embedding_model() -> HuggingFaceEmbeddings:
    """Get the local embedding model (cached after first load)."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
