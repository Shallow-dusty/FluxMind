"""Local embedding model using sentence-transformers."""

from functools import lru_cache

from src.config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_embedding_model():
    """Get the local embedding model (cached after first load)."""
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
