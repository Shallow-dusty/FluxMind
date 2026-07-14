"""Local embedding model factory."""

import hashlib
import math
import re
from functools import lru_cache

from langchain_core.embeddings import Embeddings

from src.config import EMBEDDING_MODEL

_TOKEN_RE = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)
_HASH_EMBEDDING_DIMENSIONS = 384


class HashEmbeddings(Embeddings):
    """Small deterministic embedding fallback for local development and indexing.

    This is not a semantic model. It keeps FAISS-backed workflows usable when
    optional sentence-transformers dependencies or model files are unavailable.
    Production deployments should still configure the real HuggingFace model.
    """

    def __init__(self, *, dimensions: int = _HASH_EMBEDDING_DIMENSIONS, model_name: str = "hash-fallback"):
        self.dimensions = max(int(dimensions), 8)
        self.model_name = model_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = _TOKEN_RE.findall(text.casefold()) or [text.casefold()[:64]]
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


@lru_cache(maxsize=1)
def get_embedding_model():
    """Get the configured embedding model, with a lightweight local fallback."""
    try:
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception:
        return HashEmbeddings(model_name=f"hash-fallback:{EMBEDDING_MODEL}")
