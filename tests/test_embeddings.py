import sys
import types

from src import embeddings


def test_get_embedding_model_uses_cpu_normalized_cached_instance(monkeypatch):
    calls = []

    class FakeEmbeddings:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "langchain_huggingface",
        types.SimpleNamespace(HuggingFaceEmbeddings=FakeEmbeddings),
    )
    embeddings.get_embedding_model.cache_clear()

    first = embeddings.get_embedding_model()
    second = embeddings.get_embedding_model()

    assert first is second
    assert calls == [
        {
            "model_name": embeddings.EMBEDDING_MODEL,
            "model_kwargs": {"device": "cpu"},
            "encode_kwargs": {"normalize_embeddings": True},
        }
    ]

    embeddings.get_embedding_model.cache_clear()


def test_get_embedding_model_falls_back_to_hash_embeddings(monkeypatch):
    monkeypatch.setitem(sys.modules, "langchain_huggingface", None)
    embeddings.get_embedding_model.cache_clear()

    model = embeddings.get_embedding_model()
    vectors = model.embed_documents(["PMSM sliding mode observer", "PMSM sliding mode observer"])

    assert isinstance(model, embeddings.HashEmbeddings)
    assert len(vectors) == 2
    assert len(vectors[0]) == 384
    assert vectors[0] == vectors[1]

    embeddings.get_embedding_model.cache_clear()
