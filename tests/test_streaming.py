from types import SimpleNamespace

import src.chain as chain


class _FakeStreamClient:
    class chat:
        class completions:
            @staticmethod
            def create(**_kwargs):
                deltas = [
                    {"reasoning_content": "think\nmore"},
                    {"content": "final answer"},
                ]
                for delta in deltas:
                    yield SimpleNamespace(
                        choices=[SimpleNamespace(delta=SimpleNamespace(**delta))]
                    )


def test_query_stream_formats_reasoning_before_answer(monkeypatch):
    monkeypatch.setattr(chain, "get_vector_store", lambda: None)
    monkeypatch.setattr(chain, "OpenAI", lambda **_kwargs: _FakeStreamClient())

    assert "".join(chain.query_stream("Explain SMC")) == (
        "> 💭 think\n> more\n\n---\n\nfinal answer"
    )


def test_query_stream_plain_content_without_reasoning(monkeypatch):
    class PlainClient:
        class chat:
            class completions:
                @staticmethod
                def create(**_kwargs):
                    yield SimpleNamespace(
                        choices=[
                            SimpleNamespace(delta=SimpleNamespace(content="plain answer"))
                        ]
                    )

    monkeypatch.setattr(chain, "get_vector_store", lambda: None)
    monkeypatch.setattr(chain, "OpenAI", lambda **_kwargs: PlainClient())

    assert "".join(chain.query_stream("Explain SMC")) == "plain answer"


def test_query_stream_wraps_provider_errors(monkeypatch):
    class FailingClient:
        class chat:
            class completions:
                @staticmethod
                def create(**_kwargs):
                    raise TimeoutError("timed out")

    monkeypatch.setattr(chain, "get_vector_store", lambda: None)
    monkeypatch.setattr(chain, "OpenAI", lambda **_kwargs: FailingClient())

    try:
        list(chain.query_stream("Explain SMC"))
    except chain.ProviderError as exc:
        assert exc.user_error.code == "provider_error"
    else:
        raise AssertionError("expected ProviderError")
