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
    monkeypatch.setattr(chain, "hybrid_retrieve", lambda question, *, k: [])
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

    monkeypatch.setattr(chain, "hybrid_retrieve", lambda question, *, k: [])
    monkeypatch.setattr(chain, "OpenAI", lambda **_kwargs: PlainClient())

    assert "".join(chain.query_stream("Explain SMC")) == "plain answer"


def test_query_stream_wraps_provider_errors(monkeypatch):
    class FailingClient:
        class chat:
            class completions:
                @staticmethod
                def create(**_kwargs):
                    raise TimeoutError("timed out")

    monkeypatch.setattr(chain, "hybrid_retrieve", lambda question, *, k: [])
    monkeypatch.setattr(chain, "OpenAI", lambda **_kwargs: FailingClient())

    try:
        list(chain.query_stream("Explain SMC"))
    except chain.ProviderError as exc:
        assert exc.user_error.code == "provider_timeout"
    else:
        raise AssertionError("expected ProviderError")


def test_query_stream_blocks_before_provider_when_guard_denies(monkeypatch):
    monkeypatch.setattr(chain, "hybrid_retrieve", lambda question, *, k: [])
    monkeypatch.setattr(
        chain,
        "provider_quota_guard_decision",
        lambda **_kwargs: {
            "allowed": False,
            "reason": "provider_completion_token_limit_exceeded",
            "status_code": 429,
        },
    )

    def fail_openai(**_kwargs):
        raise AssertionError("stream client should not be constructed after guard denial")

    monkeypatch.setattr(chain, "OpenAI", fail_openai)

    try:
        list(chain.query_stream("Explain SMC"))
    except chain.ProviderQuotaGuardError as exc:
        assert exc.user_error.code == "provider_completion_token_limit_exceeded"
        assert exc.user_error.status_code == 429
    else:
        raise AssertionError("expected ProviderError")
