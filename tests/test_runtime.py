import json

from src.runtime import (
    ProviderError,
    append_runtime_event,
    estimate_text_tokens,
    list_runtime_events,
    new_request_id,
    normalize_exception,
)


def test_request_ids_are_short_hex_strings():
    request_id = new_request_id()

    assert len(request_id) == 12
    assert int(request_id, 16) >= 0


def test_estimate_text_tokens_is_no_secret_rough_count():
    assert estimate_text_tokens("") == 0
    assert estimate_text_tokens("abcd") == 1
    assert estimate_text_tokens("abcdefgh") == 2


def test_normalize_provider_error_preserves_public_shape():
    error = normalize_exception(ProviderError("upstream failed"))

    assert error.code == "provider_error"
    assert error.status_code == 502
    assert error.message == "upstream failed"


def test_normalize_timeout_error():
    error = normalize_exception(TimeoutError("timed out"))

    assert error.code == "provider_timeout"
    assert error.status_code == 504


def test_normalize_provider_fixture_errors():
    empty = normalize_exception(RuntimeError("upstream_empty_output"))
    malformed = normalize_exception(RuntimeError("malformed streaming chunk"))

    assert empty.code == "provider_empty_output"
    assert malformed.code == "provider_malformed_response"


def test_runtime_events_are_listed_newest_first(tmp_path):
    path = tmp_path / "runtime_events.jsonl"

    first = append_runtime_event(
        kind="provider_failure",
        code="provider_timeout",
        message="timeout",
        request_id="req-1",
        path=path,
    )
    second = append_runtime_event(
        kind="provider_failure",
        code="provider_rate_limited",
        message="rate limited",
        request_id="req-2",
        path=path,
    )

    events = list_runtime_events(kind="provider_failure", limit=2, path=path)

    assert [event.event_id for event in events] == [second.event_id, first.event_id]
    assert events[0].request_id == "req-2"
    assert [event.event_id for event in list_runtime_events(code="provider_timeout", path=path)] == [first.event_id]
    assert [event.event_id for event in list_runtime_events(q="rate limited", path=path)] == [second.event_id]
    assert list_runtime_events(kind="query_usage", path=path) == []


def test_runtime_events_skip_malformed_lines(tmp_path):
    path = tmp_path / "runtime_events.jsonl"
    path.write_text(
        "\n".join(
            [
                "not-json",
                json.dumps(["not", "an", "event"]),
                json.dumps({"kind": "provider_failure"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    event = append_runtime_event(
        kind="provider_failure",
        code="provider_timeout",
        message="timeout",
        request_id="req-ok",
        path=path,
    )

    events = list_runtime_events(kind="provider_failure", path=path)

    assert [item.event_id for item in events] == [event.event_id]
