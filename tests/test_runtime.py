import json

from src.runtime import (
    ProviderError,
    RuntimeEvent,
    append_runtime_event,
    estimate_text_tokens,
    list_runtime_events,
    new_request_id,
    normalize_exception,
    runtime_event_to_dict,
    runtime_ownership_metadata,
    normalize_runtime_event_request_id,
)


def test_request_ids_are_short_hex_strings():
    request_id = new_request_id()
    assert len(request_id) == 12
    assert int(request_id, 16) >= 0


def test_request_id_validation():
    assert normalize_runtime_event_request_id("req-safe_1:ok") == "req-safe_1:ok"
    assert normalize_runtime_event_request_id("bad request id") is None
    assert normalize_runtime_event_request_id("") is None


def test_estimate_text_tokens():
    assert estimate_text_tokens("") == 0
    assert estimate_text_tokens("abcd") == 1
    assert estimate_text_tokens("abcdefgh") == 2


def test_normalize_known_errors():
    provider = normalize_exception(ProviderError("upstream failed"))
    timeout = normalize_exception(TimeoutError("timed out"))
    empty = normalize_exception(RuntimeError("upstream_empty_output"))
    malformed = normalize_exception(RuntimeError("malformed streaming chunk"))
    internal = normalize_exception(RuntimeError("FAISS index checksum mismatch"))

    assert (provider.code, provider.status_code, provider.message) == (
        "provider_error",
        502,
        "upstream failed",
    )
    assert (timeout.code, timeout.status_code) == ("provider_timeout", 504)
    assert empty.code == "provider_empty_output"
    assert malformed.code == "provider_malformed_response"
    assert internal.message == "FAISS index checksum mismatch"


def test_runtime_events_round_trip_and_filter(tmp_path):
    path = tmp_path / "runtime_events.jsonl"
    first = append_runtime_event(
        kind="provider_failure",
        code="provider_timeout",
        message="timeout",
        request_id="req-1",
        metadata={"endpoint": "/query"},
        path=path,
    )
    second = append_runtime_event(
        kind="query_usage",
        code="estimated_usage",
        message="usage",
        request_id="req-2",
        metadata={"estimated_total_tokens": 10},
        path=path,
    )

    assert [event.event_id for event in list_runtime_events(path=path)] == [
        second.event_id,
        first.event_id,
    ]
    assert [event.event_id for event in list_runtime_events(kind="provider_failure", path=path)] == [
        first.event_id
    ]
    assert [event.event_id for event in list_runtime_events(code="estimated_usage", path=path)] == [
        second.event_id
    ]
    assert [event.event_id for event in list_runtime_events(q="total_tokens", path=path)] == [
        second.event_id
    ]


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
        path=path,
    )

    assert [item.event_id for item in list_runtime_events(path=path)] == [event.event_id]


def test_runtime_event_view_can_omit_request_id():
    event = RuntimeEvent(
        event_id="evt",
        kind="query_usage",
        code="estimated_usage",
        message="usage",
        created_at="2026-07-27T00:00:00+00:00",
        request_id="req-1",
        metadata={"estimated_total_tokens": 10},
    )

    assert runtime_event_to_dict(event)["request_id"] == "req-1"
    assert "request_id" not in runtime_event_to_dict(event, include_request_id=False)


def test_runtime_ownership_metadata_keeps_account_context():
    metadata = runtime_ownership_metadata(
        {
            "owner_id": "owner-1",
            "owner_label": "Student",
            "ownership_source": "request",
        }
    )

    assert metadata == {
        "owner_id": "owner-1",
        "owner_label": "Student",
        "ownership_source": "request",
    }
