import json

from src.runtime import (
    ProviderError,
    RuntimeEvent,
    append_runtime_event,
    estimate_text_tokens,
    list_runtime_events,
    new_request_id,
    normalize_exception,
    runtime_event_metadata_key_is_sensitive,
    sanitize_runtime_event_request_id,
    runtime_event_to_safe_dict,
    sanitize_runtime_event_message,
    runtime_ownership_metadata,
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


def test_runtime_event_search_uses_safe_projection(tmp_path):
    path = tmp_path / "runtime_events.jsonl"
    event = append_runtime_event(
        kind="query_usage",
        code="estimated_usage",
        message=(
            "safe text tokenValue=secret-message-token "
            "sourcePath=/private/source.pdf https://internal.example/request"
        ),
        request_id="req-search",
        metadata={
            "endpoint": "/query",
            "prompt": "raw prompt text",
            "answer": "raw answer text",
            "access_token": "secret-token",
            "source_path": "/private/source.pdf",
            "workspace_count": 2,
        },
        path=path,
    )

    for sensitive_query in (
        "secret-token",
        "secret-message-token",
        "raw prompt text",
        "raw answer text",
        "/private/source.pdf",
        "internal.example",
    ):
        assert list_runtime_events(q=sensitive_query, path=path) == []
    assert [item.event_id for item in list_runtime_events(q="/query", path=path)] == [event.event_id]
    assert [item.event_id for item in list_runtime_events(q="workspace_count", path=path)] == [event.event_id]
    assert [item.event_id for item in list_runtime_events(q="req-search", path=path)] == [event.event_id]


def test_append_runtime_event_sanitizes_raw_jsonl(tmp_path):
    path = tmp_path / "runtime_events.jsonl"

    event = append_runtime_event(
        kind="query_usage",
        code="estimated_usage",
        message="tokenValue=secret-message-token ownerId=owner-secret",
        request_id="req-raw",
        metadata={
            "endpoint": "/query",
            "accessToken": "secret-access-token",
            "rawPrompt": "raw prompt text",
            "finalAnswer": "raw answer text",
            "sourcePath": "/private/source.pdf",
            "ownerUserId": "owner-user-secret",
            "workspaceId": "workspace-secret",
            "workspace_count": 2,
            "nested": {
                "authorizationHeader": "Bearer secret-token",
                "provider_total_tokens": 9,
            },
        },
        path=path,
    )
    raw = path.read_text(encoding="utf-8")
    payload = json.loads(raw)

    assert event.message == "Runtime event message redacted for no-secret projection."
    assert event.message_redacted is True
    assert event.metadata_redacted_fields == 7
    assert event.metadata == {
        "endpoint": "/query",
        "workspace_count": 2,
        "nested": {"provider_total_tokens": 9},
    }
    assert payload["message"] == "Runtime event message redacted for no-secret projection."
    assert payload["message_redacted"] is True
    assert payload["metadata_redacted_fields"] == 7
    assert payload["metadata"] == event.metadata
    for sensitive in (
        "secret-message-token",
        "owner-secret",
        "secret-access-token",
        "raw prompt text",
        "raw answer text",
        "/private/source.pdf",
        "owner-user-secret",
        "workspace-secret",
        "secret-token",
        "accessToken",
        "rawPrompt",
        "finalAnswer",
        "sourcePath",
        "ownerUserId",
        "workspaceId",
        "authorizationHeader",
    ):
        assert sensitive not in raw


def test_append_runtime_event_redacts_unsafe_request_id_from_raw_jsonl(tmp_path):
    path = tmp_path / "runtime_events.jsonl"

    event = append_runtime_event(
        kind="query_usage",
        code="estimated_usage",
        message="safe",
        request_id="Bearer secret-request-token",
        metadata={"endpoint": "/query"},
        path=path,
    )
    raw = path.read_text(encoding="utf-8")
    payload = json.loads(raw)

    assert event.request_id is None
    assert payload["request_id"] is None
    assert "secret-request-token" not in raw


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


def test_runtime_event_safe_projection_redacts_sensitive_key_variants():
    event = RuntimeEvent(
        event_id="evt-safe",
        kind="query_usage",
        code="estimated_usage",
        message="safe",
        created_at="2026-06-19T00:00:00+00:00",
        request_id="req-secret",
        metadata={
            "access_token": "secret-token",
            "accessToken": "secret-access-token",
            "tokenValue": "secret-token-value",
            "apiKey": "secret-api-key",
            "credentialValue": "secret-credential-value",
            "secret_value": "secret-value",
            "privateKey": "secret-private-key",
            "requestId": "req-secret",
            "auth_key_id": "key-secret",
            "auth_owner_id": "auth-owner-secret",
            "auth_owner_label": "Auth Owner Secret",
            "product_workspace_id": "workspace-secret",
            "workspaceId": "workspace-camel-secret",
            "ownerUserId": "owner-user-secret",
            "member_user_id": "member-user-secret",
            "userEmail": "user-secret@example.invalid",
            "file_path": "/private/source.pdf",
            "filePath": "/private/file-path.pdf",
            "sourcePath": "/private/source-path.pdf",
            "source_url": "https://internal.example/source.pdf",
            "documentURL": "https://internal.example/document.pdf",
            "rawPrompt": "raw question text",
            "finalAnswer": "raw answer text",
            "answer_mode": "explanation",
            "token_status": "valid",
            "credential_type": "x_api_key",
            "api_key_registry_configured": True,
            "provider_prompt_tokens": 4,
            "estimated_total_tokens": 9,
            "safe_count": 2,
            "workspace_count": 3,
            "user_count": 4,
            "member_count": 5,
            "nested": {
                "authorizationHeader": "Bearer secret",
                "totalTokens": 5,
            },
        },
    )

    safe = runtime_event_to_safe_dict(event, include_request_id=False)

    assert safe["request_id_present"] is True
    assert safe["metadata"] == {
        "answer_mode": "explanation",
        "token_status": "valid",
        "credential_type": "x_api_key",
        "api_key_registry_configured": True,
        "provider_prompt_tokens": 4,
        "estimated_total_tokens": 9,
        "safe_count": 2,
        "workspace_count": 3,
        "user_count": 4,
        "member_count": 5,
        "nested": {"totalTokens": 5},
    }
    assert safe["metadata_redacted_fields"] == 24
    payload = json.dumps(safe, ensure_ascii=False, sort_keys=True)
    for sensitive in (
        "secret-token",
        "secret-access-token",
        "secret-token-value",
        "secret-api-key",
        "secret-credential-value",
        "secret-value",
        "secret-private-key",
        "req-secret",
        "key-secret",
        "auth-owner-secret",
        "Auth Owner Secret",
        "workspace-secret",
        "workspace-camel-secret",
        "owner-user-secret",
        "member-user-secret",
        "user-secret@example.invalid",
        "/private/source.pdf",
        "/private/file-path.pdf",
        "/private/source-path.pdf",
        "internal.example",
        "raw question text",
        "raw answer text",
        "authorizationHeader",
    ):
        assert sensitive not in payload


def test_runtime_event_safe_projection_redacts_legacy_unsafe_request_id():
    event = RuntimeEvent(
        event_id="evt-request-id",
        kind="query_usage",
        code="estimated_usage",
        message="safe",
        created_at="2026-06-19T00:00:00+00:00",
        request_id="Bearer secret-request-token",
        metadata={"endpoint": "/query"},
    )

    safe_with_request_id = runtime_event_to_safe_dict(event, include_request_id=True)
    safe_without_request_id = runtime_event_to_safe_dict(event, include_request_id=False)

    assert "request_id" not in safe_with_request_id
    assert safe_with_request_id["request_id_present"] is True
    assert safe_with_request_id["request_id_redacted"] is True
    assert safe_without_request_id["request_id_present"] is True
    assert safe_without_request_id["request_id_redacted"] is True
    assert "secret-request-token" not in json.dumps(
        safe_with_request_id,
        ensure_ascii=False,
        sort_keys=True,
    )


def test_runtime_event_request_id_sanitizer_preserves_safe_correlation_ids():
    assert sanitize_runtime_event_request_id("req-safe_1:ok")[0] == "req-safe_1:ok"
    assert sanitize_runtime_event_request_id("Bearer secret-token") == (None, True, True)
    assert sanitize_runtime_event_request_id("bad request id with spaces") == (
        None,
        True,
        True,
    )
    assert sanitize_runtime_event_request_id("") == (None, False, False)


def test_runtime_ownership_metadata_does_not_export_owner_identifiers():
    metadata = runtime_ownership_metadata(
        {
            "owner_id": "secret-owner",
            "owner_label": "Secret Owner",
            "ownership_source": "request",
        }
    )

    assert metadata == {
        "owner_id_present": True,
        "owner_label_present": True,
        "ownership_source": "request",
    }
    assert "secret-owner" not in json.dumps(metadata, sort_keys=True)
    assert "Secret Owner" not in json.dumps(metadata, sort_keys=True)


def test_runtime_event_safe_projection_redacts_sensitive_message_values():
    event = RuntimeEvent(
        event_id="evt-message",
        kind="provider_failure",
        code="provider_error",
        message=(
            "provider failed tokenValue=secret-token "
            "sourcePath=/private/source.pdf https://internal.example/request"
        ),
        created_at="2026-06-19T00:00:00+00:00",
        request_id="req-message",
        metadata={"endpoint": "/query"},
    )

    safe = runtime_event_to_safe_dict(event, include_request_id=False)

    assert safe["message"] == "Runtime event message redacted for no-secret projection."
    assert safe["message_redacted"] is True
    payload = json.dumps(safe, ensure_ascii=False, sort_keys=True)
    assert "secret-token" not in payload
    assert "/private/source.pdf" not in payload
    assert "internal.example" not in payload


def test_runtime_event_message_sanitizer_redacts_bare_secret_like_tokens(tmp_path):
    path = tmp_path / "runtime_events.jsonl"
    event = append_runtime_event(
        kind="provider_failure",
        code="provider_error",
        message="provider rejected credential sk-testSecretToken123",
        request_id="req-bare-secret",
        metadata={"endpoint": "/query"},
        path=path,
    )
    raw = path.read_text(encoding="utf-8")

    assert event.message == "Runtime event message redacted for no-secret projection."
    assert event.message_redacted is True
    assert "sk-testSecretToken123" not in raw


def test_runtime_event_message_sanitizer_preserves_plain_operational_text():
    message, redacted = sanitize_runtime_event_message(
        "Recent provider failure rate is above the configured threshold."
    )

    assert message == "Recent provider failure rate is above the configured threshold."
    assert redacted is False


def test_runtime_event_sensitive_key_detection_preserves_safe_metric_keys():
    assert runtime_event_metadata_key_is_sensitive("access_token")
    assert runtime_event_metadata_key_is_sensitive("accessToken")
    assert runtime_event_metadata_key_is_sensitive("tokenValue")
    assert runtime_event_metadata_key_is_sensitive("x_api_key")
    assert runtime_event_metadata_key_is_sensitive("credentialValue")
    assert runtime_event_metadata_key_is_sensitive("apiKey")
    assert runtime_event_metadata_key_is_sensitive("rawPrompt")
    assert runtime_event_metadata_key_is_sensitive("finalAnswer")
    assert runtime_event_metadata_key_is_sensitive("file_path")
    assert runtime_event_metadata_key_is_sensitive("filePath")
    assert runtime_event_metadata_key_is_sensitive("sourcePath")
    assert runtime_event_metadata_key_is_sensitive("sourcePaths")
    assert runtime_event_metadata_key_is_sensitive("documentURL")
    assert runtime_event_metadata_key_is_sensitive("requestId")
    assert runtime_event_metadata_key_is_sensitive("auth_key_id")
    assert runtime_event_metadata_key_is_sensitive("authOwnerLabel")
    assert runtime_event_metadata_key_is_sensitive("product_workspace_id")
    assert runtime_event_metadata_key_is_sensitive("workspaceId")
    assert runtime_event_metadata_key_is_sensitive("ownerUserId")
    assert runtime_event_metadata_key_is_sensitive("member_user_id")
    assert runtime_event_metadata_key_is_sensitive("userEmail")
    assert not runtime_event_metadata_key_is_sensitive("product_workspace_present")
    assert not runtime_event_metadata_key_is_sensitive("answer_mode")
    assert not runtime_event_metadata_key_is_sensitive("token_status")
    assert not runtime_event_metadata_key_is_sensitive("tokenStatus")
    assert not runtime_event_metadata_key_is_sensitive("credential_type")
    assert not runtime_event_metadata_key_is_sensitive("api_key_registry_configured")
    assert not runtime_event_metadata_key_is_sensitive("provider_prompt_tokens")
    assert not runtime_event_metadata_key_is_sensitive("estimated_total_tokens")
    assert not runtime_event_metadata_key_is_sensitive("workspace_count")
    assert not runtime_event_metadata_key_is_sensitive("user_count")
    assert not runtime_event_metadata_key_is_sensitive("member_count")
    assert not runtime_event_metadata_key_is_sensitive("share_link_valid")
