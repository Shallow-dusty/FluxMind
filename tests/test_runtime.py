from src.runtime import ProviderError, new_request_id, normalize_exception


def test_request_ids_are_short_hex_strings():
    request_id = new_request_id()

    assert len(request_id) == 12
    assert int(request_id, 16) >= 0


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
