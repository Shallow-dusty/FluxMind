import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import api


def test_verify_api_token_allows_when_unconfigured(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")

    api.verify_api_token(None, None)


def test_verify_api_token_accepts_x_api_key(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "secret")

    api.verify_api_token(None, "secret")


def test_verify_api_token_accepts_bearer(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "secret")

    api.verify_api_token("Bearer secret", None)


def test_verify_api_token_rejects_invalid_token(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "secret")

    with pytest.raises(HTTPException) as exc:
        api.verify_api_token("Bearer wrong", None)

    assert exc.value.status_code == 401


def test_query_response_includes_request_id(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")
    monkeypatch.setattr(api, "query", lambda question: f"answer: {question}")

    client = TestClient(api.app)
    response = client.post(
        "/query",
        json={"question": "Explain SMC"},
        headers={"X-Request-ID": "req-test"},
    )

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "req-test"
    assert response.json() == {
        "answer": "answer: Explain SMC",
        "request_id": "req-test",
    }


def test_query_normalizes_provider_errors(monkeypatch):
    monkeypatch.setattr(api, "API_TOKEN", "")

    def fail(_question):
        raise TimeoutError("provider timed out")

    monkeypatch.setattr(api, "query", fail)

    client = TestClient(api.app)
    response = client.post(
        "/query",
        json={"question": "Explain SMC"},
        headers={"X-Request-ID": "req-timeout"},
    )

    assert response.status_code == 504
    assert response.json()["detail"] == {
        "code": "provider_timeout",
        "message": "The model provider timed out. Please retry the request.",
        "request_id": "req-timeout",
    }
