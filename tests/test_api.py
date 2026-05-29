import pytest
from fastapi import HTTPException

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
