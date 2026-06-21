import pytest
from fastapi import HTTPException

import api


def test_docker_execution_route_rejects_non_docker_backend(monkeypatch):
    monkeypatch.setattr(api, "CODE_EXECUTION_BACKEND", "local")

    with pytest.raises(HTTPException) as exc_info:
        api._require_docker_execution_backend()

    assert exc_info.value.status_code == 409
    assert "CODE_EXECUTION_BACKEND=docker" in str(exc_info.value.detail)
    assert "local" in str(exc_info.value.detail)


def test_docker_execution_route_accepts_docker_backend(monkeypatch):
    monkeypatch.setattr(api, "CODE_EXECUTION_BACKEND", "docker")

    assert api._require_docker_execution_backend() is None
