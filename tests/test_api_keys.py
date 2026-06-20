import json
import sqlite3

from src import config
from src.api_keys import (
    API_KEY_REGISTRY_SCHEMA_VERSION,
    LocalApiKeyRegistry,
    api_key_registry_backend_status,
    api_key_token_hash,
)


def test_local_api_key_registry_lifecycle_without_plaintext(tmp_path):
    registry = LocalApiKeyRegistry(tmp_path / "api_keys.sqlite3")

    created = registry.create_key(
        owner_id="lab-owner",
        owner_label="Lab Owner",
        description="pilot /private/hunter2 token=sk-secret-value",
    )
    token = created["token"]
    key_id = created["key"]["key_id"]

    assert token.startswith("fmk_")
    created_key_payload = json.dumps(created["key"], sort_keys=True)
    assert created["key"]["owner_id_present"] is True
    assert created["key"]["owner_id_fingerprint"]
    assert created["key"]["owner_label_present"] is True
    assert created["key"]["description_present"] is True
    assert created["key"]["owner_exported"] is False
    assert created["key"]["description_exported"] is False
    assert "lab-owner" not in created_key_payload
    assert "Lab Owner" not in created_key_payload
    assert "/private/hunter2" not in created_key_payload
    assert "sk-secret-value" not in created_key_payload
    assert "token_hash" not in created["key"]
    assert token not in json.dumps(registry.status(), sort_keys=True)

    verified = registry.verify_token(token, update_usage=True)
    assert verified is not None
    assert verified.key_id == key_id
    assert verified.owner_id == "lab-owner"
    assert verified.use_count == 1
    assert verified.last_used_at is not None

    listed = registry.list_keys()
    payload = json.dumps([record.to_public_dict() for record in listed], sort_keys=True)
    assert len(listed) == 1
    assert token not in payload
    assert api_key_token_hash(token) not in payload
    assert "lab-owner" not in payload
    assert "Lab Owner" not in payload
    assert "/private/hunter2" not in payload
    assert "sk-secret-value" not in payload

    revoked = registry.revoke_key(key_id)
    assert revoked is not None
    assert revoked.active is False
    assert registry.verify_token(token) is None
    assert registry.list_keys() == []
    assert len(registry.list_keys(include_revoked=True)) == 1


def test_api_key_registry_status_is_no_secret_and_handles_backends(tmp_path):
    registry = LocalApiKeyRegistry(tmp_path / "api_keys.sqlite3")
    registry.create_key(owner_id="lab-owner")

    status = api_key_registry_backend_status(backend="sqlite", db_path=tmp_path / "api_keys.sqlite3")

    assert status["schema_version"] == API_KEY_REGISTRY_SCHEMA_VERSION
    assert status["configured"] is True
    assert status["supported"] is True
    assert status["available"] is True
    assert status["active_key_count"] == 1
    assert status["secrets_exported"] is False
    assert str(tmp_path) not in json.dumps(status, sort_keys=True)

    disabled = api_key_registry_backend_status(backend="none")
    assert disabled["configured"] is False
    assert disabled["available"] is False

    unsupported = api_key_registry_backend_status(backend="https://secret.example.test")
    assert unsupported["backend"] == "custom"
    assert unsupported["configured"] is True
    assert unsupported["supported"] is False
    assert "secret.example" not in json.dumps(unsupported, sort_keys=True)


def test_api_key_registry_status_reports_invalid_sqlite(tmp_path):
    db_path = tmp_path / "api_keys.sqlite3"
    db_path.write_text("not sqlite", encoding="utf-8")

    status = api_key_registry_backend_status(backend="sqlite", db_path=db_path)

    assert status["configured"] is True
    assert status["supported"] is True
    assert status["available"] is False
    assert status["reason"] == "api_key_registry_unavailable"


def test_api_key_registry_schema_contains_only_hashes(tmp_path):
    registry = LocalApiKeyRegistry(tmp_path / "api_keys.sqlite3")
    token = registry.create_key()["token"]

    with sqlite3.connect(tmp_path / "api_keys.sqlite3") as conn:
        rows = conn.execute("SELECT token_hash FROM api_keys").fetchall()

    assert rows == [(api_key_token_hash(token),)]
    assert token not in str(rows)


def test_api_key_registry_relative_env_path_is_project_anchored(monkeypatch, tmp_path):
    monkeypatch.setenv("FLUXMIND_TEST_RELATIVE_REGISTRY", "metadata/api_keys.sqlite3")

    path = config._project_path_from_env(  # noqa: SLF001
        "FLUXMIND_TEST_RELATIVE_REGISTRY",
        tmp_path / "default.sqlite3",
    )

    assert path == config.PROJECT_ROOT / "metadata" / "api_keys.sqlite3"
