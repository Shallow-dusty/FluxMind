import json

from src.provider_runtime_rehearsal import (
    PROVIDER_RUNTIME_REHEARSAL_STATE_DIR,
    collect_provider_runtime_rehearsal,
    format_provider_runtime_rehearsal_markdown,
)


def test_provider_runtime_rehearsal_proves_local_provider_contracts(tmp_path):
    status = collect_provider_runtime_rehearsal(
        root=tmp_path / "provider-rehearsal",
        generated_at="2026-06-19T00:00:00+08:00",
    )

    payload = json.dumps(status, ensure_ascii=False, sort_keys=True)
    assert status["mode"] == "provider_runtime_rehearsal"
    assert status["ok"] is True
    assert status["local_only"] is True
    assert status["external_activation_ready"] is False
    assert status["content_exported"] is False
    assert status["secrets_exported"] is False
    assert status["paths_exported"] is False
    assert status["connectivity_checked"] is False
    assert status["image_provider"]["ok"] is True
    assert status["image_provider"]["mime_type"] == "image/svg+xml"
    assert status["image_provider"]["artifact_byte_count"] > 0
    assert status["python_execution"]["ok"] is True
    assert status["python_execution"]["success"] is True
    assert status["python_execution"]["artifact_count"] >= 1
    assert status["python_execution"]["policy_enforced"] is True
    assert status["python_execution"]["policy_violation"] is False
    assert status["octave_execution"]["ok"] is True
    assert status["execution_abuse_policy"]["ok"] is True
    assert status["execution_abuse_policy"]["python"]["policy_violation"] is True
    assert status["execution_abuse_policy"]["python"]["matched_code_count"] == 2
    assert status["execution_abuse_policy"]["octave"]["policy_violation"] is True
    assert status["execution_abuse_policy"]["octave"]["matched_code_count"] == 1
    assert status["execution_abuse_policy"]["content_exported"] is False
    assert status["provider_quota_guard"]["ok"] is True
    assert status["provider_quota_guard"]["allowed"] is True
    assert status["provider_quota_guard"]["blocked"] is True
    assert status["provider_quota_guard"]["blocked_reason"] == "provider_prompt_token_limit_exceeded"
    assert status["readiness"]["local_foundation_ready"] is True
    assert status["readiness"]["activation_ready"] is False
    assert "external_providers_disabled" in status["readiness"]["activation_blockers"]
    assert str(tmp_path) not in payload
    assert "hunter2" not in payload
    assert "sk-test" not in payload
    for sensitive in (
        "Provider rehearsal SMC observer diagram",
        "provider-runtime-rehearsal-ok",
        "provider-runtime-rehearsal",
        "summary.txt",
        "main.py",
        "main.m",
        "file://",
        "artifacts/",
        "sliding-mode-observer",
        "subprocess.run",
        "echo blocked",
        "system('echo",
    ):
        assert sensitive not in payload


def test_provider_runtime_rehearsal_markdown_is_no_secret(tmp_path):
    status = collect_provider_runtime_rehearsal(
        root=tmp_path / "provider-rehearsal",
        generated_at="2026-06-19T00:00:00+08:00",
    )

    markdown = format_provider_runtime_rehearsal_markdown(status)

    assert "# FluxMind Provider Runtime Rehearsal" in markdown
    assert "Local foundation ready: true" in markdown
    assert "External activation ready: false" in markdown
    assert "Provider Quota Guard" in markdown
    assert "Execution Abuse Policy" in markdown
    assert "Python policy violation: true" in markdown
    assert "Octave policy violation: true" in markdown
    assert "Blocked reason: provider_prompt_token_limit_exceeded" in markdown
    assert "Secrets exported: false" in markdown
    assert "Paths exported: false" in markdown
    assert str(tmp_path) not in markdown
    assert "hunter2" not in markdown
    assert "sk-test" not in markdown
    for sensitive in (
        "Provider rehearsal SMC observer diagram",
        "provider-runtime-rehearsal-ok",
        "provider-runtime-rehearsal",
        "summary.txt",
        "main.py",
        "main.m",
        "file://",
        "artifacts/",
        "sliding-mode-observer",
        "subprocess.run",
        "echo blocked",
        "system('echo",
    ):
        assert sensitive not in markdown


def test_provider_runtime_rehearsal_reuses_root_without_state_leakage(tmp_path):
    root = tmp_path / "provider-rehearsal"

    first = collect_provider_runtime_rehearsal(
        root=root,
        generated_at="2026-06-19T00:00:00+08:00",
    )
    second = collect_provider_runtime_rehearsal(
        root=root,
        generated_at="2026-06-19T00:00:01+08:00",
    )

    assert first["ok"] is True
    assert second["ok"] is True
    assert second["image_provider"]["artifact_byte_count"] > 0
    assert second["python_execution"]["artifact_count"] >= 1


def test_provider_runtime_rehearsal_preserves_root_artifacts(tmp_path):
    root = tmp_path / "provider-rehearsal"
    existing_artifact_dir = root / "artifacts"
    existing_artifact_dir.mkdir(parents=True)
    existing_file = existing_artifact_dir / "existing.txt"
    existing_file.write_text("existing-artifact", encoding="utf-8")
    stale_hidden_artifact = (
        root
        / PROVIDER_RUNTIME_REHEARSAL_STATE_DIR
        / "artifacts"
        / "stale"
        / "old.txt"
    )
    stale_hidden_artifact.parent.mkdir(parents=True)
    stale_hidden_artifact.write_text("stale-artifact", encoding="utf-8")

    status = collect_provider_runtime_rehearsal(
        root=root,
        generated_at="2026-06-19T00:00:00+08:00",
    )

    assert status["ok"] is True
    assert existing_file.read_text(encoding="utf-8") == "existing-artifact"
    state_artifact_root = root / PROVIDER_RUNTIME_REHEARSAL_STATE_DIR / "artifacts"
    assert state_artifact_root.is_dir()
    assert any(path.is_file() for path in state_artifact_root.rglob("*"))
    assert not stale_hidden_artifact.exists()
    assert sorted(path.name for path in existing_artifact_dir.iterdir()) == ["existing.txt"]
