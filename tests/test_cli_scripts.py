import io
import json
from pathlib import Path
from types import SimpleNamespace

import scripts.activation_suite as activation_suite_cli
import scripts.api_key_registry as api_key_registry_cli
import scripts.collaboration_readiness as collaboration_readiness_cli
import scripts.evaluate_rag as evaluate_rag_cli
import scripts.openapi_contract as openapi_contract_cli
import scripts.platform_migration_preflight as platform_migration_preflight_cli
import scripts.platform_migration_rehearsal as platform_migration_rehearsal_cli
import scripts.product_activation_rehearsal as product_activation_rehearsal_cli
import scripts.provider_runtime_rehearsal as provider_runtime_rehearsal_cli
import scripts.provider_readiness as provider_readiness_cli
import scripts.quality_readiness as quality_readiness_cli
import scripts.product_readiness as product_readiness_cli
import scripts.product_registry as product_registry_cli
import scripts.run_job_worker as run_job_worker_cli
import scripts.runtime_manifest as runtime_manifest_cli
import scripts.share_link_registry as share_link_registry_cli
from scripts._safe_cli import format_os_error
import scripts.storage_schema as storage_schema_cli
import scripts.update_local_references as update_refs_cli


def assert_sanitized_cli_os_error(captured, expected: str = "error: Permission denied") -> None:
    assert expected in captured.err
    assert captured.out == ""
    assert "/private" not in captured.err
    assert "hunter2" not in captured.err


def test_format_os_error_preserves_safe_messages_without_paths():
    assert format_os_error(OSError("cannot read eval")) == "cannot read eval"


def test_format_os_error_redacts_paths_urls_and_token_values():
    message = format_os_error(
        OSError(
            "cannot read /private/hunter2-eval.json from https://secret.example/path "
            "token=sk-test-secret-token"
        )
    )

    assert "cannot read" in message
    assert "[redacted]" in message
    for sensitive in (
        "/private",
        "hunter2",
        "https://secret.example",
        "sk-test-secret-token",
    ):
        assert sensitive not in message


def test_evaluate_rag_cli_writes_json_report(monkeypatch, tmp_path, capsys):
    ok_result = SimpleNamespace(ok=True, case_id="case-1", message="ok")
    provider_result = SimpleNamespace(
        ok=True,
        fixture_id="timeout",
        expected_code="provider_timeout",
        actual_code="provider_timeout",
    )
    gate_result = SimpleNamespace(ok=True, gate_id="minimum_case_count", message="ok")

    monkeypatch.setattr(evaluate_rag_cli, "load_eval_config", lambda path: {"loaded": str(path)})
    monkeypatch.setattr(
        evaluate_rag_cli,
        "evaluate_config",
        lambda config: ([ok_result], [ok_result], [ok_result], [ok_result], [provider_result], [ok_result]),
    )
    monkeypatch.setattr(evaluate_rag_cli, "evaluate_regression_gates", lambda *args, **kwargs: [gate_result])
    monkeypatch.setattr(evaluate_rag_cli, "build_evaluation_report", lambda *args, **kwargs: {"ok": True})

    report_path = tmp_path / "eval-report.json"
    monkeypatch.setattr(
        "sys.argv",
        ["evaluate_rag.py", "--file", str(tmp_path / "config.json"), "--json-report", str(report_path)],
    )

    assert evaluate_rag_cli.main() == 0
    assert json.loads(report_path.read_text(encoding="utf-8")) == {"ok": True}
    output = capsys.readouterr().out
    assert "ok   eval case case-1: ok" in output
    assert "info wrote evaluation report" in output


def test_evaluate_rag_cli_reports_failures(monkeypatch, tmp_path, capsys):
    failed = SimpleNamespace(ok=False, case_id="case-fail", message="missing source")
    provider_result = SimpleNamespace(
        ok=False,
        fixture_id="rate-limit",
        expected_code="provider_rate_limited",
        actual_code="provider_empty_output",
    )
    gate_result = SimpleNamespace(ok=False, gate_id="minimum_case_count", message="too few")

    monkeypatch.setattr(evaluate_rag_cli, "load_eval_config", lambda path: {})
    monkeypatch.setattr(
        evaluate_rag_cli,
        "evaluate_config",
        lambda config: ([failed], [], [], [], [provider_result], []),
    )
    monkeypatch.setattr(evaluate_rag_cli, "evaluate_regression_gates", lambda *args, **kwargs: [gate_result])
    monkeypatch.setattr("sys.argv", ["evaluate_rag.py", "--file", str(tmp_path / "config.json")])

    assert evaluate_rag_cli.main() == 1
    output = capsys.readouterr().out
    assert "fail eval case case-fail: missing source" in output
    assert "fail provider fixture rate-limit" in output
    assert "Failed checks:" in output
    assert "- regression gate minimum_case_count" in output


def test_evaluate_rag_cli_uses_live_urls_and_env_key(monkeypatch, tmp_path, capsys):
    result = SimpleNamespace(
        ok=True,
        case_id="case-1",
        message="ok",
        request_id_present=True,
        request_id_redacted=True,
    )
    calls = []

    monkeypatch.setattr(evaluate_rag_cli, "load_eval_config", lambda path: {"cases": []})
    monkeypatch.setattr(evaluate_rag_cli, "evaluate_config", lambda config: ([], [], [], [], [], []))
    monkeypatch.setattr(evaluate_rag_cli, "evaluate_regression_gates", lambda *args, **kwargs: [])

    def fake_live_config(config, *, base_url, api_token, timeout_s):
        calls.append(("live", base_url, api_token, timeout_s))
        return [result]

    def fake_retrieval_config(config, *, base_url, api_token, timeout_s):
        calls.append(("retrieval", base_url, api_token, timeout_s))
        return [result]

    monkeypatch.setattr(evaluate_rag_cli, "evaluate_live_config", fake_live_config)
    monkeypatch.setattr(evaluate_rag_cli, "evaluate_live_retrieval_config", fake_retrieval_config)
    monkeypatch.setenv("TOKEN_ENV", "secret-token")
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_rag.py",
            "--file",
            str(tmp_path / "config.json"),
            "--live-url",
            "https://api.example.test",
            "--retrieval-url",
            "https://api.example.test",
            "--api-key-env",
            "TOKEN_ENV",
            "--live-timeout",
            "3.5",
        ],
    )

    assert evaluate_rag_cli.main() == 0
    assert calls == [
        ("live", "https://api.example.test", "secret-token", 3.5),
        ("retrieval", "https://api.example.test", "secret-token", 3.5),
    ]
    output = capsys.readouterr().out
    assert "ok   live answer case-1: ok request_id_present=True request_id_redacted=True" in output
    assert "ok   live retrieval case-1: ok request_id_present=True request_id_redacted=True" in output
    assert "req-1" not in output


def test_runtime_manifest_cli_outputs_manifest(monkeypatch, tmp_path):
    output_path = tmp_path / "manifest.md"
    monkeypatch.setattr(runtime_manifest_cli, "collect_runtime_backup_manifest", lambda: {"ok": True})
    monkeypatch.setattr(runtime_manifest_cli, "format_runtime_backup_manifest_markdown", lambda manifest: "# Manifest")
    monkeypatch.setattr("sys.argv", ["runtime_manifest.py", "--format", "markdown", "--output", str(output_path)])

    assert runtime_manifest_cli.main() == 0
    assert output_path.read_text(encoding="utf-8") == "# Manifest\n"


def test_runtime_manifest_cli_restore_check_uses_stdin(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO('{"groups": []}'))
    monkeypatch.setattr(
        runtime_manifest_cli,
        "collect_runtime_restore_check",
        lambda manifest, project_root: {"ok": False, "root": str(project_root), "groups": manifest["groups"]},
    )
    monkeypatch.setattr("sys.argv", ["runtime_manifest.py", "--restore-check", "-", "--target-root", "/tmp/root"])

    assert runtime_manifest_cli.main() == 1
    output = json.loads(capsys.readouterr().out)
    assert output["ok"] is False
    assert output["root"] == "/tmp/root"


def test_runtime_manifest_cli_reports_read_errors(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["runtime_manifest.py", "--restore-check", "/missing/manifest.json"])

    assert runtime_manifest_cli.main() == 2
    assert "error:" in capsys.readouterr().err


def test_api_key_registry_cli_lifecycle(monkeypatch, tmp_path, capsys):
    db_path = tmp_path / "api_keys.sqlite3"
    monkeypatch.setattr(
        "sys.argv",
        [
            "api_key_registry.py",
            "--db",
            str(db_path),
            "create",
            "--owner-id",
            "lab-cli",
        ],
    )

    assert api_key_registry_cli.main() == 0
    created = json.loads(capsys.readouterr().out)
    token = created["token"]
    key_id = created["key"]["key_id"]
    assert token.startswith("fmk_")

    monkeypatch.setattr("sys.argv", ["api_key_registry.py", "--db", str(db_path), "verify", token])
    assert api_key_registry_cli.main() == 0
    verified_output = capsys.readouterr().out
    assert json.loads(verified_output)["valid"] is True
    assert token not in verified_output

    monkeypatch.setattr("sys.argv", ["api_key_registry.py", "--db", str(db_path), "revoke", key_id])
    assert api_key_registry_cli.main() == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True

    monkeypatch.setattr("sys.argv", ["api_key_registry.py", "--db", str(db_path), "verify", token])
    assert api_key_registry_cli.main() == 1
    assert json.loads(capsys.readouterr().out)["valid"] is False


def test_api_key_registry_cli_status_markdown(monkeypatch, tmp_path):
    db_path = tmp_path / "api_keys.sqlite3"
    output_path = tmp_path / "registry.md"
    monkeypatch.setattr(
        "sys.argv",
        [
            "api_key_registry.py",
            "--db",
            str(db_path),
            "--format",
            "markdown",
            "--output",
            str(output_path),
            "status",
        ],
    )

    assert api_key_registry_cli.main() == 0
    markdown = output_path.read_text(encoding="utf-8")
    assert "# FluxMind API Key Registry" in markdown
    assert "Secrets exported: false" in markdown


def test_api_key_registry_cli_rejects_markdown_create_before_writing(monkeypatch, tmp_path, capsys):
    db_path = tmp_path / "api_keys.sqlite3"
    monkeypatch.setattr(
        "sys.argv",
        [
            "api_key_registry.py",
            "--db",
            str(db_path),
            "create",
            "--format",
            "markdown",
        ],
    )

    assert api_key_registry_cli.main() == 2
    captured = capsys.readouterr()
    assert "requires --format json" in captured.err
    assert "fmk_" not in captured.out
    assert "fmk_" not in captured.err
    assert not db_path.exists()


def test_api_key_registry_cli_accepts_subcommand_output_flags(monkeypatch, tmp_path, capsys):
    db_path = tmp_path / "api_keys.sqlite3"
    monkeypatch.setattr(
        "sys.argv",
        [
            "api_key_registry.py",
            "--db",
            str(db_path),
            "status",
            "--format",
            "markdown",
        ],
    )

    assert api_key_registry_cli.main() == 0
    assert "# FluxMind API Key Registry" in capsys.readouterr().out


def test_api_key_registry_cli_reports_sqlite_errors(monkeypatch, tmp_path, capsys):
    db_path = tmp_path / "api_keys.sqlite3"
    db_path.write_text("not sqlite", encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "api_key_registry.py",
            "--db",
            str(db_path),
            "list",
        ],
    )

    assert api_key_registry_cli.main() == 2
    captured = capsys.readouterr()
    assert "error:" in captured.err
    assert str(db_path) not in captured.err


def test_api_key_registry_cli_output_os_errors_are_sanitized(monkeypatch, tmp_path, capsys):
    output_path = tmp_path / "missing" / "private-hunter2" / "api-keys.md"
    monkeypatch.setattr(
        "sys.argv",
        [
            "api_key_registry.py",
            "--output",
            str(output_path),
            "status",
        ],
    )

    assert api_key_registry_cli.main() == 2
    captured = capsys.readouterr()
    assert "error:" in captured.err
    assert "private-hunter2" not in captured.err
    assert str(output_path) not in captured.err


def test_share_link_registry_cli_lifecycle(monkeypatch, tmp_path, capsys):
    db_path = tmp_path / "share_links.sqlite3"
    monkeypatch.setattr(
        "sys.argv",
        [
            "share_link_registry.py",
            "--db",
            str(db_path),
            "create",
            "--workspace-id",
            "lab-ws",
            "--created-by-user-id",
            "owner-user",
            "--resource-kind",
            "corpus_profile",
            "--resource-ref",
            "private-profile-id",
            "--description",
            "pilot share /private/hunter2 token=sk-secret-value",
            "--max-redemptions",
            "2",
        ],
    )

    assert share_link_registry_cli.main() == 0
    created = json.loads(capsys.readouterr().out)
    token = created["token"]
    link_id = created["share_link"]["link_id"]
    assert token.startswith("fms_")
    assert "workspace_id" not in created["share_link"]
    assert created["share_link"]["workspace_present"] is True
    assert created["share_link"]["workspace_fingerprint"]
    assert "lab-ws" not in json.dumps(created["share_link"], sort_keys=True)
    assert "private-profile-id" not in json.dumps(created["share_link"], sort_keys=True)
    assert "owner-user" not in json.dumps(created["share_link"], sort_keys=True)
    assert "/private/hunter2" not in json.dumps(created["share_link"], sort_keys=True)
    assert "sk-secret-value" not in json.dumps(created["share_link"], sort_keys=True)

    monkeypatch.setattr(
        "sys.argv",
        ["share_link_registry.py", "--db", str(db_path), "resolve", token],
    )
    assert share_link_registry_cli.main() == 0
    resolved = json.loads(capsys.readouterr().out)
    rendered = json.dumps(resolved, sort_keys=True)
    assert resolved["resolution"]["valid"] is True
    assert token not in rendered
    assert "workspace_id" not in rendered
    assert "lab-ws" not in rendered
    assert "private-profile-id" not in rendered
    assert "owner-user" not in rendered
    assert "/private/hunter2" not in rendered
    assert "sk-secret-value" not in rendered

    monkeypatch.setattr(
        "sys.argv",
        [
            "share_link_registry.py",
            "--db",
            str(db_path),
            "--format",
            "markdown",
            "list",
        ],
    )
    assert share_link_registry_cli.main() == 0
    listed = capsys.readouterr().out
    assert "FluxMind Share Link Registry" in listed
    assert link_id in listed
    assert token not in listed
    assert "workspace_id" not in listed
    assert "lab-ws" not in listed
    assert "private-profile-id" not in listed
    assert "owner-user" not in listed
    assert "/private/hunter2" not in listed
    assert "sk-secret-value" not in listed

    monkeypatch.setattr(
        "sys.argv",
        ["share_link_registry.py", "--db", str(db_path), "revoke", link_id],
    )
    assert share_link_registry_cli.main() == 0
    revoked = json.loads(capsys.readouterr().out)
    assert revoked["ok"] is True
    assert revoked["share_link"]["revoked_at"]
    assert token not in json.dumps(revoked, sort_keys=True)
    assert "workspace_id" not in json.dumps(revoked, sort_keys=True)
    assert "lab-ws" not in json.dumps(revoked, sort_keys=True)


def test_share_link_registry_cli_rejects_markdown_create_before_writing(monkeypatch, tmp_path, capsys):
    db_path = tmp_path / "share_links.sqlite3"
    monkeypatch.setattr(
        "sys.argv",
        [
            "share_link_registry.py",
            "--db",
            str(db_path),
            "create",
            "--format",
            "markdown",
            "--workspace-id",
            "lab-ws",
            "--created-by-user-id",
            "owner-user",
            "--resource-ref",
            "private-profile-id",
        ],
    )

    assert share_link_registry_cli.main() == 2
    captured = capsys.readouterr()
    assert "requires --format json" in captured.err
    assert "fms_" not in captured.out
    assert "fms_" not in captured.err
    assert not db_path.exists()


def test_share_link_registry_cli_output_os_errors_are_sanitized(monkeypatch, tmp_path, capsys):
    output_path = tmp_path / "missing" / "private-hunter2" / "share-links.md"
    monkeypatch.setattr(
        "sys.argv",
        [
            "share_link_registry.py",
            "--output",
            str(output_path),
            "status",
        ],
    )

    assert share_link_registry_cli.main() == 2
    captured = capsys.readouterr()
    assert "error:" in captured.err
    assert "private-hunter2" not in captured.err
    assert str(output_path) not in captured.err


def test_share_link_registry_cli_markdown_shapes_are_no_secret():
    created = share_link_registry_cli.render_markdown(
        {
            "token": "fms_secret-token",
            "share_link": {
                "link_id": "share_1",
                "workspace_present": True,
                "workspace_fingerprint": "abc123",
                "resource_kind": "corpus_profile",
                "share_token_exported": False,
            },
        }
    )
    empty_list = share_link_registry_cli.render_markdown({"share_links": []})
    resolution = share_link_registry_cli.render_markdown(
        {
            "resolution": {
                "valid": False,
                "reason": "share_link_revoked",
                "share_token_exported": False,
                "share_link": {
                    "link_id": "share_1",
                    "resource_kind": "corpus_profile",
                    "resource_ref_fingerprint": "abc123",
                },
            }
        }
    )
    revoked = share_link_registry_cli.render_markdown(
        {
            "share_link": {
                "link_id": "share_1",
                "active": False,
                "revoked_at": "2026-06-20T00:00:00+00:00",
                "share_token_exported": False,
            }
        }
    )

    rendered = "\n".join([created, empty_list, resolution, revoked])
    assert "Created Share Link" in created
    assert "Token: shown once in JSON output only" in created
    assert "- none" in empty_list
    assert "Reason: share_link_revoked" in resolution
    assert "Revoked at: 2026-06-20T00:00:00+00:00" in revoked
    assert "fms_secret-token" not in rendered
    assert "lab-ws" not in rendered
    assert "Share token exported: false" in rendered


def test_share_link_registry_cli_revoke_missing_returns_nonzero(monkeypatch, tmp_path, capsys):
    db_path = tmp_path / "share_links.sqlite3"
    monkeypatch.setattr(
        "sys.argv",
        [
            "share_link_registry.py",
            "--db",
            str(db_path),
            "revoke",
            "share_missing",
        ],
    )

    assert share_link_registry_cli.main() == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["reason"] == "share_link_not_found"
    assert payload["share_tokens_exported"] is False


def test_share_link_registry_cli_reports_sqlite_errors(monkeypatch, tmp_path, capsys):
    db_path = tmp_path / "share_links.sqlite3"
    db_path.write_text("not sqlite", encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "share_link_registry.py",
            "--db",
            str(db_path),
            "list",
        ],
    )

    assert share_link_registry_cli.main() == 2
    captured = capsys.readouterr()
    assert "error:" in captured.err
    assert str(db_path) not in captured.err


def test_product_registry_cli_bootstrap_usage_and_markdown(monkeypatch, tmp_path, capsys):
    db_path = tmp_path / "product_registry.sqlite3"
    monkeypatch.setattr(
        "sys.argv",
        [
            "product_registry.py",
            "--db",
            str(db_path),
            "bootstrap-local",
            "--user-id",
            "lab-cli",
            "--workspace-id",
            "lab-workspace",
        ],
    )

    assert product_registry_cli.main() == 0
    created = json.loads(capsys.readouterr().out)
    assert created["workspace"]["workspace_id"] == "lab-workspace"
    assert created["workspace"]["owner_user_id"] == "lab-cli"
    assert created["secrets_exported"] is False

    monkeypatch.setattr(
        "sys.argv",
        [
            "product_registry.py",
            "--db",
            str(db_path),
            "record-usage",
            "--workspace-id",
            "lab-workspace",
            "--user-id",
            "lab-cli",
            "--metric",
            "requests",
            "--amount",
            "2",
            "--format",
            "markdown",
        ],
    )

    assert product_registry_cli.main() == 0
    markdown = capsys.readouterr().out
    assert "# FluxMind Product Registry" in markdown
    assert "Amount: 2" in markdown
    assert "Secrets exported: false" in markdown


def test_product_registry_cli_status_output_file(monkeypatch, tmp_path):
    db_path = tmp_path / "product_registry.sqlite3"
    output_path = tmp_path / "product-registry.md"
    monkeypatch.setattr(
        "sys.argv",
        [
            "product_registry.py",
            "--db",
            str(db_path),
            "--format",
            "markdown",
            "--output",
            str(output_path),
            "status",
        ],
    )

    assert product_registry_cli.main() == 0
    markdown = output_path.read_text(encoding="utf-8")
    assert "# FluxMind Product Registry" in markdown
    assert "Available: true" in markdown
    assert "Secrets exported: false" in markdown


def test_product_registry_cli_output_os_errors_are_sanitized(monkeypatch, tmp_path, capsys):
    output_path = tmp_path / "missing" / "private-hunter2" / "product.md"
    monkeypatch.setattr(
        "sys.argv",
        [
            "product_registry.py",
            "--output",
            str(output_path),
            "status",
        ],
    )

    assert product_registry_cli.main() == 2
    captured = capsys.readouterr()
    assert "error:" in captured.err
    assert "private-hunter2" not in captured.err
    assert str(output_path) not in captured.err


def test_product_registry_cli_reports_sqlite_errors(monkeypatch, tmp_path, capsys):
    db_path = tmp_path / "product_registry.sqlite3"
    db_path.write_text("not sqlite", encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "product_registry.py",
            "--db",
            str(db_path),
            "list-workspaces",
        ],
    )

    assert product_registry_cli.main() == 2
    captured = capsys.readouterr()
    assert "error:" in captured.err
    assert str(db_path) not in captured.err


def test_product_registry_cli_member_and_permission_check(monkeypatch, tmp_path, capsys):
    db_path = tmp_path / "product_registry.sqlite3"
    monkeypatch.setattr(
        "sys.argv",
        [
            "product_registry.py",
            "--db",
            str(db_path),
            "bootstrap-local",
            "--user-id",
            "owner",
            "--workspace-id",
            "lab-workspace",
        ],
    )
    assert product_registry_cli.main() == 0
    capsys.readouterr()

    monkeypatch.setattr(
        "sys.argv",
        [
            "product_registry.py",
            "--db",
            str(db_path),
            "add-member",
            "--workspace-id",
            "lab-workspace",
            "--user-id",
            "viewer",
            "--role",
            "viewer",
        ],
    )
    assert product_registry_cli.main() == 0
    assert json.loads(capsys.readouterr().out)["member"]["role"] == "viewer"

    monkeypatch.setattr(
        "sys.argv",
        [
            "product_registry.py",
            "--db",
            str(db_path),
            "check-permission",
            "--workspace-id",
            "lab-workspace",
            "--user-id",
            "viewer",
            "--action",
            "query",
        ],
    )
    assert product_registry_cli.main() == 0
    assert json.loads(capsys.readouterr().out)["permission"]["allowed"] is True

    monkeypatch.setattr(
        "sys.argv",
        [
            "product_registry.py",
            "--db",
            str(db_path),
            "check-permission",
            "--workspace-id",
            "lab-workspace",
            "--user-id",
            "viewer",
            "--action",
            "job_submit",
            "--format",
            "markdown",
        ],
    )
    assert product_registry_cli.main() == 1
    markdown = capsys.readouterr().out
    assert "Allowed: false" in markdown
    assert "Reason: product_role_forbidden" in markdown
    assert "Secrets exported: false" in markdown


def test_product_activation_rehearsal_cli_is_no_secret(monkeypatch, tmp_path, capsys):
    root = tmp_path / "rehearsal"
    output_path = tmp_path / "product-rehearsal.md"
    monkeypatch.setattr(
        "sys.argv",
        [
            "product_activation_rehearsal.py",
            "--root",
            str(root),
            "--format",
            "markdown",
            "--output",
            str(output_path),
            "--require-activation",
        ],
    )

    assert product_activation_rehearsal_cli.main() == 0
    assert capsys.readouterr().out == ""
    markdown = output_path.read_text(encoding="utf-8")
    assert "# FluxMind Product Activation Rehearsal" in markdown
    assert "OK: true" in markdown
    assert "Activation ready: true" in markdown
    assert "Secrets exported: false" in markdown
    assert "Paths exported: false" in markdown
    assert "fmk_" not in markdown
    assert str(root) not in markdown
    for sensitive in (
        "rehearsal-owner",
        "rehearsal-viewer",
        "rehearsal-workspace",
        "api_keys.sqlite3",
        "product_registry.sqlite3",
    ):
        assert sensitive not in markdown


def test_product_activation_rehearsal_cli_sanitizes_output_os_errors(monkeypatch, tmp_path, capsys):
    output_path = tmp_path / "missing" / "private-hunter2" / "product.md"
    monkeypatch.setattr(
        product_activation_rehearsal_cli,
        "collect_product_activation_rehearsal",
        lambda **kwargs: {"mode": "product_activation_rehearsal", "ok": True},
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "product_activation_rehearsal.py",
            "--output",
            str(output_path),
        ],
    )

    assert product_activation_rehearsal_cli.main() == 2
    captured = capsys.readouterr()
    assert "error: No such file or directory" in captured.err
    assert captured.out == ""
    assert str(tmp_path) not in captured.err
    assert "private-hunter2" not in captured.err


def test_collaboration_readiness_cli_default_allows_safe_foundation(monkeypatch, capsys):
    monkeypatch.setattr(
        collaboration_readiness_cli,
        "collect_collaboration_readiness",
        lambda: {
            "mode": "collaboration_readiness",
            "ok": True,
            "activation_ready": False,
            "secrets_exported": False,
        },
    )
    monkeypatch.setattr("sys.argv", ["collaboration_readiness.py"])

    assert collaboration_readiness_cli.main() == 0
    output = json.loads(capsys.readouterr().out)
    assert output["mode"] == "collaboration_readiness"
    assert output["ok"] is True
    assert output["activation_ready"] is False


def test_collaboration_readiness_cli_can_require_activation(monkeypatch, capsys):
    monkeypatch.setattr(
        collaboration_readiness_cli,
        "collect_collaboration_readiness",
        lambda: {
            "mode": "collaboration_readiness",
            "ok": True,
            "activation_ready": False,
            "blockers": {"activation": ["share_links_disabled"]},
        },
    )
    monkeypatch.setattr(
        "sys.argv",
        ["collaboration_readiness.py", "--require-activation"],
    )

    assert collaboration_readiness_cli.main() == 1
    output = json.loads(capsys.readouterr().out)
    assert output["blockers"]["activation"] == ["share_links_disabled"]


def test_collaboration_readiness_cli_writes_markdown(monkeypatch, tmp_path):
    output_path = tmp_path / "collaboration.md"
    monkeypatch.setattr(
        collaboration_readiness_cli,
        "collect_collaboration_readiness",
        lambda: {
            "mode": "collaboration_readiness",
            "ok": True,
            "activation_ready": False,
            "secrets_exported": False,
        },
    )
    monkeypatch.setattr(
        collaboration_readiness_cli,
        "format_collaboration_readiness_markdown",
        lambda status: "# FluxMind Collaboration Readiness\n\n- Secrets exported: false",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "collaboration_readiness.py",
            "--format",
            "markdown",
            "--output",
            str(output_path),
        ],
    )

    assert collaboration_readiness_cli.main() == 0
    assert output_path.read_text(encoding="utf-8").endswith("Secrets exported: false\n")


def test_collaboration_readiness_cli_reports_sanitized_os_errors(monkeypatch, tmp_path, capsys):
    output_path = tmp_path / "missing" / "private-hunter2" / "collaboration.md"
    monkeypatch.setattr(
        collaboration_readiness_cli,
        "collect_collaboration_readiness",
        lambda: {"mode": "collaboration_readiness", "ok": True},
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "collaboration_readiness.py",
            "--output",
            str(output_path),
        ],
    )

    assert collaboration_readiness_cli.main() == 2
    captured = capsys.readouterr()
    assert "error: No such file or directory" in captured.err
    assert str(tmp_path) not in captured.err
    assert "private-hunter2" not in captured.err


def test_provider_runtime_rehearsal_cli_is_no_secret(monkeypatch, tmp_path, capsys):
    root = tmp_path / "provider-rehearsal"
    output_path = tmp_path / "provider-rehearsal.md"
    monkeypatch.setattr(
        "sys.argv",
        [
            "provider_runtime_rehearsal.py",
            "--root",
            str(root),
            "--format",
            "markdown",
            "--output",
            str(output_path),
            "--require-local-foundation",
        ],
    )

    assert provider_runtime_rehearsal_cli.main() == 0
    assert capsys.readouterr().out == ""
    markdown = output_path.read_text(encoding="utf-8")
    assert "# FluxMind Provider Runtime Rehearsal" in markdown
    assert "OK: true" in markdown
    assert "Local foundation ready: true" in markdown
    assert "External activation ready: false" in markdown
    assert "Provider Quota Guard" in markdown
    assert "Blocked reason: provider_prompt_token_limit_exceeded" in markdown
    assert "Secrets exported: false" in markdown
    assert "Paths exported: false" in markdown
    assert str(root) not in markdown
    assert "hunter2" not in markdown
    for sensitive in (
        "Provider rehearsal SMC observer diagram",
        "provider-runtime-rehearsal-ok",
        "provider-runtime-rehearsal",
        "summary.txt",
        "main.py",
        "main.m",
        "file://",
    ):
        assert sensitive not in markdown


def test_provider_runtime_rehearsal_cli_sanitizes_output_os_errors(monkeypatch, tmp_path, capsys):
    output_path = tmp_path / "missing" / "private-hunter2" / "provider.md"
    monkeypatch.setattr(
        provider_runtime_rehearsal_cli,
        "collect_provider_runtime_rehearsal",
        lambda **kwargs: {"mode": "provider_runtime_rehearsal", "ok": True},
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "provider_runtime_rehearsal.py",
            "--output",
            str(output_path),
        ],
    )

    assert provider_runtime_rehearsal_cli.main() == 2
    captured = capsys.readouterr()
    assert "error: No such file or directory" in captured.err
    assert captured.out == ""
    assert str(tmp_path) not in captured.err
    assert "private-hunter2" not in captured.err


def test_activation_suite_cli_default_allows_local_foundation(monkeypatch, capsys):
    seen = {}

    def fake_collect(**kwargs):
        seen.update(kwargs)
        return {
            "local_foundation_ready": True,
            "small_group_ready": False,
            "community_ready": False,
            "full_activation_ready": False,
        }

    monkeypatch.setattr(
        activation_suite_cli,
        "collect_activation_suite",
        fake_collect,
    )
    monkeypatch.setattr(
        activation_suite_cli,
        "_load_openapi_schema",
        lambda: {"openapi": "3.1.0"},
    )
    monkeypatch.setattr("sys.argv", ["activation_suite.py", "--target-root", "/tmp/root"])

    assert activation_suite_cli.main() == 0
    output = json.loads(capsys.readouterr().out)
    assert output["local_foundation_ready"] is True
    assert output["full_activation_ready"] is False
    assert str(seen["project_root"]) == "/tmp/root"
    assert str(seen["eval_file"]) == "/tmp/root/eval/rag_baseline.json"
    assert seen["openapi_schema"] == {"openapi": "3.1.0"}


def test_activation_suite_cli_can_require_full_activation(monkeypatch, capsys):
    monkeypatch.setattr(
        activation_suite_cli,
        "collect_activation_suite",
        lambda **kwargs: {
            "local_foundation_ready": True,
            "small_group_ready": True,
            "community_ready": False,
            "full_activation_ready": False,
        },
    )
    monkeypatch.setattr(
        activation_suite_cli,
        "_load_openapi_schema",
        lambda: {"openapi": "3.1.0"},
    )
    monkeypatch.setattr(
        "sys.argv",
        ["activation_suite.py", "--require-target", "full_activation"],
    )

    assert activation_suite_cli.main() == 1
    assert json.loads(capsys.readouterr().out)["full_activation_ready"] is False


def test_activation_suite_cli_writes_markdown(monkeypatch, tmp_path):
    output_path = tmp_path / "suite.md"
    monkeypatch.setattr(
        activation_suite_cli,
        "collect_activation_suite",
        lambda **kwargs: {
            "local_foundation_ready": True,
            "small_group_ready": False,
            "community_ready": False,
            "full_activation_ready": False,
        },
    )
    monkeypatch.setattr(
        activation_suite_cli,
        "_load_openapi_schema",
        lambda: {"openapi": "3.1.0"},
    )
    monkeypatch.setattr(
        activation_suite_cli,
        "format_activation_suite_markdown",
        lambda status: "# Suite",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "activation_suite.py",
            "--format",
            "markdown",
            "--output",
            str(output_path),
        ],
    )

    assert activation_suite_cli.main() == 0
    assert output_path.read_text(encoding="utf-8") == "# Suite\n"


def test_activation_suite_cli_reports_os_errors(monkeypatch, capsys):
    def fail_collect(**kwargs):
        raise OSError(13, "Permission denied", "/private/hunter2-suite.json")

    monkeypatch.setattr(activation_suite_cli, "collect_activation_suite", fail_collect)
    monkeypatch.setattr(
        activation_suite_cli,
        "_load_openapi_schema",
        lambda: {"openapi": "3.1.0"},
    )
    monkeypatch.setattr("sys.argv", ["activation_suite.py"])

    assert activation_suite_cli.main() == 2
    captured = capsys.readouterr()
    assert "error: Permission denied" in captured.err
    assert "/private" not in captured.err
    assert "hunter2" not in captured.err


def test_openapi_contract_cli_writes_markdown(monkeypatch, tmp_path):
    output_path = tmp_path / "openapi.md"
    monkeypatch.setattr(
        openapi_contract_cli,
        "collect_openapi_contract",
        lambda schema: {"local_contract_ready": True},
    )
    monkeypatch.setattr(
        openapi_contract_cli,
        "format_openapi_contract_markdown",
        lambda status: "# OpenAPI",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "openapi_contract.py",
            "--format",
            "markdown",
            "--output",
            str(output_path),
        ],
    )

    assert openapi_contract_cli.main() == 0
    assert output_path.read_text(encoding="utf-8") == "# OpenAPI\n"


def test_openapi_contract_cli_can_require_local_contract(monkeypatch, capsys):
    monkeypatch.setattr(
        openapi_contract_cli,
        "collect_openapi_contract",
        lambda schema: {"local_contract_ready": False},
    )
    monkeypatch.setattr(
        "sys.argv",
        ["openapi_contract.py", "--require-local-contract"],
    )

    assert openapi_contract_cli.main() == 1
    assert json.loads(capsys.readouterr().out)["local_contract_ready"] is False


def test_openapi_contract_cli_can_verify_snapshot(monkeypatch, tmp_path, capsys):
    snapshot_path = tmp_path / "openapi-snapshot.json"
    snapshot_path.write_text(
        json.dumps({"mode": "openapi_contract", "operation_fingerprint": "abc"}),
        encoding="utf-8",
    )
    seen = {}
    monkeypatch.setattr(
        openapi_contract_cli,
        "collect_openapi_contract",
        lambda schema: {"local_contract_ready": True, "operation_fingerprint": "abc"},
    )

    def fake_verify(current, snapshot):
        seen["current"] = current
        seen["snapshot"] = snapshot
        return {"mode": "openapi_contract_snapshot_verify", "ok": True, "diff_count": 0}

    monkeypatch.setattr(openapi_contract_cli, "verify_openapi_contract_snapshot", fake_verify)
    monkeypatch.setattr(
        "sys.argv",
        [
            "openapi_contract.py",
            "--verify-snapshot",
            str(snapshot_path),
            "--require-no-drift",
        ],
    )

    assert openapi_contract_cli.main() == 0
    output = json.loads(capsys.readouterr().out)
    assert output["ok"] is True
    assert seen["snapshot"]["operation_fingerprint"] == "abc"
    assert seen["current"]["local_contract_ready"] is True


def test_openapi_contract_cli_returns_nonzero_for_snapshot_drift(monkeypatch, tmp_path, capsys):
    snapshot_path = tmp_path / "openapi-snapshot.json"
    snapshot_path.write_text(
        json.dumps({"mode": "openapi_contract", "operation_fingerprint": "old"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        openapi_contract_cli,
        "collect_openapi_contract",
        lambda schema: {"local_contract_ready": True, "operation_fingerprint": "new"},
    )
    monkeypatch.setattr(
        openapi_contract_cli,
        "verify_openapi_contract_snapshot",
        lambda current, snapshot: {
            "mode": "openapi_contract_snapshot_verify",
            "ok": False,
            "diff_count": 1,
        },
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "openapi_contract.py",
            "--verify-snapshot",
            str(snapshot_path),
            "--require-no-drift",
        ],
    )

    assert openapi_contract_cli.main() == 1
    assert json.loads(capsys.readouterr().out)["diff_count"] == 1


def test_openapi_contract_cli_requires_snapshot_for_no_drift_gate(monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.argv",
        ["openapi_contract.py", "--require-no-drift"],
    )

    assert openapi_contract_cli.main() == 2
    assert "--require-no-drift requires --verify-snapshot" in capsys.readouterr().err


def test_openapi_contract_cli_rejects_non_object_snapshot(monkeypatch, tmp_path, capsys):
    snapshot_path = tmp_path / "openapi-snapshot.json"
    snapshot_path.write_text('["not-an-object"]', encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "openapi_contract.py",
            "--verify-snapshot",
            str(snapshot_path),
        ],
    )

    assert openapi_contract_cli.main() == 2
    assert "snapshot JSON must be an object" in capsys.readouterr().err


def test_openapi_contract_cli_reports_sanitized_os_errors(monkeypatch, capsys):
    def fail_collect(schema):
        raise OSError(13, "Permission denied", "/private/hunter2-openapi.json")

    monkeypatch.setattr(openapi_contract_cli, "collect_openapi_contract", fail_collect)
    monkeypatch.setattr("sys.argv", ["openapi_contract.py"])

    assert openapi_contract_cli.main() == 2
    assert_sanitized_cli_os_error(capsys.readouterr())


def test_storage_schema_cli_outputs_markdown(monkeypatch, tmp_path):
    output_path = tmp_path / "schema.md"
    monkeypatch.setattr(storage_schema_cli, "storage_schema_status_for_root", lambda root: {"ok": True})
    monkeypatch.setattr(storage_schema_cli, "format_storage_schema_markdown", lambda status: "# Schema")
    monkeypatch.setattr("sys.argv", ["storage_schema.py", "--format", "markdown", "--output", str(output_path)])

    assert storage_schema_cli.main() == 0
    assert output_path.read_text(encoding="utf-8") == "# Schema\n"


def test_storage_schema_cli_returns_nonzero_for_drift(monkeypatch, capsys):
    monkeypatch.setattr(storage_schema_cli, "storage_schema_status_for_root", lambda root: {"ok": False})
    monkeypatch.setattr("sys.argv", ["storage_schema.py", "--target-root", "/tmp/root"])

    assert storage_schema_cli.main() == 1
    assert json.loads(capsys.readouterr().out) == {"ok": False}


def test_storage_schema_cli_reports_sanitized_os_errors(monkeypatch, capsys):
    def fail_status(root):
        raise OSError(13, "Permission denied", "/private/hunter2-schema.sqlite3")

    monkeypatch.setattr(storage_schema_cli, "storage_schema_status_for_root", fail_status)
    monkeypatch.setattr("sys.argv", ["storage_schema.py"])

    assert storage_schema_cli.main() == 2
    assert_sanitized_cli_os_error(capsys.readouterr())


def test_platform_migration_preflight_cli_default_allows_local_preflight(monkeypatch, capsys):
    monkeypatch.setattr(
        platform_migration_preflight_cli,
        "collect_platform_migration_preflight",
        lambda project_root: {"preflight_ok": True, "activation_ready": False},
    )
    monkeypatch.setattr("sys.argv", ["platform_migration_preflight.py", "--target-root", "/tmp/root"])

    assert platform_migration_preflight_cli.main() == 0
    assert json.loads(capsys.readouterr().out) == {
        "activation_ready": False,
        "preflight_ok": True,
    }


def test_platform_migration_preflight_cli_can_require_activation(monkeypatch, capsys):
    monkeypatch.setattr(
        platform_migration_preflight_cli,
        "collect_platform_migration_preflight",
        lambda project_root: {"preflight_ok": True, "activation_ready": False},
    )
    monkeypatch.setattr(
        "sys.argv",
        ["platform_migration_preflight.py", "--target-root", "/tmp/root", "--require-activation"],
    )

    assert platform_migration_preflight_cli.main() == 1
    assert json.loads(capsys.readouterr().out)["activation_ready"] is False


def test_platform_migration_preflight_cli_writes_markdown(monkeypatch, tmp_path):
    output_path = tmp_path / "preflight.md"
    monkeypatch.setattr(
        platform_migration_preflight_cli,
        "collect_platform_migration_preflight",
        lambda project_root: {"preflight_ok": True, "activation_ready": True},
    )
    monkeypatch.setattr(
        platform_migration_preflight_cli,
        "format_platform_migration_preflight_markdown",
        lambda status: "# Preflight",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "platform_migration_preflight.py",
            "--format",
            "markdown",
            "--output",
            str(output_path),
        ],
    )

    assert platform_migration_preflight_cli.main() == 0
    assert output_path.read_text(encoding="utf-8") == "# Preflight\n"


def test_platform_migration_preflight_cli_reports_os_errors(monkeypatch, capsys):
    def fail_collect(*, project_root):
        raise OSError(13, "Permission denied", "/private/hunter2-target")

    monkeypatch.setattr(
        platform_migration_preflight_cli,
        "collect_platform_migration_preflight",
        fail_collect,
    )
    monkeypatch.setattr("sys.argv", ["platform_migration_preflight.py"])

    assert platform_migration_preflight_cli.main() == 2
    captured = capsys.readouterr()
    assert "error: Permission denied" in captured.err
    assert "/private" not in captured.err
    assert "hunter2" not in captured.err


def test_platform_migration_rehearsal_cli_uses_temporary_staging(monkeypatch, capsys):
    seen = {}

    def fake_rehearsal(**kwargs):
        seen.update(kwargs)
        return {"rehearsal_ok": True, "staging_root_retained": True}

    monkeypatch.setattr(platform_migration_rehearsal_cli, "run_storage_migration_rehearsal", fake_rehearsal)
    monkeypatch.setattr("sys.argv", ["platform_migration_rehearsal.py", "--target-root", "/tmp/root"])

    assert platform_migration_rehearsal_cli.main() == 0
    output = json.loads(capsys.readouterr().out)
    assert output == {"rehearsal_ok": True, "staging_root_retained": False}
    assert str(seen["project_root"]) == "/tmp/root"
    assert seen["overwrite_staging"] is False
    assert seen["include_runtime_dependencies"] is False
    assert seen["include_object_manifest"] is False
    assert seen["include_job_store_manifest"] is False
    assert seen["object_key_prefix"] == "fluxmind-runtime"


def test_platform_migration_rehearsal_cli_can_include_object_manifest(monkeypatch, capsys):
    seen = {}

    def fake_rehearsal(**kwargs):
        seen.update(kwargs)
        return {
            "rehearsal_ok": True,
            "summary": {"object_manifest_ready": True},
            "object_storage_manifest": {"mode": "object_storage_migration_manifest"},
        }

    monkeypatch.setattr(platform_migration_rehearsal_cli, "run_storage_migration_rehearsal", fake_rehearsal)
    monkeypatch.setattr(
        "sys.argv",
        [
            "platform_migration_rehearsal.py",
            "--target-root",
            "/tmp/root",
            "--include-object-manifest",
            "--object-key-prefix",
            "lab-runtime",
        ],
    )

    assert platform_migration_rehearsal_cli.main() == 0
    output = json.loads(capsys.readouterr().out)
    assert output["object_storage_manifest"]["mode"] == "object_storage_migration_manifest"
    assert seen["include_object_manifest"] is True
    assert seen["object_key_prefix"] == "lab-runtime"


def test_platform_migration_rehearsal_cli_can_include_job_store_manifest(monkeypatch, capsys):
    seen = {}

    def fake_rehearsal(**kwargs):
        seen.update(kwargs)
        return {
            "rehearsal_ok": True,
            "summary": {"job_store_manifest_ready": True},
            "job_store_manifest": {"mode": "job_store_migration_manifest"},
        }

    monkeypatch.setattr(platform_migration_rehearsal_cli, "run_storage_migration_rehearsal", fake_rehearsal)
    monkeypatch.setattr(
        "sys.argv",
        [
            "platform_migration_rehearsal.py",
            "--target-root",
            "/tmp/root",
            "--include-job-store-manifest",
        ],
    )

    assert platform_migration_rehearsal_cli.main() == 0
    output = json.loads(capsys.readouterr().out)
    assert output["job_store_manifest"]["mode"] == "job_store_migration_manifest"
    assert seen["include_job_store_manifest"] is True


def test_platform_migration_rehearsal_cli_verifies_object_manifest_from_stdin(
    monkeypatch, capsys
):
    seen = {}

    def fake_verify(manifest, **kwargs):
        seen["manifest"] = manifest
        seen.update(kwargs)
        return {"mode": "object_storage_migration_manifest_verify", "ok": True}

    monkeypatch.setattr(
        platform_migration_rehearsal_cli,
        "verify_object_storage_migration_manifest",
        fake_verify,
    )
    monkeypatch.setattr("sys.stdin", io.StringIO('{"mode":"object_storage_migration_manifest"}'))
    monkeypatch.setattr(
        "sys.argv",
        [
            "platform_migration_rehearsal.py",
            "--target-root",
            "/tmp/root",
            "--verify-object-manifest",
            "-",
        ],
    )

    assert platform_migration_rehearsal_cli.main() == 0
    output = json.loads(capsys.readouterr().out)
    assert output == {"mode": "object_storage_migration_manifest_verify", "ok": True}
    assert seen["manifest"]["mode"] == "object_storage_migration_manifest"
    assert str(seen["project_root"]) == "/tmp/root"
    assert seen["include_runtime_dependencies"] is None


def test_platform_migration_rehearsal_cli_verifies_job_store_manifest_from_stdin(
    monkeypatch, capsys
):
    seen = {}

    def fake_verify(manifest, **kwargs):
        seen["manifest"] = manifest
        seen.update(kwargs)
        return {"mode": "job_store_migration_manifest_verify", "ok": True}

    monkeypatch.setattr(
        platform_migration_rehearsal_cli,
        "verify_job_store_migration_manifest",
        fake_verify,
    )
    monkeypatch.setattr("sys.stdin", io.StringIO('{"mode":"job_store_migration_manifest"}'))
    monkeypatch.setattr(
        "sys.argv",
        [
            "platform_migration_rehearsal.py",
            "--target-root",
            "/tmp/root",
            "--verify-job-store-manifest",
            "-",
        ],
    )

    assert platform_migration_rehearsal_cli.main() == 0
    output = json.loads(capsys.readouterr().out)
    assert output == {"mode": "job_store_migration_manifest_verify", "ok": True}
    assert seen["manifest"]["mode"] == "job_store_migration_manifest"
    assert str(seen["project_root"]) == "/tmp/root"


def test_platform_migration_rehearsal_cli_job_store_verify_failure_returns_nonzero(
    monkeypatch, tmp_path
):
    manifest_path = tmp_path / "job-manifest.json"
    output_path = tmp_path / "job-verify.md"
    manifest_path.write_text('{"mode":"job_store_migration_manifest"}', encoding="utf-8")
    monkeypatch.setattr(
        platform_migration_rehearsal_cli,
        "verify_job_store_migration_manifest",
        lambda manifest, **kwargs: {
            "mode": "job_store_migration_manifest_verify",
            "ok": False,
        },
    )
    monkeypatch.setattr(
        platform_migration_rehearsal_cli,
        "format_job_store_migration_verify_markdown",
        lambda status: "# Job Verify",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "platform_migration_rehearsal.py",
            "--verify-job-store-manifest",
            str(manifest_path),
            "--format",
            "markdown",
            "--output",
            str(output_path),
        ],
    )

    assert platform_migration_rehearsal_cli.main() == 1
    assert output_path.read_text(encoding="utf-8") == "# Job Verify\n"


def test_platform_migration_rehearsal_cli_verify_failure_returns_nonzero(
    monkeypatch, tmp_path
):
    manifest_path = tmp_path / "object-manifest.json"
    output_path = tmp_path / "verify.md"
    manifest_path.write_text('{"mode":"object_storage_migration_manifest"}', encoding="utf-8")
    monkeypatch.setattr(
        platform_migration_rehearsal_cli,
        "verify_object_storage_migration_manifest",
        lambda manifest, **kwargs: {
            "mode": "object_storage_migration_manifest_verify",
            "ok": False,
        },
    )
    monkeypatch.setattr(
        platform_migration_rehearsal_cli,
        "format_object_storage_migration_verify_markdown",
        lambda status: "# Verify",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "platform_migration_rehearsal.py",
            "--verify-object-manifest",
            str(manifest_path),
            "--format",
            "markdown",
            "--output",
            str(output_path),
        ],
    )

    assert platform_migration_rehearsal_cli.main() == 1
    assert output_path.read_text(encoding="utf-8") == "# Verify\n"


def test_platform_migration_rehearsal_cli_retained_staging_can_fail(monkeypatch, capsys):
    monkeypatch.setattr(
        platform_migration_rehearsal_cli,
        "run_storage_migration_rehearsal",
        lambda **kwargs: {"rehearsal_ok": False, "blockers": ["staging_root_not_empty"]},
    )
    monkeypatch.setattr(
        "sys.argv",
        ["platform_migration_rehearsal.py", "--staging-root", "/tmp/stage", "--overwrite-staging"],
    )

    assert platform_migration_rehearsal_cli.main() == 1
    assert json.loads(capsys.readouterr().out)["blockers"] == ["staging_root_not_empty"]


def test_platform_migration_rehearsal_cli_writes_markdown(monkeypatch, tmp_path):
    output_path = tmp_path / "rehearsal.md"
    monkeypatch.setattr(
        platform_migration_rehearsal_cli,
        "run_storage_migration_rehearsal",
        lambda **kwargs: {"rehearsal_ok": True},
    )
    monkeypatch.setattr(
        platform_migration_rehearsal_cli,
        "format_storage_migration_rehearsal_markdown",
        lambda status: "# Rehearsal",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "platform_migration_rehearsal.py",
            "--format",
            "markdown",
            "--output",
            str(output_path),
        ],
    )

    assert platform_migration_rehearsal_cli.main() == 0
    assert output_path.read_text(encoding="utf-8") == "# Rehearsal\n"


def test_platform_migration_rehearsal_cli_reports_os_errors(monkeypatch, capsys):
    def fail_rehearsal(**kwargs):
        raise OSError(13, "Permission denied", "/private/hunter2-runtime")

    monkeypatch.setattr(
        platform_migration_rehearsal_cli,
        "run_storage_migration_rehearsal",
        fail_rehearsal,
    )
    monkeypatch.setattr("sys.argv", ["platform_migration_rehearsal.py"])

    assert platform_migration_rehearsal_cli.main() == 2
    captured = capsys.readouterr()
    assert "error: Permission denied" in captured.err
    assert "/private" not in captured.err
    assert "hunter2" not in captured.err


def test_product_readiness_cli_default_allows_local_foundation(monkeypatch, capsys):
    monkeypatch.setattr(
        product_readiness_cli,
        "collect_product_readiness",
        lambda: {"local_foundation_ready": True, "activation_ready": False},
    )
    monkeypatch.setattr("sys.argv", ["product_readiness.py"])

    assert product_readiness_cli.main() == 0
    assert json.loads(capsys.readouterr().out) == {
        "activation_ready": False,
        "local_foundation_ready": True,
    }


def test_product_readiness_cli_can_require_activation(monkeypatch, capsys):
    monkeypatch.setattr(
        product_readiness_cli,
        "collect_product_readiness",
        lambda: {"local_foundation_ready": True, "activation_ready": False},
    )
    monkeypatch.setattr("sys.argv", ["product_readiness.py", "--require-activation"])

    assert product_readiness_cli.main() == 1
    assert json.loads(capsys.readouterr().out)["activation_ready"] is False


def test_product_readiness_cli_writes_markdown(monkeypatch, tmp_path):
    output_path = tmp_path / "product.md"
    monkeypatch.setattr(
        product_readiness_cli,
        "collect_product_readiness",
        lambda: {"local_foundation_ready": True, "activation_ready": True},
    )
    monkeypatch.setattr(
        product_readiness_cli,
        "format_product_readiness_markdown",
        lambda status: "# Product",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "product_readiness.py",
            "--format",
            "markdown",
            "--output",
            str(output_path),
        ],
    )

    assert product_readiness_cli.main() == 0
    assert output_path.read_text(encoding="utf-8") == "# Product\n"


def test_product_readiness_cli_reports_sanitized_os_errors(monkeypatch, capsys):
    def fail_collect():
        raise OSError(13, "Permission denied", "/private/hunter2-product.sqlite3")

    monkeypatch.setattr(product_readiness_cli, "collect_product_readiness", fail_collect)
    monkeypatch.setattr("sys.argv", ["product_readiness.py"])

    assert product_readiness_cli.main() == 2
    assert_sanitized_cli_os_error(capsys.readouterr())


def test_provider_readiness_cli_default_allows_local_foundation(monkeypatch, capsys):
    monkeypatch.setattr(
        provider_readiness_cli,
        "collect_provider_readiness",
        lambda: {"local_foundation_ready": True, "activation_ready": False},
    )
    monkeypatch.setattr("sys.argv", ["provider_readiness.py"])

    assert provider_readiness_cli.main() == 0
    assert json.loads(capsys.readouterr().out) == {
        "activation_ready": False,
        "local_foundation_ready": True,
    }


def test_provider_readiness_cli_can_require_activation(monkeypatch, capsys):
    monkeypatch.setattr(
        provider_readiness_cli,
        "collect_provider_readiness",
        lambda: {"local_foundation_ready": True, "activation_ready": False},
    )
    monkeypatch.setattr("sys.argv", ["provider_readiness.py", "--require-activation"])

    assert provider_readiness_cli.main() == 1
    assert json.loads(capsys.readouterr().out)["activation_ready"] is False


def test_provider_readiness_cli_writes_markdown(monkeypatch, tmp_path):
    output_path = tmp_path / "provider.md"
    monkeypatch.setattr(
        provider_readiness_cli,
        "collect_provider_readiness",
        lambda: {"local_foundation_ready": True, "activation_ready": True},
    )
    monkeypatch.setattr(
        provider_readiness_cli,
        "format_provider_readiness_markdown",
        lambda status: "# Provider",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "provider_readiness.py",
            "--format",
            "markdown",
            "--output",
            str(output_path),
        ],
    )

    assert provider_readiness_cli.main() == 0
    assert output_path.read_text(encoding="utf-8") == "# Provider\n"


def test_provider_readiness_cli_reports_sanitized_os_errors(monkeypatch, capsys):
    def fail_collect():
        raise OSError(13, "Permission denied", "/private/hunter2-provider.json")

    monkeypatch.setattr(provider_readiness_cli, "collect_provider_readiness", fail_collect)
    monkeypatch.setattr("sys.argv", ["provider_readiness.py"])

    assert provider_readiness_cli.main() == 2
    assert_sanitized_cli_os_error(capsys.readouterr())


def test_quality_readiness_cli_default_allows_local_foundation(monkeypatch, capsys):
    monkeypatch.setattr(
        quality_readiness_cli,
        "collect_quality_readiness",
        lambda **kwargs: {
            "local_foundation_ready": True,
            "small_group_ready": False,
            "community_ready": False,
        },
    )
    monkeypatch.setattr("sys.argv", ["quality_readiness.py"])

    assert quality_readiness_cli.main() == 0
    assert json.loads(capsys.readouterr().out) == {
        "community_ready": False,
        "local_foundation_ready": True,
        "small_group_ready": False,
    }


def test_quality_readiness_cli_can_require_community(monkeypatch, capsys):
    monkeypatch.setattr(
        quality_readiness_cli,
        "collect_quality_readiness",
        lambda **kwargs: {
            "local_foundation_ready": True,
            "small_group_ready": True,
            "community_ready": False,
        },
    )
    monkeypatch.setattr("sys.argv", ["quality_readiness.py", "--require-target", "community"])

    assert quality_readiness_cli.main() == 1
    assert json.loads(capsys.readouterr().out)["community_ready"] is False


def test_quality_readiness_cli_writes_markdown(monkeypatch, tmp_path):
    output_path = tmp_path / "quality.md"
    monkeypatch.setattr(
        quality_readiness_cli,
        "collect_quality_readiness",
        lambda **kwargs: {
            "local_foundation_ready": True,
            "small_group_ready": True,
            "community_ready": True,
        },
    )
    monkeypatch.setattr(
        quality_readiness_cli,
        "format_quality_readiness_markdown",
        lambda status: "# Quality",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "quality_readiness.py",
            "--format",
            "markdown",
            "--output",
            str(output_path),
        ],
    )

    assert quality_readiness_cli.main() == 0
    assert output_path.read_text(encoding="utf-8") == "# Quality\n"


def test_quality_readiness_cli_reports_os_errors(monkeypatch, capsys):
    def fail_collect(**kwargs):
        raise OSError(13, "Permission denied", "/private/hunter2-eval.json")

    monkeypatch.setattr(quality_readiness_cli, "collect_quality_readiness", fail_collect)
    monkeypatch.setattr("sys.argv", ["quality_readiness.py"])

    assert quality_readiness_cli.main() == 2
    captured = capsys.readouterr()
    assert "error: Permission denied" in captured.err
    assert "/private" not in captured.err
    assert "hunter2" not in captured.err


def test_run_job_worker_cli_prints_claimed_jobs(monkeypatch, capsys):
    created = []

    class FakeWorker:
        def __init__(self, *, worker_id, lease_seconds):
            created.append((worker_id, lease_seconds))

        def run_polling(self, *, poll_interval_s, max_jobs):
            assert poll_interval_s == 0.1
            assert max_jobs == 2
            return [SimpleNamespace(job_id="job-1", kind="index_rebuild", status="succeeded", worker_id="worker-1")]

    monkeypatch.setattr(run_job_worker_cli, "LocalDurableJobWorker", FakeWorker)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_job_worker.py",
            "--worker-id",
            "worker-1",
            "--lease-seconds",
            "42",
            "--loop",
            "--max-jobs",
            "2",
            "--poll-interval-s",
            "0.1",
        ],
    )

    assert run_job_worker_cli.main() == 0
    assert created == [("worker-1", 42)]
    assert "job_id=job-1 kind=index_rebuild status=succeeded worker_id=worker-1" in capsys.readouterr().out


def test_run_job_worker_cli_prints_no_due_jobs(monkeypatch, capsys):
    class FakeWorker:
        def __init__(self, *, worker_id, lease_seconds):
            pass

        def run_until_empty(self, *, max_jobs):
            assert max_jobs == 1
            return []

    monkeypatch.setattr(run_job_worker_cli, "LocalDurableJobWorker", FakeWorker)
    monkeypatch.setattr("sys.argv", ["run_job_worker.py"])

    assert run_job_worker_cli.main() == 0
    assert capsys.readouterr().out == "no_due_jobs=1\n"


def test_update_local_references_rewrite_text():
    updated, count = update_refs_cli.rewrite_text(
        "old=/home/shallow/04.AI-Prism/80.FluxMind "
        "archive=/home/shallow/04.AI-Prism/90.Archive/80-FluxMind"
    )

    assert count == 2
    assert "/home/shallow/04.AI-Prism/11.FluxMind" in updated
    assert "/home/shallow/04.AI-Prism/90.Archive/11-FluxMind-PreFormal" in updated


def test_update_local_references_dry_run_and_apply(monkeypatch, tmp_path, capsys):
    target = tmp_path / "config.toml"
    target.write_text("path='/home/shallow/04.AI-Prism/80.FluxMind'\n", encoding="utf-8")
    missing = tmp_path / "missing.toml"
    monkeypatch.setattr(update_refs_cli, "TARGETS", [target, missing])

    monkeypatch.setattr("sys.argv", ["update_local_references.py"])
    assert update_refs_cli.main() == 0
    dry_run_output = capsys.readouterr().out
    assert "would update" in dry_run_output
    assert "skip missing" in dry_run_output
    assert "dry-run only" in dry_run_output
    assert "80.FluxMind" in target.read_text(encoding="utf-8")

    monkeypatch.setattr("sys.argv", ["update_local_references.py", "--apply"])
    assert update_refs_cli.main() == 0
    assert "11.FluxMind" in target.read_text(encoding="utf-8")
    assert target.with_name("config.toml.bak-fluxmind-11").exists()


def test_update_local_references_apply_without_hits_returns_nonzero(monkeypatch, tmp_path):
    target = tmp_path / "config.toml"
    target.write_text("path='/tmp/current'\n", encoding="utf-8")
    monkeypatch.setattr(update_refs_cli, "TARGETS", [target])
    monkeypatch.setattr("sys.argv", ["update_local_references.py", "--apply"])

    assert update_refs_cli.main() == 1
    assert not target.with_name("config.toml.bak-fluxmind-11").exists()
