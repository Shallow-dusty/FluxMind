import io
import json
from pathlib import Path
from types import SimpleNamespace

import scripts.evaluate_rag as evaluate_rag_cli
import scripts.platform_migration_preflight as platform_migration_preflight_cli
import scripts.platform_migration_rehearsal as platform_migration_rehearsal_cli
import scripts.provider_readiness as provider_readiness_cli
import scripts.product_readiness as product_readiness_cli
import scripts.run_job_worker as run_job_worker_cli
import scripts.runtime_manifest as runtime_manifest_cli
import scripts.storage_schema as storage_schema_cli
import scripts.update_local_references as update_refs_cli


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
    result = SimpleNamespace(ok=True, case_id="case-1", message="ok", request_id="req-1")
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
    assert "ok   live answer case-1: ok request_id=req-1" in output
    assert "ok   live retrieval case-1: ok request_id=req-1" in output


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
        raise OSError("cannot read target")

    monkeypatch.setattr(
        platform_migration_preflight_cli,
        "collect_platform_migration_preflight",
        fail_collect,
    )
    monkeypatch.setattr("sys.argv", ["platform_migration_preflight.py"])

    assert platform_migration_preflight_cli.main() == 2
    assert "error: cannot read target" in capsys.readouterr().err


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
        raise OSError("cannot copy runtime")

    monkeypatch.setattr(
        platform_migration_rehearsal_cli,
        "run_storage_migration_rehearsal",
        fail_rehearsal,
    )
    monkeypatch.setattr("sys.argv", ["platform_migration_rehearsal.py"])

    assert platform_migration_rehearsal_cli.main() == 2
    assert "error: cannot copy runtime" in capsys.readouterr().err


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
