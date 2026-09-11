import io
import json
from types import SimpleNamespace

import scripts.evaluate_rag as evaluate_rag_cli
import scripts.run_job_worker as run_job_worker_cli
import scripts.runtime_manifest as runtime_manifest_cli
from scripts._cli import format_error


def test_cli_error_formatting_preserves_actionable_context():
    error = OSError(
        "cannot read /private/report.json from https://secret.example/path "
        "token=sk-test-secret-token"
    )

    message = format_error(error)
    assert "cannot read" in message
    assert "/private/report.json" in message
    assert "secret.example" in message


def test_evaluate_rag_cli_writes_json_report(monkeypatch, tmp_path, capsys):
    case = SimpleNamespace(ok=True, case_id="case-1", message="ok")
    provider = SimpleNamespace(
        ok=True,
        fixture_id="timeout",
        expected_code="provider_timeout",
        actual_code="provider_timeout",
    )
    gate = SimpleNamespace(ok=True, gate_id="minimum_case_count", message="ok")
    monkeypatch.setattr(
        evaluate_rag_cli,
        "load_eval_config",
        lambda path: {"loaded": str(path)},
    )
    monkeypatch.setattr(
        evaluate_rag_cli,
        "evaluate_config",
        lambda config: ([case], [case], [case], [case], [provider], [case]),
    )
    monkeypatch.setattr(
        evaluate_rag_cli,
        "evaluate_regression_gates",
        lambda *args, **kwargs: [gate],
    )
    monkeypatch.setattr(
        evaluate_rag_cli,
        "build_evaluation_report",
        lambda *args, **kwargs: {"ok": True},
    )
    report_path = tmp_path / "eval-report.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_rag.py",
            "--file",
            str(tmp_path / "config.json"),
            "--json-report",
            str(report_path),
        ],
    )

    assert evaluate_rag_cli.main() == 0
    assert json.loads(report_path.read_text(encoding="utf-8")) == {"ok": True}
    assert "ok   eval case case-1: ok" in capsys.readouterr().out


def test_evaluate_rag_cli_reports_failed_gate(monkeypatch, tmp_path, capsys):
    failed = SimpleNamespace(ok=False, case_id="case-fail", message="missing source")
    gate = SimpleNamespace(
        ok=False,
        gate_id="minimum_case_count",
        message="too few",
    )
    monkeypatch.setattr(evaluate_rag_cli, "load_eval_config", lambda path: {})
    monkeypatch.setattr(
        evaluate_rag_cli,
        "evaluate_config",
        lambda config: ([failed], [], [], [], [], []),
    )
    monkeypatch.setattr(
        evaluate_rag_cli,
        "evaluate_regression_gates",
        lambda *args, **kwargs: [gate],
    )
    monkeypatch.setattr(
        "sys.argv",
        ["evaluate_rag.py", "--file", str(tmp_path / "config.json")],
    )

    assert evaluate_rag_cli.main() == 1
    output = capsys.readouterr().out
    assert "fail eval case case-fail" in output
    assert "regression gate minimum_case_count" in output


def test_runtime_manifest_cli_outputs_manifest(monkeypatch, tmp_path):
    output_path = tmp_path / "manifest.md"
    monkeypatch.setattr(
        runtime_manifest_cli,
        "collect_runtime_backup_manifest",
        lambda: {"ok": True},
    )
    monkeypatch.setattr(
        runtime_manifest_cli,
        "format_runtime_backup_manifest_markdown",
        lambda manifest: "# Manifest",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "runtime_manifest.py",
            "--format",
            "markdown",
            "--output",
            str(output_path),
        ],
    )

    assert runtime_manifest_cli.main() == 0
    assert output_path.read_text(encoding="utf-8") == "# Manifest\n"


def test_runtime_manifest_cli_restore_check_uses_stdin(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO('{"groups": []}'))
    monkeypatch.setattr(
        runtime_manifest_cli,
        "collect_runtime_restore_check",
        lambda manifest, project_root: {
            "ok": False,
            "root": str(project_root),
            "groups": manifest["groups"],
        },
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "runtime_manifest.py",
            "--restore-check",
            "-",
            "--target-root",
            "/tmp/root",
        ],
    )

    assert runtime_manifest_cli.main() == 1
    output = json.loads(capsys.readouterr().out)
    assert output["ok"] is False
    assert output["root"] == "/tmp/root"


def test_run_job_worker_cli_prints_claimed_jobs(monkeypatch, capsys):
    created = []

    class FakeWorker:
        def __init__(self, *, worker_id, lease_seconds):
            created.append((worker_id, lease_seconds))

        def run_polling(self, *, poll_interval_s, max_jobs):
            assert poll_interval_s == 0.1
            assert max_jobs == 2
            return [
                SimpleNamespace(
                    job_id="job-1",
                    kind="index_rebuild",
                    status="succeeded",
                    worker_id="worker-1",
                )
            ]

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
    assert "job_id=job-1" in capsys.readouterr().out


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
