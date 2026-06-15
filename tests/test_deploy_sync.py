from pathlib import Path

import pytest

from scripts import deploy_sync


def test_build_rsync_command_is_dry_run_by_default_and_excludes_runtime_state():
    command = deploy_sync.build_rsync_command(
        host="root@example.test",
        remote_path="/opt/fluxmind/",
        apply=False,
    )

    assert command[:6] == ["rsync", "-az", "--delete", "--itemize-changes", "--human-readable", "--dry-run"]
    assert command[-1] == "root@example.test:/opt/fluxmind/"
    for pattern in deploy_sync.REQUIRED_RUNTIME_EXCLUDES:
        assert ["--exclude", pattern] in [command[index : index + 2] for index in range(len(command) - 1)]
    assert ["--exclude", "venv/"] in [command[index : index + 2] for index in range(len(command) - 1)]
    assert ["--exclude", "models/"] in [command[index : index + 2] for index in range(len(command) - 1)]
    assert ["--exclude", ".coverage"] in [command[index : index + 2] for index in range(len(command) - 1)]


def test_build_rsync_command_apply_removes_dry_run_only():
    dry_run = deploy_sync.build_rsync_command(
        host="root@example.test",
        remote_path="/opt/fluxmind/",
        apply=False,
    )
    applied = deploy_sync.build_rsync_command(
        host="root@example.test",
        remote_path="/opt/fluxmind/",
        apply=True,
    )

    assert "--dry-run" in dry_run
    assert "--dry-run" not in applied
    assert "--delete" in applied
    assert applied[-2:] == dry_run[-2:]


def test_validate_excludes_refuses_missing_runtime_exclude():
    excludes = tuple(item for item in deploy_sync.DEPLOY_EXCLUDES if item != "models/")

    with pytest.raises(RuntimeError, match="models/"):
        deploy_sync.validate_excludes(excludes)


def test_validate_project_root_requires_fluxmind_files(tmp_path):
    (tmp_path / "api.py").write_text("", encoding="utf-8")

    with pytest.raises(RuntimeError, match="app.py"):
        deploy_sync.validate_project_root(tmp_path)


def test_build_restart_command_uses_explicit_services():
    command = deploy_sync.build_restart_command(
        services=("fluxmind-api.service", "fluxmind-ui.service"),
    )

    assert command == (
        "systemctl restart fluxmind-api.service fluxmind-ui.service && "
        "systemctl is-active fluxmind-api.service fluxmind-ui.service"
    )


def test_main_dry_run_skips_restart(monkeypatch, capsys):
    commands = []

    def fake_run_command(command, *, check=True):
        commands.append(command)

    monkeypatch.setattr(deploy_sync, "run_command", fake_run_command)
    monkeypatch.setattr(
        "sys.argv",
        ["deploy_sync.py", "--host", "root@example.test", "--remote-path", "/srv/fluxmind/", "--restart"],
    )

    assert deploy_sync.main() == 0
    assert len(commands) == 1
    assert commands[0][0] == "rsync"
    assert "--dry-run" in commands[0]
    assert commands[0][-1] == "root@example.test:/srv/fluxmind/"
    output = capsys.readouterr().out
    assert "dry_run=1 use --apply to sync" in output
    assert "skip_restart=1 dry-run mode" in output


def test_main_apply_restart_uses_custom_services(monkeypatch):
    commands = []

    def fake_run_command(command, *, check=True):
        commands.append(command)

    monkeypatch.setattr(deploy_sync, "run_command", fake_run_command)
    monkeypatch.setattr(
        "sys.argv",
        [
            "deploy_sync.py",
            "--apply",
            "--restart",
            "--host",
            "root@example.test",
            "--service",
            "fluxmind-api.service",
            "--service",
            "fluxmind-worker.service",
        ],
    )

    assert deploy_sync.main() == 0
    assert len(commands) == 2
    assert commands[0][0] == "rsync"
    assert "--dry-run" not in commands[0]
    assert commands[1][:4] == ["ssh", "-o", "BatchMode=yes", "root@example.test"]
    assert commands[1][4] == (
        "systemctl restart fluxmind-api.service fluxmind-worker.service && "
        "systemctl is-active fluxmind-api.service fluxmind-worker.service"
    )


def test_main_reports_project_root_error(monkeypatch, capsys):
    monkeypatch.setattr(deploy_sync, "validate_project_root", lambda root=deploy_sync.PROJECT_ROOT: (_ for _ in ()).throw(RuntimeError("bad root")))
    monkeypatch.setattr("sys.argv", ["deploy_sync.py"])

    assert deploy_sync.main() == 2
    assert "bad root" in capsys.readouterr().err
