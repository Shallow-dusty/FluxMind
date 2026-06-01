from pathlib import Path
import hashlib

from src.capabilities import CodeExecutionRequest, ImageGenerationRequest
from src.execution_templates import PYTHON_EXECUTION_TEMPLATES
from src.providers import (
    LocalArtifactStore,
    LocalOctaveExecutionProvider,
    LocalPythonExecutionProvider,
    MockImageGenerationProvider,
    docker_execution_status,
)


def test_mock_image_provider_writes_svg_artifact(tmp_path: Path):
    provider = MockImageGenerationProvider(LocalArtifactStore(tmp_path))

    artifact = provider.generate(
        ImageGenerationRequest(
            prompt="SMC observer",
            diagram_template="sliding-mode-observer",
        )
    )

    assert artifact.kind == "image"
    assert artifact.mime_type == "image/svg+xml"
    assert artifact.metadata["provider"] == "local"
    assert artifact.metadata["diagram_template"] == "sliding-mode-observer"
    artifact_path = Path(artifact.uri.removeprefix("file://"))
    assert artifact_path.exists()
    svg = artifact_path.read_text(encoding="utf-8")
    assert "Sliding-Mode Observer" in svg
    assert "estimated states feedback" in svg
    assert artifact.metadata["checksum_sha256"] == hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    assert artifact.metadata["byte_count"] == str(artifact_path.stat().st_size)


def test_mock_image_provider_supports_paper_redraft_template(tmp_path: Path):
    provider = MockImageGenerationProvider(LocalArtifactStore(tmp_path))

    artifact = provider.generate(
        ImageGenerationRequest(
            prompt="Redraft the response curve from Fig. 2",
            diagram_template="paper-figure-redraft",
        )
    )

    artifact_path = Path(artifact.uri.removeprefix("file://"))
    svg = artifact_path.read_text(encoding="utf-8")
    assert artifact.metadata["diagram_template"] == "paper-figure-redraft"
    assert "Paper Figure Redraft" in svg
    assert "redraft" in svg


def test_local_python_execution_provider_runs_snippet():
    provider = LocalPythonExecutionProvider()

    result = provider.run(
        CodeExecutionRequest(
            language="python",
            entrypoint="main.py",
            files={"main.py": "print('fluxmind-ok')"},
        )
    )

    assert result.success is True
    assert result.stdout == "fluxmind-ok\n"
    assert result.stderr == ""
    assert result.runtime_metadata["language"] == "python"
    assert result.runtime_metadata["entrypoint"] == "main.py"
    assert result.runtime_metadata["input_file_count"] == "1"
    assert result.runtime_metadata["input_total_bytes"] == str(len("print('fluxmind-ok')".encode("utf-8")))
    assert result.runtime_metadata["provider_runtime"] == "python-local"
    assert result.runtime_metadata["runtime_available"] == "true"
    assert result.runtime_metadata["filesystem_isolation"] == "temporary_workdir"
    assert result.runtime_metadata["network_policy_enforced"] == "false"
    assert result.runtime_metadata["python_executable"]
    assert result.runtime_metadata["python_version"]
    assert result.runtime_metadata["python_implementation"]
    assert result.runtime_metadata["timeout_s"] == "30"
    assert result.runtime_metadata["cpu_time_s"] == "30"
    assert result.runtime_metadata["memory_mb"] == "512"
    assert result.runtime_metadata["memory_limit_enforced"] in {"true", "false"}
    assert result.runtime_metadata["cpu_limit_enforced"] in {"true", "false"}
    assert result.runtime_metadata["max_files"] == "32"
    assert result.runtime_metadata["max_file_bytes"] == str(256 * 1024)
    assert result.runtime_metadata["max_total_file_bytes"] == str(1024 * 1024)


def test_docker_execution_status_reports_not_configured(monkeypatch):
    monkeypatch.setattr("src.providers.shutil.which", lambda _name: "/usr/bin/docker")

    status = docker_execution_status(configured_backend="local", image="python:3.12-slim")

    assert status["configured"] is False
    assert status["available"] is False
    assert status["reason"] == "not_configured"


def test_docker_execution_status_reports_permission_denied(monkeypatch):
    class Completed:
        returncode = 1
        stdout = ""
        stderr = "permission denied while trying to connect to the docker API"

    monkeypatch.setattr("src.providers.shutil.which", lambda _name: "/usr/bin/docker")
    monkeypatch.setattr("src.providers.subprocess.run", lambda *_args, **_kwargs: Completed())

    status = docker_execution_status(configured_backend="docker", image="python:3.12-slim")

    assert status["configured"] is True
    assert status["available"] is False
    assert status["reason"] == "docker_permission_denied"


def test_local_python_execution_provider_captures_generated_files(tmp_path: Path):
    provider = LocalPythonExecutionProvider(LocalArtifactStore(tmp_path / "artifacts"))

    result = provider.run(
        CodeExecutionRequest(
            language="python",
            entrypoint="main.py",
            files={
                "main.py": (
                    "from pathlib import Path\n"
                    "Path('summary.txt').write_text('done', encoding='utf-8')\n"
                    "Path('plot.png').write_bytes(b'\\x89PNG\\r\\n\\x1a\\n')\n"
                )
            },
        )
    )

    assert result.success is True
    assert [artifact.title for artifact in result.artifacts] == ["plot.png", "summary.txt"]
    assert result.artifacts[0].kind == "plot"
    assert result.artifacts[0].mime_type == "image/png"
    assert result.artifacts[0].metadata["checksum_sha256"]
    assert result.artifacts[0].metadata["byte_count"] == "8"
    assert result.artifacts[1].kind == "text"
    assert result.artifacts[1].metadata["checksum_sha256"] == hashlib.sha256(b"done").hexdigest()
    assert result.artifacts[1].metadata["byte_count"] == "4"
    assert Path(result.artifacts[1].uri.removeprefix("file://")).read_text(encoding="utf-8") == "done"


def test_local_python_execution_provider_runs_smc_template(tmp_path: Path):
    provider = LocalPythonExecutionProvider(LocalArtifactStore(tmp_path / "artifacts"))

    result = provider.run(
        CodeExecutionRequest(
            language="python",
            entrypoint="main.py",
            files={"main.py": PYTHON_EXECUTION_TEMPLATES["smc_reaching_law"]},
        )
    )

    assert result.success is True
    assert "wrote smc_reaching_law.csv and smc_reaching_law.svg" in result.stdout
    assert [artifact.title for artifact in result.artifacts] == [
        "smc_reaching_law.csv",
        "smc_reaching_law.svg",
    ]
    assert result.artifacts[0].kind == "text"
    assert result.artifacts[1].kind == "plot"


def test_local_python_execution_provider_rejects_file_path_escape():
    provider = LocalPythonExecutionProvider()

    result = provider.run(
        CodeExecutionRequest(
            language="python",
            entrypoint="main.py",
            files={
                "main.py": "print('should-not-run')",
                "../escape.txt": "escape",
            },
        )
    )

    assert result.success is False
    assert result.exit_code == 2
    assert "escapes workdir" in result.stderr


def test_local_python_execution_provider_rejects_absolute_entrypoint():
    provider = LocalPythonExecutionProvider()

    result = provider.run(
        CodeExecutionRequest(
            language="python",
            entrypoint="/tmp/main.py",
            files={"main.py": "print('should-not-run')"},
        )
    )

    assert result.success is False
    assert result.exit_code == 2
    assert "Invalid execution file path" in result.stderr


def test_local_python_execution_provider_rejects_too_many_files():
    provider = LocalPythonExecutionProvider()
    files = {"main.py": "print('should-not-run')"}
    files.update({f"file-{index}.txt": "x" for index in range(32)})

    result = provider.run(
        CodeExecutionRequest(
            language="python",
            entrypoint="main.py",
            files=files,
        )
    )

    assert result.success is False
    assert result.exit_code == 2
    assert "Too many execution input files" in result.stderr


def test_local_python_execution_provider_rejects_large_file():
    provider = LocalPythonExecutionProvider()

    result = provider.run(
        CodeExecutionRequest(
            language="python",
            entrypoint="main.py",
            files={
                "main.py": "print('should-not-run')",
                "large.txt": "x" * (256 * 1024 + 1),
            },
        )
    )

    assert result.success is False
    assert result.exit_code == 2
    assert "too large" in result.stderr


def test_local_python_execution_provider_rejects_large_total_input():
    provider = LocalPythonExecutionProvider()
    files = {"main.py": "print('should-not-run')"}
    files.update({f"chunk-{index}.txt": "x" * (128 * 1024) for index in range(8)})

    result = provider.run(
        CodeExecutionRequest(
            language="python",
            entrypoint="main.py",
            files=files,
        )
    )

    assert result.success is False
    assert result.exit_code == 2
    assert "too large in total" in result.stderr


def test_local_python_execution_provider_skips_symlink_artifacts(tmp_path: Path):
    provider = LocalPythonExecutionProvider(LocalArtifactStore(tmp_path / "artifacts"))

    result = provider.run(
        CodeExecutionRequest(
            language="python",
            entrypoint="main.py",
            files={
                "main.py": (
                    "from pathlib import Path\n"
                    "Path('owned.txt').write_text('ok', encoding='utf-8')\n"
                    "Path('leak.txt').symlink_to('/etc/passwd')\n"
                )
            },
        )
    )

    assert result.success is True
    assert [artifact.title for artifact in result.artifacts] == ["owned.txt"]


def test_local_python_execution_provider_rejects_non_python():
    provider = LocalPythonExecutionProvider()

    result = provider.run(
        CodeExecutionRequest(language="octave", entrypoint="main.m", files={})
    )

    assert result.success is False
    assert "Unsupported local language" in result.stderr


def test_local_octave_execution_provider_reports_missing_binary(monkeypatch):
    monkeypatch.setattr("src.providers.shutil.which", lambda _name: None)
    provider = LocalOctaveExecutionProvider()

    result = provider.run(
        CodeExecutionRequest(
            language="octave",
            entrypoint="main.m",
            files={"main.m": "disp('ok');"},
        )
    )

    assert result.success is False
    assert result.exit_code == 127
    assert "GNU Octave executable not found" in result.stderr
    assert result.runtime_metadata["language"] == "octave"
    assert result.runtime_metadata["entrypoint"] == "main.m"
    assert result.runtime_metadata["input_file_count"] == "1"
    assert result.runtime_metadata["provider_runtime"] == "gnu-octave-local"
    assert result.runtime_metadata["runtime_available"] == "false"
    assert result.runtime_metadata["octave_available"] == "false"
    assert result.runtime_metadata["octave_executable"] == "octave"
    assert result.runtime_metadata["filesystem_isolation"] == "temporary_workdir"
    assert result.runtime_metadata["network_policy_enforced"] == "false"
    assert result.runtime_metadata["memory_limit_enforced"] == "false"
    assert result.runtime_metadata["cpu_limit_enforced"] == "false"


def test_local_octave_execution_provider_rejects_file_path_escape(tmp_path: Path, monkeypatch):
    fake_octave = tmp_path / "octave"
    fake_octave.write_text("#!/bin/sh\necho should-not-run\n", encoding="utf-8")
    fake_octave.chmod(0o755)
    monkeypatch.setattr("src.providers.shutil.which", lambda _name: str(fake_octave))
    provider = LocalOctaveExecutionProvider()

    result = provider.run(
        CodeExecutionRequest(
            language="octave",
            entrypoint="../main.m",
            files={"../main.m": "disp('bad');"},
        )
    )

    assert result.success is False
    assert result.exit_code == 2
    assert "escapes workdir" in result.stderr


def test_local_octave_execution_provider_rejects_large_file(tmp_path: Path, monkeypatch):
    fake_octave = tmp_path / "octave"
    fake_octave.write_text("#!/bin/sh\necho should-not-run\n", encoding="utf-8")
    fake_octave.chmod(0o755)
    monkeypatch.setattr("src.providers.shutil.which", lambda _name: str(fake_octave))
    provider = LocalOctaveExecutionProvider()

    result = provider.run(
        CodeExecutionRequest(
            language="octave",
            entrypoint="main.m",
            files={
                "main.m": "disp('bad');",
                "large.txt": "x" * (256 * 1024 + 1),
            },
        )
    )

    assert result.success is False
    assert result.exit_code == 2
    assert "too large" in result.stderr


def test_local_octave_execution_provider_runs_when_binary_exists(tmp_path: Path, monkeypatch):
    fake_octave = tmp_path / "octave"
    fake_octave.write_text(
        "#!/bin/sh\n"
        "echo octave-ok\n"
        "printf 'artifact-ok' > result.txt\n",
        encoding="utf-8",
    )
    fake_octave.chmod(0o755)
    monkeypatch.setattr("src.providers.shutil.which", lambda _name: str(fake_octave))
    provider = LocalOctaveExecutionProvider(LocalArtifactStore(tmp_path / "artifacts"))

    result = provider.run(
        CodeExecutionRequest(
            language="octave",
            entrypoint="main.m",
            files={"main.m": "disp('ok');"},
        )
    )

    assert result.success is True
    assert result.stdout == "octave-ok\n"
    assert result.runtime_metadata["provider_runtime"] == "gnu-octave-local"
    assert result.runtime_metadata["runtime_available"] == "true"
    assert result.runtime_metadata["octave_available"] == "true"
    assert result.runtime_metadata["octave_executable"] == "octave"
    assert result.runtime_metadata["octave_executable_resolved"] == "octave"
    assert result.runtime_metadata["network_policy_enforced"] == "false"
    assert result.runtime_metadata["cpu_time_s"] == "30"
    assert result.runtime_metadata["cpu_limit_enforced"] in {"true", "false"}
    assert result.artifacts[0].title == "result.txt"
    assert result.artifacts[0].metadata["checksum_sha256"] == hashlib.sha256(b"artifact-ok").hexdigest()
    assert result.artifacts[0].metadata["byte_count"] == str(len(b"artifact-ok"))
    assert result.artifacts[0].metadata["runtime"] == "gnu-octave-local"
