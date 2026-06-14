import hashlib
import io
import subprocess
from pathlib import Path

from src.capabilities import CodeExecutionRequest, ImageGenerationRequest
from src.execution_templates import OCTAVE_EXECUTION_TEMPLATES, PYTHON_EXECUTION_TEMPLATES
from src.execution_policy import POLICY_VIOLATION_EXIT_CODE
from src.providers import (
    BoundedStreamReader,
    DockerExecutionProvider,
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
    assert result.runtime_metadata["max_stdout_bytes"] == str(64 * 1024)
    assert result.runtime_metadata["max_stderr_bytes"] == str(64 * 1024)
    assert result.runtime_metadata["stdout_bytes"] == str(len("fluxmind-ok\n".encode("utf-8")))
    assert result.runtime_metadata["stderr_bytes"] == "0"
    assert result.runtime_metadata["stdout_truncated"] == "false"
    assert result.runtime_metadata["stderr_truncated"] == "false"
    assert result.runtime_metadata["output_truncated"] == "false"
    assert result.runtime_metadata["execution_policy"] == "local-safe-v1"
    assert result.runtime_metadata["execution_policy_enforced"] == "true"
    assert result.runtime_metadata["execution_policy_violations"] == "0"
    assert result.runtime_metadata["policy_violation"] == "false"


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


def test_docker_execution_status_reports_timeout_oserror_unavailable_and_ok(monkeypatch):
    monkeypatch.setattr("src.providers.shutil.which", lambda _name: "/usr/bin/docker")

    def timeout_run(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("docker", 3)

    monkeypatch.setattr("src.providers.subprocess.run", timeout_run)
    assert docker_execution_status(configured_backend="docker", image="python")["reason"] == "docker_timeout"

    def oserror_run(*_args, **_kwargs):
        raise FileNotFoundError("missing")

    monkeypatch.setattr("src.providers.subprocess.run", oserror_run)
    assert docker_execution_status(configured_backend="docker", image="python")["reason"] == "FileNotFoundError"

    class Unavailable:
        returncode = 2
        stdout = ""
        stderr = "Cannot connect to the Docker daemon"

    monkeypatch.setattr("src.providers.subprocess.run", lambda *_args, **_kwargs: Unavailable())
    unavailable = docker_execution_status(configured_backend="docker", image="python")
    assert unavailable["available"] is False
    assert unavailable["reason"] == "docker_unavailable"

    class Available:
        returncode = 0
        stdout = "25.0.0\n"
        stderr = ""

    monkeypatch.setattr("src.providers.subprocess.run", lambda *_args, **_kwargs: Available())
    available = docker_execution_status(configured_backend="docker", image="python")
    assert available["available"] is True
    assert available["reason"] == "ok"
    assert available["docker_server_version"] == "25.0.0"


def test_bounded_stream_reader_handles_text_stream_and_zero_limit():
    reader = BoundedStreamReader(io.StringIO("abcdef"), limit_bytes=0)

    reader.start()
    reader.join()

    assert reader.total_bytes == 6
    assert reader.truncated is True
    assert reader.text() == ""


def test_local_artifact_store_classifies_binary_file_artifacts(tmp_path: Path):
    artifact = LocalArtifactStore(tmp_path).write_bytes(
        "binary/result.bin",
        b"\x00\x01",
        "application/octet-stream",
    )

    assert artifact.kind == "file"
    assert artifact.title == "result.bin"
    assert artifact.metadata["byte_count"] == "2"


def test_docker_execution_provider_reports_missing_docker(monkeypatch):
    monkeypatch.setattr("src.providers.shutil.which", lambda _name: None)
    provider = DockerExecutionProvider()

    result = provider.run(
        CodeExecutionRequest(
            language="python",
            entrypoint="main.py",
            files={"main.py": "print('should-not-run')"},
        )
    )

    assert result.success is False
    assert result.exit_code == 127
    assert "Docker executable not found" in result.stderr
    assert result.runtime_metadata["provider_runtime"] == "docker-python"
    assert result.runtime_metadata["runtime_available"] == "false"
    assert result.runtime_metadata["filesystem_isolation"] == "docker_container_bind_mount"
    assert result.runtime_metadata["network_policy_enforced"] == "true"
    assert result.runtime_metadata["memory_limit_enforced"] == "true"
    assert result.runtime_metadata["cpu_limit_enforced"] == "true"
    assert result.runtime_metadata["docker_image"]


def test_docker_execution_provider_runs_with_sandbox_flags_and_collects_artifacts(
    tmp_path: Path,
    monkeypatch,
):
    captured: dict[str, list[str]] = {}

    class FakePopen:
        returncode = 0

        def __init__(self, command, **_kwargs):
            captured["command"] = command
            mount = command[command.index("-v") + 1]
            workdir = Path(mount.split(":", 1)[0])
            assert (workdir / "main.py").read_text(encoding="utf-8") == "print('docker-ok')"
            (workdir / "result.txt").write_text("artifact-ok", encoding="utf-8")

        def poll(self):
            return self.returncode

        def communicate(self, timeout=None):
            return "docker-ok\n", ""

    monkeypatch.setattr("src.providers.shutil.which", lambda _name: "/usr/bin/docker")
    monkeypatch.setattr("src.providers.subprocess.Popen", FakePopen)
    provider = DockerExecutionProvider(LocalArtifactStore(tmp_path / "artifacts"), image="python:3.12-slim")

    result = provider.run(
        CodeExecutionRequest(
            language="python",
            entrypoint="main.py",
            files={"main.py": "print('docker-ok')"},
            memory_mb=128,
        )
    )

    assert result.success is True
    assert result.stdout == "docker-ok\n"
    command = captured["command"]
    assert command[:2] == ["/usr/bin/docker", "run"]
    assert "--rm" in command
    assert command[command.index("--network") + 1] == "none"
    assert command[command.index("--memory") + 1] == "128m"
    assert command[command.index("--cpus") + 1] == "1"
    assert command[command.index("--pids-limit") + 1] == "64"
    assert command[command.index("--security-opt") + 1] == "no-new-privileges"
    assert command[command.index("--cap-drop") + 1] == "ALL"
    assert "--read-only" in command
    assert command[-3:] == ["python:3.12-slim", "python", "main.py"]
    assert result.runtime_metadata["provider_runtime"] == "docker-python"
    assert result.runtime_metadata["runtime_available"] == "true"
    assert result.runtime_metadata["docker_image"] == "python:3.12-slim"
    assert result.runtime_metadata["docker_returncode"] == "0"
    assert result.artifacts[0].title == "result.txt"
    assert result.artifacts[0].metadata["runtime"] == "docker-python"
    assert result.artifacts[0].metadata["docker_image"] == "python:3.12-slim"
    assert result.artifacts[0].metadata["checksum_sha256"] == hashlib.sha256(b"artifact-ok").hexdigest()


def test_local_python_execution_provider_truncates_large_output(monkeypatch):
    monkeypatch.setattr("src.providers.CODE_EXECUTION_MAX_STDOUT_BYTES", 32)
    monkeypatch.setattr("src.providers.CODE_EXECUTION_MAX_STDERR_BYTES", 48)
    provider = LocalPythonExecutionProvider()

    result = provider.run(
        CodeExecutionRequest(
            language="python",
            entrypoint="main.py",
            files={
                "main.py": (
                    "print('O' * 100)\n"
                    "raise RuntimeError('E' * 100)\n"
                )
            },
        )
    )

    assert result.success is False
    assert len(result.stdout.encode("utf-8")) == 32
    assert len(result.stderr.encode("utf-8")) == 48
    assert int(result.runtime_metadata["stdout_bytes"]) > 32
    assert int(result.runtime_metadata["stderr_bytes"]) > 48
    assert result.runtime_metadata["max_stdout_bytes"] == "32"
    assert result.runtime_metadata["max_stderr_bytes"] == "48"
    assert result.runtime_metadata["stdout_truncated"] == "true"
    assert result.runtime_metadata["stderr_truncated"] == "true"
    assert result.runtime_metadata["output_truncated"] == "true"


def test_docker_execution_provider_truncates_output_in_fallback_path(monkeypatch):
    class FakePopen:
        returncode = 0

        def __init__(self, *_args, **_kwargs):
            pass

        def poll(self):
            return self.returncode

        def communicate(self, timeout=None):
            return "O" * 30, "E" * 20

    monkeypatch.setattr("src.providers.CODE_EXECUTION_MAX_STDOUT_BYTES", 10)
    monkeypatch.setattr("src.providers.CODE_EXECUTION_MAX_STDERR_BYTES", 8)
    monkeypatch.setattr("src.providers.shutil.which", lambda _name: "/usr/bin/docker")
    monkeypatch.setattr("src.providers.subprocess.Popen", FakePopen)
    provider = DockerExecutionProvider(image="python:3.12-slim")

    result = provider.run(
        CodeExecutionRequest(
            language="python",
            entrypoint="main.py",
            files={"main.py": "print('docker-output')"},
        )
    )

    assert result.success is True
    assert result.stdout == "O" * 10
    assert result.stderr == "E" * 8
    assert result.runtime_metadata["stdout_bytes"] == "30"
    assert result.runtime_metadata["stderr_bytes"] == "20"
    assert result.runtime_metadata["stdout_truncated"] == "true"
    assert result.runtime_metadata["stderr_truncated"] == "true"


def test_docker_execution_provider_rejects_policy_violation_before_container(monkeypatch):
    monkeypatch.setattr("src.providers.shutil.which", lambda _name: "/usr/bin/docker")

    def fail_popen(*_args, **_kwargs):
        raise AssertionError("policy violation should not start docker")

    monkeypatch.setattr("src.providers.subprocess.Popen", fail_popen)
    provider = DockerExecutionProvider()

    result = provider.run(
        CodeExecutionRequest(
            language="python",
            entrypoint="main.py",
            files={"main.py": "import subprocess\nsubprocess.run(['echo', 'bad'])\n"},
        )
    )

    assert result.success is False
    assert result.exit_code == POLICY_VIOLATION_EXIT_CODE
    assert result.runtime_metadata["policy_violation"] == "true"
    assert result.runtime_metadata["execution_policy_enforced"] == "true"
    assert "python_import_not_allowed" in result.stderr


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


def test_local_python_execution_provider_limits_generated_artifacts(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr("src.providers.CODE_EXECUTION_MAX_ARTIFACTS", 2)
    monkeypatch.setattr("src.providers.CODE_EXECUTION_MAX_ARTIFACT_BYTES", 12)
    monkeypatch.setattr("src.providers.CODE_EXECUTION_MAX_ARTIFACT_TOTAL_BYTES", 20)
    monkeypatch.setattr("src.providers.CODE_EXECUTION_MAX_ARTIFACT_CANDIDATES", 16)
    provider = LocalPythonExecutionProvider(LocalArtifactStore(tmp_path / "artifacts"))

    result = provider.run(
        CodeExecutionRequest(
            language="python",
            entrypoint="main.py",
            files={
                "main.py": (
                    "from pathlib import Path\n"
                    "Path('a.txt').write_text('aaaa', encoding='utf-8')\n"
                    "Path('b.txt').write_text('bbbb', encoding='utf-8')\n"
                    "Path('c.txt').write_text('c' * 20, encoding='utf-8')\n"
                    "Path('d.txt').write_text('dddd', encoding='utf-8')\n"
                    "Path('e.txt').write_text('eeee', encoding='utf-8')\n"
                )
            },
        )
    )

    assert result.success is True
    assert [artifact.title for artifact in result.artifacts] == ["a.txt", "b.txt"]
    assert result.runtime_metadata["max_artifacts"] == "2"
    assert result.runtime_metadata["max_artifact_bytes"] == "12"
    assert result.runtime_metadata["max_artifact_total_bytes"] == "20"
    assert result.runtime_metadata["artifact_candidate_count"] == "5"
    assert result.runtime_metadata["artifact_exported_count"] == "2"
    assert result.runtime_metadata["artifact_exported_bytes"] == "8"
    assert result.runtime_metadata["artifact_skipped_too_large_count"] == "1"
    assert result.runtime_metadata["artifact_skipped_count_limit"] == "2"
    assert result.runtime_metadata["artifact_collection_truncated"] == "true"


def test_local_python_execution_provider_limits_total_artifact_bytes(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr("src.providers.CODE_EXECUTION_MAX_ARTIFACTS", 10)
    monkeypatch.setattr("src.providers.CODE_EXECUTION_MAX_ARTIFACT_BYTES", 12)
    monkeypatch.setattr("src.providers.CODE_EXECUTION_MAX_ARTIFACT_TOTAL_BYTES", 6)
    monkeypatch.setattr("src.providers.CODE_EXECUTION_MAX_ARTIFACT_CANDIDATES", 16)
    provider = LocalPythonExecutionProvider(LocalArtifactStore(tmp_path / "artifacts"))

    result = provider.run(
        CodeExecutionRequest(
            language="python",
            entrypoint="main.py",
            files={
                "main.py": (
                    "from pathlib import Path\n"
                    "Path('a.txt').write_text('aaaa', encoding='utf-8')\n"
                    "Path('b.txt').write_text('bbbb', encoding='utf-8')\n"
                )
            },
        )
    )

    assert result.success is True
    assert [artifact.title for artifact in result.artifacts] == ["a.txt"]
    assert result.runtime_metadata["artifact_skipped_total_bytes_limit"] == "1"
    assert result.runtime_metadata["artifact_collection_truncated"] == "true"


def test_local_python_execution_provider_limits_artifact_candidate_scan(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr("src.providers.CODE_EXECUTION_MAX_ARTIFACTS", 10)
    monkeypatch.setattr("src.providers.CODE_EXECUTION_MAX_ARTIFACT_BYTES", 12)
    monkeypatch.setattr("src.providers.CODE_EXECUTION_MAX_ARTIFACT_TOTAL_BYTES", 100)
    monkeypatch.setattr("src.providers.CODE_EXECUTION_MAX_ARTIFACT_CANDIDATES", 3)
    provider = LocalPythonExecutionProvider(LocalArtifactStore(tmp_path / "artifacts"))

    result = provider.run(
        CodeExecutionRequest(
            language="python",
            entrypoint="main.py",
            files={
                "main.py": (
                    "from pathlib import Path\n"
                    "for name in ['a.txt', 'b.txt', 'c.txt', 'd.txt']:\n"
                    "    Path(name).write_text('ok', encoding='utf-8')\n"
                )
            },
        )
    )

    assert result.success is True
    exported_titles = [artifact.title for artifact in result.artifacts]
    assert set(exported_titles).issubset({"a.txt", "b.txt", "c.txt", "d.txt"})
    assert 0 < len(exported_titles) < 4
    assert result.runtime_metadata["artifact_scanned_entries"] == "3"
    assert result.runtime_metadata["artifact_scan_truncated"] == "true"
    assert result.runtime_metadata["artifact_collection_truncated"] == "true"


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
    assert result.runtime_metadata["execution_policy_checked_files"] == "1"
    assert result.runtime_metadata["policy_violation"] == "false"


def test_local_python_execution_provider_runs_pmsm_current_step_template(tmp_path: Path):
    provider = LocalPythonExecutionProvider(LocalArtifactStore(tmp_path / "artifacts"))

    result = provider.run(
        CodeExecutionRequest(
            language="python",
            entrypoint="main.py",
            files={"main.py": PYTHON_EXECUTION_TEMPLATES["pmsm_current_step"]},
        )
    )

    assert result.success is True
    assert "wrote pmsm_current_step.csv and pmsm_current_step.svg" in result.stdout
    assert [artifact.title for artifact in result.artifacts] == [
        "pmsm_current_step.csv",
        "pmsm_current_step.svg",
    ]
    assert result.artifacts[0].kind == "text"
    assert result.artifacts[0].mime_type == "text/csv"
    assert result.artifacts[1].kind == "plot"
    assert result.runtime_metadata["policy_violation"] == "false"


def test_octave_execution_templates_are_well_formed():
    # Octave is not installed in CI, so this statically checks template breadth and
    # the expected output filenames without invoking the octave binary.
    assert set(OCTAVE_EXECUTION_TEMPLATES) >= {
        "hello",
        "pmsm_current_decay",
        "smc_sign_switching",
    }
    assert "pmsm_current_decay.csv" in OCTAVE_EXECUTION_TEMPLATES["pmsm_current_decay"]
    assert "smc_sign_switching.csv" in OCTAVE_EXECUTION_TEMPLATES["smc_sign_switching"]
    for body in OCTAVE_EXECUTION_TEMPLATES.values():
        assert body.strip()


def test_local_python_execution_provider_rejects_policy_violation():
    provider = LocalPythonExecutionProvider()

    result = provider.run(
        CodeExecutionRequest(
            language="python",
            entrypoint="main.py",
            files={"main.py": "import subprocess\nsubprocess.run(['echo', 'bad'])\n"},
        )
    )

    assert result.success is False
    assert result.exit_code == POLICY_VIOLATION_EXIT_CODE
    assert result.runtime_metadata["policy_violation"] == "true"
    assert result.runtime_metadata["execution_policy_violations"] != "0"
    assert "python_import_not_allowed" in result.stderr
    assert "subprocess" in result.stderr


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


def test_local_python_execution_provider_rejects_missing_entrypoint_file():
    provider = LocalPythonExecutionProvider()

    result = provider.run(
        CodeExecutionRequest(
            language="python",
            entrypoint="main.py",
            files={"helper.py": "print('not main')"},
        )
    )

    assert result.success is False
    assert result.exit_code == 2
    assert "Entrypoint not found: main.py" in result.stderr


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
    assert result.runtime_metadata["execution_policy_enforced"] == "true"
    assert result.runtime_metadata["policy_violation"] == "false"


def test_local_octave_execution_provider_rejects_policy_violation_before_runtime_lookup(monkeypatch):
    def fail_which(_name):
        raise AssertionError("policy violation should not look up octave")

    monkeypatch.setattr("src.providers.shutil.which", fail_which)
    provider = LocalOctaveExecutionProvider()

    result = provider.run(
        CodeExecutionRequest(
            language="octave",
            entrypoint="main.m",
            files={"main.m": "system('echo bad');"},
        )
    )

    assert result.success is False
    assert result.exit_code == POLICY_VIOLATION_EXIT_CODE
    assert result.runtime_metadata["policy_violation"] == "true"
    assert "octave_shell_call" in result.stderr


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


def test_local_octave_execution_provider_rejects_missing_entrypoint_file(tmp_path: Path, monkeypatch):
    fake_octave = tmp_path / "octave"
    fake_octave.write_text("#!/bin/sh\necho should-not-run\n", encoding="utf-8")
    fake_octave.chmod(0o755)
    monkeypatch.setattr("src.providers.shutil.which", lambda _name: str(fake_octave))
    provider = LocalOctaveExecutionProvider()

    result = provider.run(
        CodeExecutionRequest(
            language="octave",
            entrypoint="main.m",
            files={"helper.m": "disp('not main');"},
        )
    )

    assert result.success is False
    assert result.exit_code == 2
    assert "Entrypoint not found: main.m" in result.stderr


def test_docker_execution_provider_reports_popen_oserror(monkeypatch):
    monkeypatch.setattr("src.providers.shutil.which", lambda _name: "/usr/bin/docker")

    def fail_popen(*_args, **_kwargs):
        raise OSError("docker unavailable")

    monkeypatch.setattr("src.providers.subprocess.Popen", fail_popen)
    provider = DockerExecutionProvider()

    result = provider.run(
        CodeExecutionRequest(
            language="python",
            entrypoint="main.py",
            files={"main.py": "print('never')"},
        )
    )

    assert result.success is False
    assert result.exit_code == 127
    assert "Docker execution backend unavailable" in result.stderr
    assert result.runtime_metadata["runtime_available"] == "false"
    assert result.runtime_metadata["docker_error"] == "OSError"


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
