from types import SimpleNamespace
import importlib.util
from pathlib import Path

from src.capabilities import CodeExecutionResult, GeneratedArtifact


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "docker_execution_smoke.py"
_SPEC = importlib.util.spec_from_file_location("docker_execution_smoke", _SCRIPT_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
docker_execution_smoke = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(docker_execution_smoke)


def test_docker_execution_smoke_image_overrides():
    assert docker_execution_smoke._image_for_language(
        "python",
        python_image="custom-python:latest",
        octave_image=None,
    ) == "custom-python:latest"
    assert docker_execution_smoke._image_for_language(
        "octave",
        python_image=None,
        octave_image="custom-octave:latest",
    ) == "custom-octave:latest"


def test_docker_execution_smoke_image_overrides_ignore_blank_values():
    assert docker_execution_smoke._image_for_language(
        "python",
        python_image=" ",
        octave_image=None,
    ) == docker_execution_smoke.DOCKER_PYTHON_EXECUTION_IMAGE
    assert docker_execution_smoke._image_for_language(
        "octave",
        python_image=None,
        octave_image=" ",
    ) == docker_execution_smoke.DOCKER_OCTAVE_EXECUTION_IMAGE


def test_docker_execution_smoke_requires_expected_result_artifact(monkeypatch):
    class FakeProvider:
        def __init__(self, _store, *, image):
            self.image = image

        def run(self, _request):
            return CodeExecutionResult(
                exit_code=0,
                stdout="ok\n",
                stderr="",
                runtime_metadata={"docker_image": docker_execution_smoke.DOCKER_PYTHON_EXECUTION_IMAGE},
                artifacts=[
                    GeneratedArtifact(
                        kind="text",
                        uri="file:///tmp/other.txt",
                        mime_type="text/plain",
                        title="other.txt",
                    )
                ],
            )

    monkeypatch.setattr(docker_execution_smoke, "DockerExecutionProvider", FakeProvider)

    result = docker_execution_smoke.run_case("python", mode="provider")

    assert result["ok"] is False
    assert result["expected_artifact"] == "result.txt"
    assert result["expected_artifact_found"] is False


def test_docker_execution_smoke_accepts_expected_result_artifact(monkeypatch):
    class FakeProvider:
        def __init__(self, _store, *, image):
            self.image = image

        def run(self, _request):
            return CodeExecutionResult(
                exit_code=0,
                stdout="ok\n",
                stderr="",
                runtime_metadata={"docker_image": docker_execution_smoke.DOCKER_PYTHON_EXECUTION_IMAGE},
                artifacts=[
                    GeneratedArtifact(
                        kind="text",
                        uri="file:///tmp/result.txt",
                        mime_type="text/plain",
                        title="result.txt",
                    )
                ],
            )

    monkeypatch.setattr(docker_execution_smoke, "DockerExecutionProvider", FakeProvider)

    result = docker_execution_smoke.run_case("python", mode="provider")

    assert result["ok"] is True
    assert result["expected_artifact_found"] is True


def test_docker_execution_smoke_job_mode_applies_python_image_override(monkeypatch):
    previous_image = docker_execution_smoke.jobs_module.DOCKER_PYTHON_EXECUTION_IMAGE
    captured = {}

    class FakeRunner:
        def __init__(self, *_args, **_kwargs):
            pass

        def run_local_python(self, _request):
            captured["image"] = docker_execution_smoke.jobs_module.DOCKER_PYTHON_EXECUTION_IMAGE
            return SimpleNamespace(
                status="succeeded",
                result={
                    "exit_code": 0,
                    "stdout": "ok\n",
                    "stderr": "",
                    "runtime_metadata": {
                        "provider_runtime": "docker-python",
                        "docker_image": "custom-python:latest",
                    },
                },
                artifacts=[
                    {
                        "title": "result.txt",
                        "kind": "text",
                        "mime_type": "text/plain",
                        "uri": "file:///tmp/result.txt",
                    }
                ],
                error=None,
                job_id="job-smoke",
            )

    monkeypatch.setattr(docker_execution_smoke.jobs_module, "CODE_EXECUTION_BACKEND", "docker")
    monkeypatch.setattr(docker_execution_smoke, "LocalJobRunner", FakeRunner)

    result = docker_execution_smoke.run_case(
        "python",
        mode="job",
        python_image="custom-python:latest",
    )

    assert result["ok"] is True
    assert captured["image"] == "custom-python:latest"
    assert result["actual_image"] == "custom-python:latest"
    assert result["image_matches"] is True
    assert result["runtime_metadata"]["backend"] == "docker"
    assert docker_execution_smoke.jobs_module.DOCKER_PYTHON_EXECUTION_IMAGE == previous_image


def test_docker_execution_smoke_job_mode_applies_octave_image_override(monkeypatch):
    previous_image = docker_execution_smoke.jobs_module.DOCKER_OCTAVE_EXECUTION_IMAGE
    captured = {}

    class FakeRunner:
        def __init__(self, *_args, **_kwargs):
            pass

        def run_local_octave(self, _request):
            captured["image"] = docker_execution_smoke.jobs_module.DOCKER_OCTAVE_EXECUTION_IMAGE
            return SimpleNamespace(
                status="succeeded",
                result={
                    "exit_code": 0,
                    "stdout": "ok\n",
                    "stderr": "",
                    "runtime_metadata": {
                        "provider_runtime": "docker-octave",
                        "docker_image": "custom-octave:latest",
                    },
                },
                artifacts=[
                    {
                        "title": "result.txt",
                        "kind": "text",
                        "mime_type": "text/plain",
                        "uri": "file:///tmp/result.txt",
                    }
                ],
                error=None,
                job_id="job-smoke",
            )

    monkeypatch.setattr(docker_execution_smoke.jobs_module, "CODE_EXECUTION_BACKEND", "docker")
    monkeypatch.setattr(docker_execution_smoke, "LocalJobRunner", FakeRunner)

    result = docker_execution_smoke.run_case(
        "octave",
        mode="job",
        octave_image="custom-octave:latest",
    )

    assert result["ok"] is True
    assert captured["image"] == "custom-octave:latest"
    assert result["actual_image"] == "custom-octave:latest"
    assert result["image_matches"] is True
    assert result["runtime_metadata"]["backend"] == "docker"
    assert docker_execution_smoke.jobs_module.DOCKER_OCTAVE_EXECUTION_IMAGE == previous_image


def test_docker_execution_smoke_job_mode_rejects_non_docker_backend(monkeypatch):
    monkeypatch.setattr(docker_execution_smoke.jobs_module, "CODE_EXECUTION_BACKEND", "local")

    result = docker_execution_smoke.run_case("python", mode="job")

    assert result["ok"] is False
    assert result["job_status"] == "not_started"
    assert result["runtime_metadata"]["backend"] == "local"
    assert "CODE_EXECUTION_BACKEND=docker" in result["stderr"]
