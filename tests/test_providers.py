from pathlib import Path

from src.capabilities import CodeExecutionRequest, ImageGenerationRequest
from src.providers import (
    LocalArtifactStore,
    LocalPythonExecutionProvider,
    MockImageGenerationProvider,
)


def test_mock_image_provider_writes_svg_artifact(tmp_path: Path):
    provider = MockImageGenerationProvider(LocalArtifactStore(tmp_path))

    artifact = provider.generate(ImageGenerationRequest(prompt="SMC observer"))

    assert artifact.kind == "image"
    assert artifact.mime_type == "image/svg+xml"
    assert artifact.metadata["provider"] == "local"
    assert Path(artifact.uri.removeprefix("file://")).exists()


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


def test_local_python_execution_provider_rejects_non_python():
    provider = LocalPythonExecutionProvider()

    result = provider.run(
        CodeExecutionRequest(language="octave", entrypoint="main.m", files={})
    )

    assert result.success is False
    assert "Unsupported local language" in result.stderr
