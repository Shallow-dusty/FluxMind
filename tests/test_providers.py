from pathlib import Path

from src.capabilities import CodeExecutionRequest, ImageGenerationRequest
from src.providers import (
    LocalArtifactStore,
    LocalOctaveExecutionProvider,
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
    assert result.artifacts[1].kind == "text"
    assert Path(result.artifacts[1].uri.removeprefix("file://")).read_text(encoding="utf-8") == "done"


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
    assert result.artifacts[0].title == "result.txt"
    assert result.artifacts[0].metadata["runtime"] == "gnu-octave-local"
