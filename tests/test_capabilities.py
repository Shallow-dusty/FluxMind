from src.capabilities import (
    CodeExecutionRequest,
    CodeExecutionResult,
    GeneratedArtifact,
    ImageGenerationRequest,
)


def test_capability_dataclasses_are_provider_neutral():
    image_request = ImageGenerationRequest(prompt="Draw a PMSM observer block diagram")
    artifact = GeneratedArtifact(
        kind="image",
        uri="artifact://diagram.png",
        mime_type="image/png",
        metadata={"provider": "test"},
    )
    execution_request = CodeExecutionRequest(
        language="python",
        entrypoint="main.py",
        files={"main.py": "print('ok')"},
    )
    execution_result = CodeExecutionResult(
        exit_code=0,
        stdout="ok\n",
        stderr="",
        artifacts=[artifact],
    )

    assert image_request.style == "engineering-diagram"
    assert execution_request.timeout_s == 30
    assert execution_result.success is True
    assert execution_result.artifacts[0].uri == "artifact://diagram.png"
