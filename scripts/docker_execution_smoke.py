#!/usr/bin/env python3
"""Smoke-test FluxMind Docker code execution providers.

This is an explicit operator tool, not an automatic startup check. It runs
small Python and/or Octave jobs through DockerExecutionProvider and verifies
that generated files are collected as artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capabilities import CodeExecutionRequest
from src.config import (
    ARTIFACTS_DIR,
    DOCKER_OCTAVE_EXECUTION_IMAGE,
    DOCKER_PYTHON_EXECUTION_IMAGE,
)
from src.providers import DockerExecutionProvider, LocalArtifactStore
import src.jobs as jobs_module
from src.jobs import LocalJobRunner, LocalJobStore


PYTHON_FILES = {
    "main.py": """from pathlib import Path

Path("result.txt").write_text("fluxmind python docker smoke\\n", encoding="utf-8")
print("python smoke ok")
""",
}

OCTAVE_FILES = {
    "main.m": """fid = fopen("result.txt", "w");
fprintf(fid, "fluxmind octave docker smoke\\n");
fclose(fid);
disp("octave smoke ok");
""",
}


def _case_request(language: str) -> CodeExecutionRequest:
    if language == "python":
        return CodeExecutionRequest(
            language="python",
            entrypoint="main.py",
            files=PYTHON_FILES,
            timeout_s=30,
            memory_mb=512,
        )
    if language == "octave":
        return CodeExecutionRequest(
            language="octave",
            entrypoint="main.m",
            files=OCTAVE_FILES,
            timeout_s=45,
            memory_mb=768,
        )
    raise ValueError(f"Unsupported language: {language}")


def _image_for_language(language: str, *, python_image: str | None, octave_image: str | None) -> str:
    if language == "octave":
        return (octave_image or "").strip() or DOCKER_OCTAVE_EXECUTION_IMAGE
    return (python_image or "").strip() or DOCKER_PYTHON_EXECUTION_IMAGE


def _artifact_summaries_from_provider_result(result) -> list[dict[str, Any]]:
    return [
        {
            "title": artifact.title,
            "kind": artifact.kind,
            "mime_type": artifact.mime_type,
            "uri": artifact.uri,
        }
        for artifact in result.artifacts
    ]


def _artifact_summaries_from_job(job) -> list[dict[str, Any]]:
    return [
        {
            "title": artifact.get("title"),
            "kind": artifact.get("kind"),
            "mime_type": artifact.get("mime_type"),
            "uri": artifact.get("uri"),
        }
        for artifact in job.artifacts
    ]


def _case_payload(
    *,
    language: str,
    image: str,
    mode: str,
    exit_code: int,
    stdout: str,
    stderr: str,
    artifact_summaries: list[dict[str, Any]],
    runtime_metadata: dict[str, Any],
    job_status: str | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    runtime_metadata = dict(runtime_metadata)
    runtime_metadata.setdefault(
        "backend",
        jobs_module.CODE_EXECUTION_BACKEND if mode == "job" else "provider-direct",
    )
    expected_artifact_found = any(
        artifact.get("title") == "result.txt" for artifact in artifact_summaries
    )
    actual_image = str(runtime_metadata.get("docker_image", ""))
    image_matches = actual_image == image
    ok = (
        exit_code == 0
        and expected_artifact_found
        and image_matches
        and job_status in {None, "succeeded"}
    )
    return {
        "language": language,
        "image": image,
        "actual_image": actual_image,
        "image_matches": image_matches,
        "mode": mode,
        "ok": ok,
        "exit_code": exit_code,
        "stdout": stdout,
        "stderr": stderr,
        "artifact_count": len(artifact_summaries),
        "expected_artifact": "result.txt",
        "expected_artifact_found": expected_artifact_found,
        "artifacts": artifact_summaries,
        "runtime_metadata": runtime_metadata,
        "job_status": job_status,
        "job_id": job_id,
    }


def _run_provider_case(
    language: str,
    *,
    image: str,
) -> dict[str, Any]:
    provider = DockerExecutionProvider(
        LocalArtifactStore(ARTIFACTS_DIR),
        image=image,
    )
    result = provider.run(_case_request(language))
    return _case_payload(
        language=language,
        image=image,
        mode="provider",
        exit_code=result.exit_code,
        stdout=result.stdout,
        stderr=result.stderr,
        artifact_summaries=_artifact_summaries_from_provider_result(result),
        runtime_metadata=result.runtime_metadata,
    )


def _run_job_case(
    language: str,
    *,
    image: str,
) -> dict[str, Any]:
    if jobs_module.CODE_EXECUTION_BACKEND != "docker":
        return _case_payload(
            language=language,
            image=image,
            mode="job",
            exit_code=2,
            stdout="",
            stderr=(
                "Job-mode Docker smoke requires CODE_EXECUTION_BACKEND=docker. "
                f"Current backend is {jobs_module.CODE_EXECUTION_BACKEND or 'local'}."
            ),
            artifact_summaries=[],
            runtime_metadata={"backend": jobs_module.CODE_EXECUTION_BACKEND or "local"},
            job_status="not_started",
        )
    if language == "octave":
        image_attr = "DOCKER_OCTAVE_EXECUTION_IMAGE"
    else:
        image_attr = "DOCKER_PYTHON_EXECUTION_IMAGE"
    previous_image = getattr(jobs_module, image_attr)
    setattr(jobs_module, image_attr, image)
    with tempfile.TemporaryDirectory(prefix="fluxmind-docker-job-smoke-") as tmp:
        try:
            runner = LocalJobRunner(
                LocalJobStore(Path(tmp) / "jobs.jsonl"),
                artifact_root=ARTIFACTS_DIR,
                record_runtime_events=False,
            )
            request = _case_request(language)
            if language == "octave":
                job = runner.run_local_octave(request)
            else:
                job = runner.run_local_python(request)
            result = job.result or {}
            return _case_payload(
                language=language,
                image=image,
                mode="job",
                exit_code=int(result.get("exit_code", 1)),
                stdout=str(result.get("stdout", "")),
                stderr=str(result.get("stderr", job.error.get("message", "") if job.error else "")),
                artifact_summaries=_artifact_summaries_from_job(job),
                runtime_metadata=dict(result.get("runtime_metadata", {})),
                job_status=job.status,
                job_id=job.job_id,
            )
        finally:
            setattr(jobs_module, image_attr, previous_image)


def run_case(
    language: str,
    *,
    python_image: str | None = None,
    octave_image: str | None = None,
    mode: str = "job",
) -> dict[str, Any]:
    image = _image_for_language(
        language,
        python_image=python_image,
        octave_image=octave_image,
    )
    if mode == "provider":
        return _run_provider_case(language, image=image)
    return _run_job_case(language, image=image)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--language",
        choices=["all", "python", "octave"],
        default="all",
        help="Docker execution language to smoke-test.",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format.",
    )
    parser.add_argument(
        "--mode",
        choices=["job", "provider"],
        default="job",
        help="Run through the job boundary or call DockerExecutionProvider directly.",
    )
    parser.add_argument(
        "--python-image",
        default=None,
        help="Override the Python Docker image for this smoke run.",
    )
    parser.add_argument(
        "--octave-image",
        default=None,
        help="Override the Octave Docker image for this smoke run.",
    )
    args = parser.parse_args()

    languages = ["python", "octave"] if args.language == "all" else [args.language]
    results = [
        run_case(
            language,
            python_image=args.python_image,
            octave_image=args.octave_image,
            mode=args.mode,
        )
        for language in languages
    ]
    ok = all(result["ok"] for result in results)

    if args.format == "json":
        print(json.dumps({"ok": ok, "results": results}, ensure_ascii=False, indent=2))
    else:
        print(f"Docker execution smoke: {'ok' if ok else 'failed'}")
        for result in results:
            print(
                f"- {result['language']}: mode={result['mode']} ok={result['ok']} "
                f"exit_code={result['exit_code']} artifacts={result['artifact_count']} "
                f"result_txt={result['expected_artifact_found']} image_matches={result['image_matches']} "
                f"image={result['image']} actual_image={result['actual_image']} "
                f"backend={result['runtime_metadata'].get('backend', '')}"
            )
            if not result["ok"]:
                stderr = str(result["stderr"]).strip()
                if stderr:
                    print(f"  stderr: {stderr}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
