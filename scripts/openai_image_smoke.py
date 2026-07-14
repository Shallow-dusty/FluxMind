#!/usr/bin/env python3
"""Run one configured OpenAI image-generation smoke through the job/artifact path."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.capabilities import ImageGenerationRequest  # noqa: E402
from src.config import IMAGE_PROVIDER_BACKEND, OPENAI_IMAGE_API_KEY, OPENAI_IMAGE_MODEL  # noqa: E402
from src.jobs import LocalJobRunner  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test configured OpenAI image generation.")
    parser.add_argument(
        "--prompt",
        default="Draw a publication-ready PMSM field-oriented control block diagram with current loop, inverter, motor, observer, and feedback signals.",
    )
    parser.add_argument(
        "--template",
        default="pmsm-control-loop",
        choices=["generic", "sliding-mode-observer", "pmsm-control-loop", "paper-figure-redraft"],
    )
    parser.add_argument("--size", default="1024x1024")
    parser.add_argument("--style", default="engineering-diagram")
    args = parser.parse_args()

    backend = (IMAGE_PROVIDER_BACKEND or "").strip().lower()
    if backend != "openai":
        print(f"IMAGE_PROVIDER_BACKEND must be openai for this smoke; current={IMAGE_PROVIDER_BACKEND or 'local-mock'}")
        return 2
    if not OPENAI_IMAGE_API_KEY:
        print("OPENAI_IMAGE_API_KEY or OPENAI_API_KEY is required for this smoke.")
        return 2

    job = LocalJobRunner().run_image_generation(
        ImageGenerationRequest(
            prompt=args.prompt,
            style=args.style,
            size=args.size,
            diagram_template=args.template,
        )
    )
    print(f"job_id={job.job_id}")
    print(f"status={job.status}")
    print(f"backend={IMAGE_PROVIDER_BACKEND}")
    print(f"model={OPENAI_IMAGE_MODEL}")
    if job.error:
        print(f"error={job.error}")
        return 1
    for artifact in job.artifacts:
        print(f"artifact={artifact.get('title')} mime={artifact.get('mime_type')}")
    return 0 if job.status == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
