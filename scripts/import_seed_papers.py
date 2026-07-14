#!/usr/bin/env python3
"""Import curated open-access seed papers into the bundled library."""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIBRARY_DIR = PROJECT_ROOT / "papers" / "library"
MANIFEST_FILE = LIBRARY_DIR / "manifest.json"
USER_AGENT = "FluxMind seed-paper importer/1.0"


@dataclass(frozen=True)
class SeedPaper:
    filename: str
    title: str
    year: int
    topic: str
    topic_tags: list[str]
    source_url: str
    pdf_url: str
    venue: str = ""
    authors: str = ""
    doi: str = ""
    arxiv_id: str = ""
    license: str = ""

    def manifest_entry(self) -> dict[str, object]:
        payload = asdict(self)
        payload.pop("filename", None)
        return {key: value for key, value in payload.items() if value not in ("", [], None)}


SEED_PAPERS: tuple[SeedPaper, ...] = (
    SeedPaper(
        filename="zhang-2023-arxiv-smo-vector-control-pmsm.pdf",
        title="Simulation of Non-inductive Vector Control of Permanent Magnet Synchronous Motor Based on Sliding Mode Observer",
        authors="Caiyue Zhang; Zipin Liu; Bowen Xu",
        year=2023,
        topic="PMSM SMO simulation",
        topic_tags=["PMSM", "sliding mode observer", "sensorless control", "vector control", "Simulink"],
        venue="arXiv",
        arxiv_id="2305.04046",
        source_url="https://arxiv.org/abs/2305.04046",
        pdf_url="https://arxiv.org/pdf/2305.04046",
        license="arXiv open access",
    ),
    SeedPaper(
        filename="aremu-2025-arxiv-smc-pmsm-benchmark.pdf",
        title="Sliding-Mode Control Strategies for PMSM: Benchmarking and Comparative Simulation Study",
        authors="Mubarak Badamasi Aremu; Abdullah Ajasa; Ali Nasir",
        year=2025,
        topic="PMSM SMC benchmark",
        topic_tags=["PMSM", "sliding mode control", "benchmark", "super twisting", "adaptive SMC"],
        venue="arXiv",
        arxiv_id="2512.06603",
        source_url="https://arxiv.org/abs/2512.06603",
        pdf_url="https://arxiv.org/pdf/2512.06603",
        license="arXiv open access",
    ),
    SeedPaper(
        filename="microchip-2024-an4398-pmsm-smo-foc.pdf",
        title="Sensorless Field Oriented Control for a Permanent Magnet Synchronous Motor Using Sliding Mode Observer",
        year=2024,
        topic="PMSM SMO application note",
        topic_tags=["PMSM", "field oriented control", "sliding mode observer", "application note"],
        venue="Microchip Application Note AN4398",
        source_url="https://www.microchip.com/",
        pdf_url="https://ww1.microchip.com/downloads/aemDocuments/documents/MCU32/ApplicationNotes/ApplicationNotes/AN4398-Sensorless-Field-Oriented-Control-for-a-Permanent-Magnet-Synchronous-Motor-Using-Sliding-Mode-Observer-DS00004398.pdf",
        license="vendor application note",
    ),
    SeedPaper(
        filename="diva-2016-pmsm-sensorless-mras-smo.pdf",
        title="Sensorless Control of PMSM",
        year=2016,
        topic="PMSM MRAS and SMO",
        topic_tags=["PMSM", "sensorless control", "MRAS", "sliding mode observer"],
        venue="DiVA thesis",
        source_url="https://www.diva-portal.org/smash/record.jsf?pid=diva2:1032918",
        pdf_url="https://www.diva-portal.org/smash/get/diva2%3A1032918/FULLTEXT01.pdf",
        license="open access thesis",
    ),
    SeedPaper(
        filename="applsci-2025-smo-air-compressor-pmsm.pdf",
        title="Sliding Mode Observer-Based Sensorless Control Strategy for Permanent Magnet Synchronous Motor Drives",
        year=2025,
        topic="PMSM SMO compressor drive",
        topic_tags=["PMSM", "sliding mode observer", "sensorless control", "air compressor"],
        venue="Applied Sciences",
        source_url="https://www.mdpi.com/2076-3417/15/20/11206",
        pdf_url="https://res.mdpi.com/d_attachment/applsci/applsci-15-11206/article_deploy/applsci-15-11206.pdf",
        license="CC BY",
    ),
    SeedPaper(
        filename="applsci-2026-fuzzy-smc-sensorless-pmsm.pdf",
        title="Sensorless Control of PMSM Based on Fuzzy Sliding Mode Control",
        year=2026,
        topic="fuzzy SMC sensorless PMSM",
        topic_tags=["PMSM", "sensorless control", "fuzzy sliding mode", "sliding mode observer"],
        venue="Applied Sciences",
        source_url="https://www.mdpi.com/2076-3417/16/5/2544",
        pdf_url="https://res.mdpi.com/d_attachment/applsci/applsci-16-02544/article_deploy/applsci-16-02544.pdf",
        license="CC BY",
    ),
    SeedPaper(
        filename="sensors-2025-dynamic-self-adjusting-pmsm-smo.pdf",
        title="A Dynamic Self-Adjusting System for Permanent Magnet Synchronous Motor Sensorless Control",
        year=2025,
        topic="PMSM adaptive SMO",
        topic_tags=["PMSM", "sliding mode observer", "sensorless control", "dynamic self-adjusting"],
        venue="Sensors",
        source_url="https://www.mdpi.com/1424-8220/25/12/3623",
        pdf_url="https://res.mdpi.com/d_attachment/sensors/sensors-25-03623/article_deploy/sensors-25-03623.pdf",
        license="CC BY",
    ),
    SeedPaper(
        filename="symmetry-2025-adaptive-fractional-smc-pmsm.pdf",
        title="PMSM Speed Control Based on Improved Adaptive Fractional-Order Sliding Mode Control",
        year=2025,
        topic="fractional-order SMC",
        topic_tags=["PMSM", "fractional-order sliding mode", "disturbance observer", "speed control"],
        venue="Symmetry",
        source_url="https://www.mdpi.com/2073-8994/17/5/736",
        pdf_url="https://res.mdpi.com/d_attachment/symmetry/symmetry-17-00736/article_deploy/symmetry-17-00736.pdf",
        license="CC BY",
    ),
    SeedPaper(
        filename="symmetry-2025-symmetric-pmsm-smc.pdf",
        title="Sliding Mode Control of Symmetric Permanent Magnet Synchronous Motors",
        year=2025,
        topic="symmetric PMSM SMC",
        topic_tags=["PMSM", "sliding mode control", "symmetric motor", "speed control"],
        venue="Symmetry",
        source_url="https://www.mdpi.com/2073-8994/17/12/2057",
        pdf_url="https://res.mdpi.com/d_attachment/symmetry/symmetry-17-02057/article_deploy/symmetry-17-02057.pdf",
        license="CC BY",
    ),
    SeedPaper(
        filename="fractalfract-2024-fractional-super-twisting-pmsm.pdf",
        title="Robust Speed Control of Permanent Magnet Synchronous Motor Using Variable-Gain Fractional-Order Super-Twisting Sliding Mode Control",
        year=2024,
        topic="fractional super-twisting SMC",
        topic_tags=["PMSM", "super twisting", "fractional-order control", "sliding-mode disturbance observer"],
        venue="Fractal and Fractional",
        source_url="https://www.mdpi.com/2504-3110/8/7/368",
        pdf_url="https://res.mdpi.com/d_attachment/fractalfract/fractalfract-08-00368/article_deploy/fractalfract-08-00368.pdf",
        license="CC BY",
    ),
    SeedPaper(
        filename="electronics-2025-improved-back-emf-observer-pmsm.pdf",
        title="Robust Sensorless PMSM Control with Improved Back-EMF Observer",
        year=2025,
        topic="back-EMF observer",
        topic_tags=["PMSM", "sensorless control", "back-EMF observer", "sliding mode observer"],
        venue="Electronics",
        source_url="https://www.mdpi.com/2079-9292/14/7/1238",
        pdf_url="https://res.mdpi.com/d_attachment/electronics/electronics-14-01238/article_deploy/electronics-14-01238.pdf",
        license="CC BY",
    ),
    SeedPaper(
        filename="energies-2025-position-sensorless-control-pmsm.pdf",
        title="Position Sensorless Control of Permanent Magnet Synchronous Motor",
        year=2025,
        topic="PMSM sensorless control",
        topic_tags=["PMSM", "position sensorless control", "observer", "flux estimation"],
        venue="Energies",
        source_url="https://www.mdpi.com/1996-1073/18/10/2531",
        pdf_url="https://res.mdpi.com/d_attachment/energies/energies-18-02531/article_deploy/energies-18-02531.pdf",
        license="CC BY",
    ),
    SeedPaper(
        filename="energies-2025-model-free-adaptive-fuzzy-smo-pmsm.pdf",
        title="Model-Free Adaptive Fuzzy Sliding-Mode Observer Control for PMSM",
        year=2025,
        topic="model-free fuzzy SMO",
        topic_tags=["PMSM", "model-free adaptive control", "fuzzy control", "sliding-mode observer"],
        venue="Energies",
        source_url="https://www.mdpi.com/1996-1073/18/8/1877",
        pdf_url="https://res.mdpi.com/d_attachment/energies/energies-18-01877/article_deploy/energies-18-01877.pdf",
        license="CC BY",
    ),
    SeedPaper(
        filename="machines-2025-current-observation-smc-pmsm.pdf",
        title="A Real-Time Demanded Current Observation-Based Sliding Mode Control Method for PMSM",
        year=2025,
        topic="current-observation SMC",
        topic_tags=["PMSM", "sliding mode control", "current observation", "real-time control"],
        venue="Machines",
        source_url="https://www.mdpi.com/2075-1702/13/2/146",
        pdf_url="https://res.mdpi.com/d_attachment/machines/machines-13-00146/article_deploy/machines-13-00146.pdf",
        license="CC BY",
    ),
    SeedPaper(
        filename="electronics-2024-smc-model-predictive-speed-pmsm.pdf",
        title="Sliding Mode Speed Control for PMSM Based on Model Predictive Control",
        year=2024,
        topic="SMC model predictive speed control",
        topic_tags=["PMSM", "sliding mode control", "model predictive control", "disturbance observer"],
        venue="Electronics",
        source_url="https://www.mdpi.com/2079-9292/13/13/2561",
        pdf_url="https://res.mdpi.com/d_attachment/electronics/electronics-13-02561/article_deploy/electronics-13-02561.pdf",
        license="CC BY",
    ),
    SeedPaper(
        filename="applsci-2022-super-twisting-svr-dob-pmsm.pdf",
        title="Super-Twisting Sliding Mode Control with SVR Disturbance Observer for PMSM Speed Regulation",
        authors="Ahyeong Choi; Hyunchang Kim; Mingyuan Hu; Youngjae Kim; Hyeongki Ahn; Kwanho You",
        year=2022,
        topic="super-twisting SMC SVR DOB",
        topic_tags=["PMSM", "super twisting", "disturbance observer", "SVR", "speed regulation"],
        venue="Applied Sciences",
        doi="10.3390/app122110749",
        source_url="https://www.mdpi.com/2076-3417/12/21/10749",
        pdf_url="https://res.mdpi.com/d_attachment/applsci/applsci-12-10749/article_deploy/applsci-12-10749.pdf",
        license="CC BY",
    ),
    SeedPaper(
        filename="energies-2022-synergetic-smc-pmsm.pdf",
        title="Improvement of PMSM Sensorless Control Based on Synergetic and Sliding Mode Controllers",
        year=2022,
        topic="synergetic and SMC control",
        topic_tags=["PMSM", "sliding mode control", "synergetic control", "sensorless control"],
        venue="Energies",
        source_url="https://www.mdpi.com/1996-1073/15/6/2208",
        pdf_url="https://res.mdpi.com/d_attachment/energies/energies-15-02208/article_deploy/energies-15-02208.pdf",
        license="CC BY",
    ),
    SeedPaper(
        filename="mathematics-2024-fast-variable-speed-smc-pmsm.pdf",
        title="Speed Control for PMSM with Fast Variable-Speed Sliding Mode Control via High-Gain Disturbance Observer",
        authors="Hengqiang Wang; Guangming Zhang; Xiaojun Liu",
        year=2024,
        topic="fast variable-speed SMC",
        topic_tags=["PMSM", "sliding mode control", "high-gain disturbance observer", "speed control"],
        venue="Mathematics",
        doi="10.3390/math12132036",
        source_url="https://www.mdpi.com/2227-7390/12/13/2036",
        pdf_url="https://res.mdpi.com/d_attachment/mathematics/mathematics-12-02036/article_deploy/mathematics-12-02036.pdf",
        license="CC BY",
    ),
    SeedPaper(
        filename="applsci-2022-backstepping-smc-pmsm.pdf",
        title="Backstepping Sliding Mode Control of a Permanent Magnet Synchronous Motor",
        year=2022,
        topic="backstepping SMC",
        topic_tags=["PMSM", "backstepping", "sliding mode control", "disturbance compensation"],
        venue="Applied Sciences",
        source_url="https://www.mdpi.com/2076-3417/12/21/11225",
        pdf_url="https://res.mdpi.com/d_attachment/applsci/applsci-12-11225/article_deploy/applsci-12-11225.pdf",
        license="CC BY",
    ),
    SeedPaper(
        filename="mathematics-2024-surface-mount-pmsm-sensorless.pdf",
        title="Sensorless Control of Surface-Mount Permanent-Magnet Synchronous Motors",
        year=2024,
        topic="surface-mount PMSM sensorless control",
        topic_tags=["SPMSM", "sensorless control", "sliding mode observer", "control implementation"],
        venue="Mathematics",
        source_url="https://www.mdpi.com/2227-7390/12/13/2029",
        pdf_url="https://res.mdpi.com/d_attachment/mathematics/mathematics-12-02029/article_deploy/mathematics-12-02029.pdf",
        license="CC BY",
    ),
    SeedPaper(
        filename="processes-2024-high-speed-domain-pmsm-sensorless.pdf",
        title="Sensorless Position Control in High-Speed Domain of PMSM Based on an Improved Sliding Mode Observer",
        year=2024,
        topic="high-speed PMSM sensorless control",
        topic_tags=["PMSM", "high-speed domain", "sliding mode observer", "sensorless control"],
        venue="Processes",
        source_url="https://www.mdpi.com/2227-9717/12/11/2581",
        pdf_url="https://res.mdpi.com/d_attachment/processes/processes-12-02581/article_deploy/processes-12-02581.pdf",
        license="CC BY",
    ),
    SeedPaper(
        filename="sensors-2025-ultra-high-speed-pr-observer-pmsm.pdf",
        title="Sensorless Control of Ultra-High-Speed PMSM via Improved PR Observer",
        year=2025,
        topic="ultra-high-speed PMSM observer",
        topic_tags=["PMSM", "ultra-high-speed", "sensorless control", "observer"],
        venue="Sensors",
        source_url="https://www.mdpi.com/1424-8220/25/5/1290",
        pdf_url="https://res.mdpi.com/d_attachment/sensors/sensors-25-01290/article_deploy/sensors-25-01290.pdf",
        license="CC BY",
    ),
)


def load_manifest() -> dict[str, dict[str, object]]:
    if not MANIFEST_FILE.exists():
        return {}
    return json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))


def write_manifest(manifest: dict[str, dict[str, object]]) -> None:
    LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=LIBRARY_DIR,
            prefix=".manifest.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(dict(sorted(manifest.items())), handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        temp_path.replace(MANIFEST_FILE)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def download_pdf(url: str, *, timeout_s: int) -> bytes:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout_s) as response:
        content = response.read()
    if not content.lstrip().startswith(b"%PDF"):
        raise ValueError("downloaded content is not a PDF")
    return content


def write_pdf(path: Path, content: bytes) -> None:
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(content)
        temp_path.replace(path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def current_pdf_count() -> int:
    if not LIBRARY_DIR.exists():
        return 0
    return len([path for path in LIBRARY_DIR.glob("*.pdf") if path.is_file()])


def import_papers(*, dry_run: bool, limit: int | None, timeout_s: int) -> tuple[int, list[str]]:
    LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    downloaded = 0
    failures: list[str] = []
    for paper in SEED_PAPERS:
        target = LIBRARY_DIR / paper.filename
        if target.exists():
            manifest[paper.filename] = paper.manifest_entry()
            continue
        if limit is not None and downloaded >= limit:
            continue
        if dry_run:
            print(f"missing {paper.filename}")
            continue
        try:
            content = download_pdf(paper.pdf_url, timeout_s=timeout_s)
            write_pdf(target, content)
            manifest[paper.filename] = paper.manifest_entry()
            downloaded += 1
            print(f"downloaded {paper.filename}")
        except (HTTPError, URLError, TimeoutError, ValueError, OSError) as exc:
            failures.append(f"{paper.filename}: {exc}")
            print(f"failed {paper.filename}: {exc}")
    if not dry_run:
        write_manifest(manifest)
    return downloaded, failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Import curated open-access seed papers.")
    parser.add_argument("--dry-run", action="store_true", help="List missing seed papers without downloading.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of new PDFs to download.")
    parser.add_argument("--require-count", type=int, default=0, help="Fail if library PDF count is below this number.")
    parser.add_argument("--timeout-s", type=int, default=45, help="Per-PDF download timeout.")
    args = parser.parse_args()

    before = current_pdf_count()
    downloaded, failures = import_papers(dry_run=args.dry_run, limit=args.limit, timeout_s=args.timeout_s)
    after = current_pdf_count()
    print(f"library_pdfs_before={before}")
    print(f"library_pdfs_after={after}")
    print(f"downloaded={downloaded}")
    print(f"failed={len(failures)}")
    if failures:
        print("failures:")
        for failure in failures:
            print(f"- {failure}")
    if args.require_count and after < args.require_count:
        print(f"required_count={args.require_count}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
