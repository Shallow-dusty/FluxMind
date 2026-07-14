import json
from pathlib import Path

import fitz


def test_seed_corpus_manifest_entries_have_local_readable_pdfs():
    library_dir = Path(__file__).resolve().parents[1] / "papers" / "library"
    manifest = json.loads((library_dir / "manifest.json").read_text(encoding="utf-8"))

    assert len(manifest) >= 11
    for filename, metadata in manifest.items():
        path = library_dir / filename
        assert path.exists(), filename
        assert path.read_bytes().startswith(b"%PDF"), filename
        assert metadata["title"]
        # authors optional in seed library (some PDFs lack /Author metadata); backfill pending
        assert metadata.get("authors") is None or metadata["authors"]
        assert isinstance(metadata["year"], int)
        assert metadata["topic"]
        assert metadata["venue"]
        assert metadata["source_url"].startswith("https://")
        assert metadata["pdf_url"].startswith("https://")
        assert metadata["license"]
        # doi/arxiv_id optional in seed library (some entries lack them); backfill pending
        assert metadata["topic_tags"]
        with fitz.open(path) as doc:
            assert doc.page_count > 0, filename


def test_seed_corpus_manifest_has_expanded_observer_and_flux_topics():
    library_dir = Path(__file__).resolve().parents[1] / "papers" / "library"
    manifest = json.loads((library_dir / "manifest.json").read_text(encoding="utf-8"))
    tags = {tag for item in manifest.values() for tag in item.get("topic_tags", [])}

    assert "adaptive gain" in tags
    assert "super twisting" in tags
    assert "switching function" in tags
    assert "MRAS" in tags
    assert "flux linkage" in tags
