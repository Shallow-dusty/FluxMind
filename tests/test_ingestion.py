from pathlib import Path

import pytest

from src import ingestion


def test_safe_pdf_name_preserves_unicode_and_strips_reserved_chars():
    assert ingestion._safe_pdf_name(" 磁链 观测<>:\"|?*.PDF ") == "磁链-观测.pdf"


def test_safe_pdf_name_uses_basename_to_prevent_path_traversal():
    assert ingestion._safe_pdf_name("../../磁链 观测.pdf") == "磁链-观测.pdf"


def test_safe_pdf_name_rejects_non_pdf():
    with pytest.raises(ValueError, match="Only PDF"):
        ingestion._safe_pdf_name("notes.txt")


def test_resolve_unique_path_adds_suffix(tmp_path: Path):
    target = tmp_path / "paper.pdf"
    target.write_text("already here", encoding="utf-8")

    assert ingestion._resolve_unique_path(target) == tmp_path / "paper-1.pdf"
