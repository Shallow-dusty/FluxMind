import json

import pytest

from src.quality_readiness import (
    collect_quality_readiness,
    format_quality_readiness_markdown,
)


def test_quality_readiness_reports_current_source_quality_without_live_evidence():
    status = collect_quality_readiness(generated_at="2026-06-16T00:00:00+00:00")

    assert status["mode"] == "quality_readiness"
    assert status["local_foundation_ready"] is True
    assert status["small_group_ready"] is False
    assert status["community_ready"] is False
    assert status["live_evidence_included"] is False
    assert status["content_exported"] is False
    assert status["secrets_exported"] is False
    assert status["paths_exported"] is False
    assert status["metrics"]["answer_case_count"] >= 42
    assert status["metrics"]["retrieval_eval_question_count"] >= 107
    assert "small_group_live_retrieval_result_count_gap" in status["blockers"]["maturity"]
    assert "community_live_answer_result_count_gap" in status["blockers"]["maturity"]


def test_quality_readiness_merges_live_report_counts(tmp_path):
    live_report = tmp_path / "live-report.json"
    live_report.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "quality_maturity": {
                    "metrics": {
                        "live_retrieval_result_count": 107,
                        "live_answer_result_count": 0,
                    }
                },
                "summary": {
                    "live_retrieval": {"total": 107, "ok": 107, "failed": 0},
                    "live_answers": {"total": 0, "ok": 0, "failed": 0},
                },
            }
        ),
        encoding="utf-8",
    )

    status = collect_quality_readiness(
        generated_at="2026-06-16T00:00:00+00:00",
        live_report_paths=[live_report],
    )

    assert status["small_group_ready"] is True
    assert status["community_ready"] is False
    assert status["metrics"]["live_retrieval_result_count"] == 107
    assert status["reports"][0]["name"] == "live-report.json"
    assert status["reports"][0]["ok"] is True
    assert "small_group_live_retrieval_result_count_gap" not in status["blockers"]["maturity"]
    assert "community_live_answer_result_count_gap" in status["blockers"]["maturity"]


def test_quality_readiness_blocks_live_answer_quality_below_threshold(tmp_path):
    eval_file = tmp_path / "eval.json"
    eval_file.write_text(
        json.dumps(
            {
                "quality_gates": {
                    "minimum_live_answer_pass_rate": 1.0,
                    "minimum_average_live_answer_term_coverage": 0.75,
                    "minimum_live_retrieval_pass_rate": 1.0,
                },
                "quality_maturity_targets": [
                    {"id": "self_use", "required_metrics": {}},
                    {
                        "id": "small_group",
                        "required_metrics": {"live_retrieval_result_count": 2},
                    },
                    {
                        "id": "community",
                        "required_metrics": {
                            "live_retrieval_result_count": 2,
                            "live_answer_result_count": 2,
                        },
                    },
                ],
                "cases": [],
            }
        ),
        encoding="utf-8",
    )
    live_report = tmp_path / "live-report.json"
    live_report.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "summary": {
                    "live_retrieval": {"total": 2, "ok": 2, "failed": 0},
                    "live_answers": {"total": 2, "ok": 1, "failed": 1},
                },
                "results": {
                    "live_answers": [
                        {"case_id": "a", "ok": True, "answer_term_coverage": 1.0},
                        {"case_id": "b", "ok": False, "answer_term_coverage": 0.2},
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    status = collect_quality_readiness(
        eval_file=eval_file,
        generated_at="2026-06-16T00:00:00+00:00",
        live_report_paths=[live_report],
        project_root=tmp_path,
    )

    assert status["small_group_ready"] is True
    assert status["community_ready"] is False
    assert status["metrics"]["live_answer_pass_rate"] == 0.5
    assert status["metrics"]["average_live_answer_term_coverage"] == 0.6
    assert "community_live_answer_pass_rate_gap" in status["blockers"]["maturity"]
    assert "community_average_live_answer_term_coverage_gap" in status["blockers"]["maturity"]


def test_quality_readiness_accepts_live_answer_quality_at_threshold(tmp_path):
    eval_file = tmp_path / "eval.json"
    eval_file.write_text(
        json.dumps(
            {
                "quality_gates": {
                    "minimum_live_answer_pass_rate": 1.0,
                    "minimum_average_live_answer_term_coverage": 0.75,
                    "minimum_live_retrieval_pass_rate": 1.0,
                },
                "quality_maturity_targets": [
                    {"id": "self_use", "required_metrics": {}},
                    {
                        "id": "small_group",
                        "required_metrics": {"live_retrieval_result_count": 2},
                    },
                    {
                        "id": "community",
                        "required_metrics": {
                            "live_retrieval_result_count": 2,
                            "live_answer_result_count": 2,
                        },
                    },
                ],
                "cases": [],
            }
        ),
        encoding="utf-8",
    )
    live_report = tmp_path / "live-report.json"
    live_report.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "summary": {
                    "live_retrieval": {"total": 2, "ok": 2, "failed": 0},
                    "live_answers": {"total": 2, "ok": 2, "failed": 0},
                },
                "results": {
                    "live_answers": [
                        {"case_id": "a", "ok": True, "answer_term_coverage": 0.8},
                        {"case_id": "b", "ok": True, "answer_term_coverage": 0.9},
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    status = collect_quality_readiness(
        eval_file=eval_file,
        generated_at="2026-06-16T00:00:00+00:00",
        live_report_paths=[live_report],
        project_root=tmp_path,
    )

    assert status["small_group_ready"] is True
    assert status["community_ready"] is True
    assert status["metrics"]["live_retrieval_pass_rate"] == 1.0
    assert status["metrics"]["live_answer_pass_rate"] == 1.0
    assert status["metrics"]["average_live_answer_term_coverage"] == pytest.approx(0.85)
    assert status["blockers"]["maturity"] == []


def test_quality_readiness_blocks_unreadable_live_report(tmp_path):
    missing_report = tmp_path / "missing.json"

    status = collect_quality_readiness(
        generated_at="2026-06-16T00:00:00+00:00",
        live_report_paths=[missing_report],
    )

    assert status["local_foundation_ready"] is False
    assert status["reports"][0]["ok"] is False
    assert status["reports"][0]["reason"] == "unreadable"
    assert "live_report_unreadable" in status["blockers"]["maturity"]


def test_format_quality_readiness_markdown_omits_report_paths(tmp_path):
    live_report = tmp_path / "report.json"
    live_report.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "summary": {"live_retrieval": {"total": 50}, "live_answers": {"total": 1}},
            }
        ),
        encoding="utf-8",
    )
    status = collect_quality_readiness(
        generated_at="2026-06-16T00:00:00+00:00",
        live_report_paths=[live_report],
    )

    markdown = format_quality_readiness_markdown(status)

    assert "# FluxMind Quality Readiness" in markdown
    assert "Local foundation ready:" in markdown
    assert "Community ready:" in markdown
    assert str(tmp_path) not in markdown
    assert "hunter2" not in markdown
