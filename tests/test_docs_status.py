from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_repo_status_records_post_deployment_git_boundary():
    text = (PROJECT_ROOT / "docs" / "REPO_STATUS.md").read_text(encoding="utf-8")

    assert "Verified repo baseline         d1e5326 test: recalibrate live retrieval baseline" in text
    assert "Last deployed source/eval baseline d1e5326 test: recalibrate live retrieval baseline" in text
    assert "Live verification follow-up    18-paper corpus rebuild and 86/86 live retrieval refreshed on 2026-06-15 14:29 CST" in text
    assert "HEAD          a51a060" not in text
    assert "origin/main   a51a060" not in text
    assert "before the deployment-record follow-up" not in text


def test_roadmap_near_term_plan_starts_from_deployed_baseline():
    text = (PROJECT_ROOT / "docs" / "PLATFORM_AUDIT_AND_ROADMAP.md").read_text(encoding="utf-8")

    assert "Treat `d1e5326` as the current pushed/deployed source/eval baseline" in text
    assert "Decide whether to push the current 36 local commits" not in text
