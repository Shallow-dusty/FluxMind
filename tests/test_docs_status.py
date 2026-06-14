from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_repo_status_records_post_deployment_git_boundary():
    text = (PROJECT_ROOT / "docs" / "REPO_STATUS.md").read_text(encoding="utf-8")

    assert "Last clean local/origin state  391ac7f test: harden coverage and expand seed corpus" in text
    assert "Deployed source baseline       391ac7f test: harden coverage and expand seed corpus" in text
    assert "Deployment record follow-up    this docs refresh after 2026-06-15 04:23 live verification" in text
    assert "HEAD          a51a060" not in text
    assert "origin/main   a51a060" not in text
    assert "before the deployment-record follow-up" not in text


def test_roadmap_near_term_plan_starts_from_deployed_baseline():
    text = (PROJECT_ROOT / "docs" / "PLATFORM_AUDIT_AND_ROADMAP.md").read_text(encoding="utf-8")

    assert "Treat `391ac7f` as the current pushed/deployed no-key baseline" in text
    assert "Decide whether to push the current 36 local commits" not in text
