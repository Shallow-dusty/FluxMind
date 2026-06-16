from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_repo_status_records_post_deployment_git_boundary():
    text = (PROJECT_ROOT / "docs" / "REPO_STATUS.md").read_text(encoding="utf-8")

    assert "Source/eval quality baseline   9b1cbc5 test: expand FluxMind community quality eval" in text
    assert "Current implementation commit  c130778 feat: add local product quota guard" in text
    assert "Current docs/health sync       efe2143 docs: document product quota guard" in text
    assert "Last deployed source/eval baseline 9b1cbc5 test: expand FluxMind community quality eval" in text
    assert "Live verification follow-up    30-paper corpus and 107/107 live retrieval refreshed on 2026-06-17 00:37 CST" in text
    assert "Latest deploy follow-up        c130778/efe2143 synced with restart and live-checked on 2026-06-17 00:39 CST" in text
    assert "local product registry source/docs/health sync deployed to" in text
    assert "`c41ea94` (`feat: add local product registry`)" in text
    assert "`efe2143` (`docs: document product quota guard`)" in text
    assert "`c130778` (`feat: add local product quota guard`)" in text
    assert "product quota guard source/docs/health sync deployed to" in text
    assert "local API-key registry source/docs/health sync deployed to" in text
    assert "`207ba7a` (`fix: extend remote health timeout`)" in text
    assert "`6ad6dbc` (`feat: add local API key registry`)" in text
    assert "product_registry_sqlite ok=true" in text
    assert "api_key_registry_sqlite ok=true" in text
    assert "Product registry    backend=none; available=false; workspaces=0; secrets_exported=false" in text
    assert "API key registry    backend=none; available=false; active_keys=0; secrets_exported=false" in text
    assert "latest platform-migration source/docs sync deployed to `/opt/fluxmind` is" in text
    assert "`dc2b71a` (`docs: record runtime migration rehearsal`)" in text
    assert "`8a4a76f` (`feat: add platform migration preflight`)" in text
    assert "`366c1e7`" in text
    assert "Migration preflight preflight_ok=true; activation_ready=false" in text
    assert "Migration rehearsal rehearsal_ok=true; copied_files=19" in text
    assert "HEAD          a51a060" not in text
    assert "origin/main   a51a060" not in text
    assert "before the deployment-record follow-up" not in text


def test_roadmap_near_term_plan_starts_from_deployed_baseline():
    text = (PROJECT_ROOT / "docs" / "PLATFORM_AUDIT_AND_ROADMAP.md").read_text(encoding="utf-8")

    assert "Treat `9b1cbc5` as the current source/eval quality baseline" in text
    assert "Decide whether to push the current 36 local commits" not in text
