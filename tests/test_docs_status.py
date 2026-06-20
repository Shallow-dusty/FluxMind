from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_repo_status_records_post_deployment_git_boundary():
    text = (PROJECT_ROOT / "docs" / "REPO_STATUS.md").read_text(encoding="utf-8")

    assert "Source/eval quality baseline   9b1cbc5 test: expand FluxMind community quality eval" in text
    assert "Current implementation commit  95f1760 test: add octave-aware code-output eval" in text
    assert "Current docs/health sync       e4da2e9 docs: document octave-aware eval status" in text
    assert "Last deployed source/eval baseline 9b1cbc5 test: expand FluxMind community quality eval" in text
    assert "Live verification follow-up    30-paper corpus and 107/107 live retrieval refreshed on 2026-06-17 02:37 CST" in text
    assert "Latest deploy follow-up        95f1760/e4da2e9 synced without restart and live-checked on 2026-06-17 02:59 CST" in text
    assert "`177dd4e` (`feat: gate quality readiness on live answer metrics`)" in text
    assert "`35338d2` (`docs: clarify live answer quality readiness`)" in text
    assert "`fa512df` (`fix: tolerate partial live quality result objects`)" in text
    assert "`95f1760`" in text
    assert "`e4da2e9` (`docs: document octave-aware eval status`)" in text
    assert "Octave-aware code-output fallback" in text
    assert "pmsm_current_decay" in text
    assert "structured runtime-unavailable diagnostic" in text
    assert "live answer count/pass-rate/term-coverage gates" in text
    assert "live_retrieval_pass_rate=1.0" in text
    assert "code_output_case_count=13" in text
    assert "live_answer_result_count=0" in text
    assert "local product registry source/docs/health sync deployed to" in text
    assert "`c41ea94` (`feat: add local product registry`)" in text
    assert "`efe2143` (`docs: document product quota guard`)" in text
    assert "`c130778` (`feat: add local product quota guard`)" in text
    assert "product quota guard source/docs/health sync deployed to" in text
    assert "`3c85999` (`docs: document local product RBAC guard`)" in text
    assert "`c7ecbf6` (`feat: add local product RBAC guard`)" in text
    assert "product RBAC guard source/docs/health sync deployed to" in text
    assert "`b05c28d` (`docs: document product registry management`)" in text
    assert "`645be5d` (`feat: add local product registry management`)" in text
    assert "product registry management source/docs/health sync deployed" in text
    assert "`517756f`" in text
    assert "(`docs: document object manifest verifier`)" in text
    assert "`45e4cc6`" in text
    assert "(`feat: verify object storage migration manifests`)" in text
    assert "object-storage migration manifest verifier source/docs/health sync" in text
    assert "local API-key registry source/docs/health sync deployed to" in text
    assert "`207ba7a` (`fix: extend remote health timeout`)" in text
    assert "`6ad6dbc` (`feat: add local API key registry`)" in text
    assert "product_registry_sqlite ok=true" in text
    assert "api_key_registry_sqlite ok=true" in text
    assert "Product registry    backend=none; available=false; workspaces=0; secrets_exported=false" in text
    assert "Product registry management installed=true; route ok; backend=none by default" in text
    assert "Object manifest     rehearsal_ok=true; objects=19; unique=18;" in text
    assert "Object verify       ok=true; checked=19; missing=0; mismatched=0; extra=0;" in text
    assert "Product RBAC guard  installed=true; enabled=false by default; admin metric present" in text
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
    assert "Latest deploy follow-up        45e4cc6/517756f synced" not in text
    assert "Current local app-code HEAD    b1212e2 feat: expose local activation admin surfaces" in text
    assert "ba7c243 feat: add provider quota guard and safe runtime events" in text
    assert "4ea219c fix: harden no-secret local projections" in text
    assert "1ebfde3 feat: add durable job-store migration manifests" in text
    assert "39ddaee feat: add local activation readiness tools" in text
    assert "committed locally in the stack above, not pushed" in text
    assert "not committed, pushed, or deployed" not in text


def test_roadmap_near_term_plan_starts_from_deployed_baseline():
    text = (PROJECT_ROOT / "docs" / "PLATFORM_AUDIT_AND_ROADMAP.md").read_text(encoding="utf-8")

    assert "Treat `9b1cbc5` as the current source/eval quality baseline" in text
    assert "Decide whether to push the current 36 local commits" not in text


def test_agent_bootstrap_docs_include_current_no_secret_commands():
    agents = (PROJECT_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    claude = (PROJECT_ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    docs_index = (PROJECT_ROOT / "docs" / "README.md").read_text(encoding="utf-8")
    required_snippets = [
        "scripts/product_activation_rehearsal.py --format markdown --require-activation",
        "scripts/share_link_registry.py status --format markdown",
        "scripts/collaboration_readiness.py --format markdown",
        "scripts/provider_runtime_rehearsal.py --format markdown --require-local-foundation",
        "scripts/activation_suite.py --format markdown --require-target local_foundation",
        "scripts/platform_migration_rehearsal.py --include-object-manifest --include-job-store-manifest",
        "scripts/platform_migration_rehearsal.py --verify-job-store-manifest",
        "durable job-store migration manifest",
        "per-target current/expected/gap",
    ]

    for snippet in required_snippets:
        assert snippet in agents
        assert snippet in claude
    assert "Project bootstrap         README.md, AGENTS.md, CLAUDE.md" in docs_index


def test_readme_bilingual_verification_commands_include_current_no_secret_surface():
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    required_commands = [
        "python scripts/product_activation_rehearsal.py --format markdown --require-activation",
        "python scripts/share_link_registry.py status --format markdown",
        "python scripts/collaboration_readiness.py --format markdown",
        "python scripts/provider_runtime_rehearsal.py --format markdown --require-local-foundation",
        "python scripts/activation_suite.py --format markdown --require-target local_foundation",
        "python scripts/platform_migration_rehearsal.py --include-object-manifest --include-job-store-manifest --output /tmp/fluxmind-object-and-job-rehearsal.json",
        "python scripts/platform_migration_rehearsal.py --verify-job-store-manifest /tmp/fluxmind-object-and-job-rehearsal.json --format markdown",
    ]

    for command in required_commands:
        assert readme.count(command) == 2
