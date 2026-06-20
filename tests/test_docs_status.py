from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_repo_status_records_post_deployment_git_boundary():
    text = (PROJECT_ROOT / "docs" / "REPO_STATUS.md").read_text(encoding="utf-8")

    assert "Source/eval quality baseline   9b1cbc5 test: expand FluxMind community quality eval" in text
    assert "Current implementation commit  042e6d0 fix: redact API key public metadata" in text
    assert "Current docs/health sync       docs: refresh git and documentation drift status (this commit)" in text
    assert "by the twenty-five local commits below" in text
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
    assert "Current local app-code HEAD    042e6d0 fix: redact API key public metadata" in text
    assert "docs: refresh git and documentation drift status (this commit)" in text
    assert "042e6d0 fix: redact API key public metadata" in text
    assert "c7b6d9d docs: refresh git and drift status" in text
    assert "6066547 docs: record runtime event redaction audit" in text
    assert "1173ea8 fix: redact runtime event metadata values" in text
    assert "85eb2b5 docs: record execution input audit status" in text
    assert "69bb9e7 fix: handle execution input path conflicts" in text
    assert "1f97b7b docs: record product registry audit status" in text
    assert "12f4205 fix: guard product registry orphan writes" in text
    assert "05fae15 docs: record artifact path audit status" in text
    assert "51fee7e fix: harden local artifact path resolution" in text
    assert "f2d2da1 docs: record corpus metadata audit status" in text
    assert "5065418 fix: preserve same-name corpus metadata" in text
    assert "830d05d docs: record job lease audit status" in text
    assert "bae5f88 fix: guard terminal job lease release" in text
    assert "fac2c6b docs: record share-link event evidence audit" in text
    assert "ea8a7a2 fix: preserve share-link workspace event evidence" in text
    assert "3ae6842 docs: refresh share-link audit status" in text
    assert "c56c285 fix: redact share-link workspace identifiers" in text
    assert "e93dba5 docs: refresh FluxMind activation status" in text
    assert "ba7c243 feat: add provider quota guard and safe runtime events" in text
    assert "4ea219c fix: harden no-secret local projections" in text
    assert "1ebfde3 feat: add durable job-store migration manifests" in text
    assert "39ddaee feat: add local activation readiness tools" in text
    assert "b1212e2 feat: expose local activation admin surfaces" in text
    assert "API-key public metadata projection follow-up on 2026-06-20 16:16 CST" in text
    assert "create/list/verify/revoke now removes raw owner IDs" in text
    assert "presence booleans and short fingerprints" in text
    assert "Git/documentation drift refresh on 2026-06-20 20:10 CST" in text
    assert "share tokens/URLs exported=false" in text
    assert "no OpenAPI no-secret snapshot drift" in text
    assert "Git/docs drift refresh on 2026-06-20 16:08 CST" in text
    assert "pass, 17 docs/feature-audit/" in text
    assert "No production deployment was" in text
    assert "Runtime event metadata-value redaction follow-up on 2026-06-20 16:01 CST" in text
    assert "pass, 616 tests, 2 known warnings" in text
    assert "sensitive string values under otherwise safe keys" in text
    assert "hidden values cannot be rediscovered through `q`" in text
    assert "Execution input materialization follow-up on 2026-06-20 15:50 CST" in text
    assert "pass, 614 tests, 2 known warnings" in text
    assert "execution input materialization conflict handling" in text
    assert "regular-file entrypoint guards" in text
    assert "without exposing the temporary" in text
    assert "normal failed" in text
    assert "Product registry referential-integrity follow-up on 2026-06-20 15:39 CST" in text
    assert "pass, 610 tests, 2 known warnings" in text
    assert "The follow-up hardens the local product registry ledger against orphan writes." in text
    assert "workspace, while usage and quota-decision writes also require an active product" in text
    assert "unsafe member user IDs are not echoed by the CLI" in text
    assert "Artifact path-resolution follow-up on 2026-06-20 15:26 CST" in text
    assert "pass, 608 tests, 2 known warnings" in text
    assert "rejects nonlocal file artifact URIs" in text
    assert "canonical `..` aliases" in text
    assert "Corpus same-name metadata follow-up on 2026-06-20 15:18 CST" in text
    assert "pass, 606 tests, 2 known warnings" in text
    assert "source-path-specific entries" in text
    assert "same-name library/upload PDFs" in text
    assert "Job lease-release follow-up on 2026-06-20 15:05 CST" in text
    assert "completed-job worker provenance" in text
    assert "pass, 605 tests, 2 known warnings" in text
    assert "fingerprint=15bdfa2ae5ec34f1d0045c38b7137cf2b31a27857b1571a035a8efc12d61d18c" in text
    assert "pass, ok=true, diff_count=0 against the" in text
    assert "no storage-schema drift" in text
    assert "Share-link no-secret follow-up on 2026-06-20 12:22 CST" in text
    assert "Share-link event-evidence follow-up on 2026-06-20 15:00 CST" in text
    assert "workspace_present" in text
    assert "workspace_fingerprint" in text
    assert "product_workspace_present=true" in text
    assert "corrupted SQLite status fallback" in text
    assert "raw `workspace_id`" in text
    assert "committed locally in the stack above, not pushed" in text
    assert "not committed, pushed, or deployed" not in text


def test_roadmap_near_term_plan_starts_from_deployed_baseline():
    text = (PROJECT_ROOT / "docs" / "PLATFORM_AUDIT_AND_ROADMAP.md").read_text(encoding="utf-8")

    assert "Treat `9b1cbc5` as the current source/eval quality baseline" in text
    assert "Decide whether to push the current 36 local commits" not in text
    assert "pass, 616 tests, 2 known warnings" in text
    assert "execution-input-materialization" in text
    assert "runtime-event-metadata-value-redaction" in text


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
