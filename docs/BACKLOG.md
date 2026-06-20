# FluxMind Implementation Backlog

Last updated: 2026-06-20

For reading order and document ownership, see `docs/README.md`. Current git and
verification state is tracked in `docs/REPO_STATUS.md`.

This backlog turns the platform roadmap into concrete implementation packages.
It is intentionally ordered by dependency, not by excitement.

## Completion Snapshot

Confirmed on: 2026-06-16

The current completed scope is the no-key/local platform baseline:

```text
Work package          Confirmed state
--------------------  -------------------------------------------------------
WP0 production guard  complete for current deployed baseline; remains ongoing ops
WP1 RAG quality       complete for offline/recorded/live-retrieval no-key gates
WP2 corpus/storage    complete for local JSON/SQLite/filesystem baseline
WP3 job system        complete for local durable worker bridge and lease model
WP4 image/diagrams    complete for provider-neutral no-key SVG/artifact flow
                     plus external image-provider readiness blockers
WP5 code execution    complete for local Python/Octave-compatible dev providers
                     plus hosted execution/MATLAB readiness blockers
WP6 product shell     complete for local no-secret admin/reporting foundation
                     plus product/provider-readiness preflights
```

Current hardening progress through 2026-06-20: the automated suite has 602
passing tests, the repository has a coverage command/gate with 89% total branch
coverage over `api`, `scripts`, and `src`, and the curated seed library has been
expanded to 30 open-access papers. Recent hardening passes added constant-time
API token comparison, tolerant runtime JSON/JSONL state parsing, atomic
active-paper selection writes, `.coverage` deploy-sync exclusion, blank and
unsafe request-ID sanitation, `/query/report` download header preservation, and
full-history stable artifact-ID export from the local job history. Unsafe
request IDs with bearer/token/secret-like values or invalid correlation-id
characters are not echoed into response IDs or API-access runtime events, and
unsafe legacy runtime-event request IDs are projected as
`request_id_present`/`request_id_redacted` booleans instead of raw values. The
local API-key, product, and share-link registry CLIs also sanitize output-write
and SQLite registry errors without exporting output paths or crashing during
error reporting. The
live answer/retrieval eval JSON report path also stores request-ID evidence as
`request_id_present`/`request_id_redacted` booleans instead of copying raw live
request IDs into archived quality reports. The
API-access audit event now emits only route presence plus a short route-template
fingerprint instead of raw request paths or route strings. The
`/admin/events` viewer also sanitizes runtime-event metadata before returning
events, redacts common sensitive key variants such as access-token, API-key,
token-value, camelCase/PascalCase path or URL fields, raw prompt, and raw answer
fields while preserving safe aggregate status/count/token metrics, and applies
search to that sanitized projection rather than raw JSONL content. Top-level
runtime-event messages with URL/path/token/prompt/answer-like value assignments
or bare `sk-...` secret-like tokens are also replaced in the admin-facing
projection. Artifact list/search
responses, job artifact sub-objects, Streamlit artifact views,
generated-artifact RAG context, and download filenames now use a public
projection that omits raw artifact URIs, local paths, titles, owner IDs/labels,
prompts, and source-reference values while preserving stable IDs and safe
metadata flags/counts. Download filenames also reject unexpected artifact IDs,
and public cost summaries accept bounded finite numeric values only. `GET /jobs` now returns a
no-secret job summary and searches only safe status/ID/count/code/ownership-source
fields rather than raw request payloads, execution results, logs, owner labels,
idempotency keys, or artifact metadata; owner IDs and owner labels are
represented only by presence flags, while exact `owner_id` filtering remains
available for local operator workflows. The Streamlit latest-job panel renders
the same no-secret summary boundary. Corpus profile report downloads also use
one shared API/UI helper that derives filenames from the normalized saved profile
ID rather than the raw path parameter and hashes secret-like profile IDs before
they reach filenames.
The local artifact store now validates relative targets under the artifact
root, writes through atomic replacement so preexisting destination symlinks are
not followed, rejects symlink parent escapes, and rejects symlink/non-regular
copy sources. Artifact and job registry list limits are clamped at the helper
layer, and malformed SQLite mirror payloads are treated as cache misses so
artifact stable-ID export and job lookups can fall back to persisted job
history.
The product shell now also has a no-secret local product activation rehearsal
that creates disposable SQLite API-key and product registries, verifies
hash-only key lifecycle, workspace RBAC, cross-workspace isolation denials,
request quota limiting, local billing attribution, and `product_readiness`
activation in one temporary run without exporting raw tokens, workspace/user
identifiers, or paths. Reusing a fixed rehearsal root writes under a
dedicated hidden state directory and resets only rehearsal-owned SQLite/WAL/SHM
files before each run, so prior rehearsal state cannot inflate key counts or
quota usage and caller-root files with the same names are preserved. FastAPI and
Streamlit now expose the same drill as an explicit on-demand admin surface with
no-secret JSON/Markdown output and report download.
The provider side now also has a no-secret provider runtime rehearsal that
exercises deterministic SVG generation, local Python execution with artifact
capture, the Octave-compatible runtime-unavailable/available branch, Docker
readiness reporting, execution abuse-policy denial for unsafe Python/Octave
snippets, and provider-readiness local foundation while keeping real external
provider activation explicitly disabled. The abuse-policy rehearsal exports only
booleans and counts, not source snippets, stdout/stderr, or paths. Reusing a
fixed provider rehearsal root writes generated artifacts under a dedicated
hidden state directory, resets stale rehearsal-owned artifacts on each run, and
preserves caller-root `artifacts/` files. FastAPI and Streamlit now expose the
same drill as an explicit on-demand admin surface with no-secret JSON/Markdown
output and report download. Provider readiness also blocks activation when the
provider quota guard is enabled but request token limits are invalid, matching
the runtime guard's 503 behavior.
Provider quota/cost guard, query-cost parsing, and artifact public cost metadata
now treat non-finite `NaN`/`Infinity` values and extreme decimal exponents as
invalid no-secret configuration, so status/readiness, artifact projections, and
provider pre-call guard decisions do not crash or emit unbounded public strings
when local cost settings are malformed.
Provider usage extraction now treats upstream token counts as optional metadata:
malformed fields fall back to the next valid provider field, zero-token counts
are preserved, and totals are derived from prompt/completion counts when no
valid total is returned.
The platform migration rehearsal now also has a no-secret durable job-store
manifest and verifier for staged `jobs.sqlite3` state. It represents job rows
and idempotency claims with SHA-256 tokens plus aggregate counts, so worker
migration scope can be checked without exporting job payloads, owner IDs,
request IDs, worker IDs, idempotency keys, logs, artifacts, or execution output.
The job-store verifier now evaluates scheduled/due fields relative to the
manifest timestamp and compares idempotency-claim metadata for matching claim
tokens, so unchanged persisted job rows are not falsely rejected by wall-clock
drift and forged claim metadata is still detected.
Explicit migration rehearsal staging roots are also rejected when they equal the
source project, sit under it, or contain it, so `--overwrite-staging` cannot
clear source runtime state by accident.
The platform-readiness foundation now also separates distributed job-store
readiness from metadata database readiness through `DISTRIBUTED_JOB_STORE_*`
configuration and no-secret admin/report/metrics fields. A production migration
preflight now composes local storage-schema evidence, no-secret runtime backup
manifest, restore dry-run, and read-only local job-store contract checks into a
single CLI gate. A local migration rehearsal now stages required runtime state
into a temporary or explicit staging root, then verifies restore-check and
schema integrity before any external backend is activated. FastAPI and
Streamlit now expose that temporary local drill as an explicit on-demand admin
surface with no-secret JSON/Markdown output; the public API/UI projection keeps
full object/job-store manifest lists out of the response and returns only
summary counts, booleans, and reason codes. The product shell now also has
no-secret product-readiness and provider-readiness preflights that
separate local foundations from real identity/quota/billing activation and real
image-provider, hosted sandbox, MATLAB, and provider quota/cost-guard
activation. The eval report also
carries staged quality-maturity targets for self-use, small-group, and community
readiness. `scripts/quality_readiness.py` now turns those targets into a
no-secret preflight that can merge explicit live eval reports and emit a target
gap summary with current/expected/gap values; self-use and the latest
live-verified small-group lane are met, while community remains a measured gap
before broader release work. It also emits no-secret evidence collection plans
for the next maturity target and the community target, with placeholder
`evaluate_rag.py` and `quality_readiness.py` commands instead of concrete report
paths, URLs, prompts, answers, source content, or credentials.
The current product-shell slice also adds optional local SQLite API key and
product registries. The API-key registry stores token hashes only and supports
create/list/verify/revoke plus FastAPI auth integration; `create` requires JSON
output so the one-time raw token cannot be silently lost in Markdown mode. The
product registry stores local users, workspaces, quota limits, usage events, and billing
attribution records for no-secret readiness checks; when explicitly enabled, it
also guards `/query*` routes with local request quotas, enforces local
workspace-role permissions on query/job/corpus/admin write paths, and records
metadata-only usage or RBAC denial attribution. It now also exposes a local
operator management surface through `/admin/product-registry/*` and the
Streamlit admin panel for workspace creation, member roles, quota limits, and
permission checks; when local product RBAC is enabled, workspace list/detail
and permission-check reads also require local admin-write permission instead of
being readable by viewer tokens. The Streamlit direct management forms also
require `FLUXMIND_STREAMLIT_PRODUCT_REGISTRY_MANAGEMENT_ENABLED=true`, so
enabling the SQLite registry alone does not expose public UI write controls. The
local product activation rehearsal now proves those SQLite contracts together
without enabling an external identity provider or payment system. Both
registries are covered by storage-schema, runtime-manifest, admin-inventory,
CLI, API/UI anchors, and unit/API tests.
The share-link slice now also has an optional local SQLite token lifecycle
registry. It stores only token hashes, returns the raw share token once on
create, supports list/revoke/resolve through CLI, `/admin/share-links*`
routes, and an explicitly enabled Streamlit operator panel, records
metadata-only `share_link_admin` events, and projects only counts, booleans,
and presence/fingerprints outside the registry internals.
The share-link SQLite file is covered by storage-schema, runtime-manifest,
admin-inventory, platform-migration runtime groups, health anchors, and
unit/API/CLI tests.

The incomplete scope is production platformization: real external providers,
hosted sandboxes, MATLAB licensing, external identity providers,
identity-backed quotas, external billing/payment providers, external distributed
worker/storage activation, and live external database/object-storage migration
execution tests. Those remain planned, intentionally disabled, or
decision-gated.

## WP0: Stabilize Current Production

Status: complete for the current deployed no-key baseline; ongoing operational
checks still apply before each deploy

- Keep CI green: `python -m pytest` and `python scripts/health_check.py`.
- Keep Trace-Twin health green with:
  `python scripts/health_check.py --ssh-host root@100.100.233.26`.
- Provider-error normalization exists for UI and API responses.
- Request IDs exist for `/query`, Streamlit responses, and logs.
- FastAPI startup starts retrieval warmup for an already-present FAISS index in
  the background and recovers queued jobs; it does not synchronously rebuild a
  missing index or block the API port while warming retrieval state.
- `/health` reports process liveness, while `/ready` reports retrieval warmup
  readiness for the existing FAISS index.
- `scripts/health_check.py` reports local/remote FAISS index size and active
  paper count when available.
- `scripts/health_check.py` verifies that API startup avoids synchronous index
  rebuilds, that the non-blocking readiness route exists, that remote chunk
  metadata rows exist when an active corpus is present, and that deployed
  `/corpus/chunks` filters can return a sampled chunk while rejecting an
  impossible query.
- `scripts/health_check.py --url` retries transient HTTP warmup failures such
  as 502/503/504/429 before reporting endpoint failure.
- `scripts/health_check.py --ssh-host ...` includes recent journal error lines
  for the UI/API services and source-code anchors for the deployed API-key
  create-output guard, product-registry read RBAC guard, and explicit Streamlit
  product-registry management flag.
- `scripts/deploy_sync.py` wraps the production source sync with a dry-run
  default and required excludes for secrets, virtual environments, models, and
  mutable runtime state before allowing `rsync --delete`. Coverage data
  (`.coverage`) is also excluded so local test artifacts are not copied to
  `/opt/fluxmind`.
- API token checks use constant-time comparison for configured `X-API-Key` or
  bearer credentials while preserving the existing no-token local mode.

Acceptance:

- Local tests pass.
- Public UI/API return 200.
- Remote systemd services, ports, model config, and disk checks pass.
- Production source sync is reproducible without copying over runtime state.
- Browser translation guard remains covered by tests.

## WP1: RAG Quality Baseline

Status: complete for the current no-key RAG quality baseline: offline baseline, source/page fixture verification, recorded-answer
scoring, optional live `/query/inspect` regression scoring, no-LLM
`/query/retrieve` retrieval diagnostics, local hybrid retrieval, deterministic
BM25-lite lexical reranking, optional local CrossEncoder reranking,
generated-answer citation-inspection metadata, numbered-citation prompt guards,
metadata-only retrieval trace events/admin summaries/metrics and local
retrieval-quality advisory alerts,
retrieval-only source/page cases, 13 local code-output artifact cases
(12 Python, one Octave-compatible runtime-aware case; four Python cases are
job-backed) across reusable execution templates plus
paper-specific local fixtures, 30 seeded PDF equation/table/figure/algorithm
structure extraction acceptance cases, optional JSON eval report export, and
aggregate eval-set regression gates implemented. The baseline now has a 42-case
small-group quality gate with 42 recorded answers plus 65 retrieval-only cases
for 107 total no-LLM retrieval questions. It gates 145 expected source/page refs, topic-tag coverage,
ontology-group coverage, and eval-lane coverage for retrieval, answer quality,
equation fidelity, code generation, forum-style debugging, failure modes, and
paper-to-code reports; external/service reranking plus richer PDF layout
extraction remain planned, while broader real Octave execution coverage depends
on installing an `octave` binary in CI/runtime.

- `eval/rag_baseline.json` contains a 42-case control-engineering answer
  evaluation set plus 65 retrieval-only source/page cases, with a small domain
  ontology for SMC, PMSM/FOC, observer/estimation, and implementation-trust
  groups.
- Answer cases record expected source papers/pages, fixture snippets, recorded
  answers, required answer terms, minimum answer-term coverage, topic tags, and
  eval lanes.
- `src.chain.validate_numbered_citations()` validates answer citations like
  `[1]` against retrieved document refs.
- `scripts/evaluate_rag.py` runs the offline baseline without network calls and
  fails recorded answers that miss required refs or key-term coverage thresholds.
- `retrieval_only_cases` verify source/page snippets without requiring fixture
  or recorded answers, so no-LLM retrieval coverage can grow independently from
  answer-quality fixtures.
- `code_output_cases` run local no-key execution fixtures in a temporary
  artifact store and verify stdout terms, plot/text artifacts, checksums/byte
  metadata, runtime metadata, provider/direct mode, local job-backed mode, and
  reusable execution-template coverage without writing project runtime artifacts.
  They include an Octave-compatible PMSM current-decay template case that passes
  either by producing the expected artifact when `octave` is installed or by
  matching the structured runtime-unavailable diagnostic when it is absent.
- `pdf_structure_cases` verify that representative seeded PDF pages expose
  equation, table, figure, and algorithm markers through no-key PyMuPDF extraction, so
  paper-to-code workflows can fail locally when source layout anchors disappear.
  The real corpus now includes a numbered algorithm anchor from the DSMO/LQR
  current-control paper, so algorithm extraction is covered by both synthetic
  marker tests and seeded-PDF evaluation.
- Offline expected refs now verify that the referenced PDF exists, the page is
  parseable, and the configured snippet appears on that page.
- Provider failure fixtures cover timeout, 429/rate-limit, empty output, and
  malformed streaming chunks.
- Answer modes exist in the prompt/API/UI: explanation, derivation,
  implementation, literature review, and code generation.
- `src.chain.hybrid_retrieve()` merges FAISS vector hits with local keyword
  matches from an expanded candidate pool in the indexed docstore, dedupes
  chunks, and keeps the final context bounded by `TOP_K`.
- `src.chain.rerank_documents()` applies a deterministic no-key BM25-lite
  lexical reranker before context formatting, with a first pass that preserves
  source diversity so one high-scoring paper cannot fill the entire context for
  comparison questions.
- `src.chain.learned_rerank_documents()` can apply a no-key local CrossEncoder
  reranker before final context assembly when `RERANKER_MODEL` points to an
  existing local model path. Empty or missing paths fall back to BM25-lite and
  do not trigger runtime model downloads.
- `src.chain.query_with_metadata()` returns the generated answer, retrieved
  context refs, and numbered citation validation; `POST /query/inspect` exposes
  the same metadata for authenticated inspection without changing the
  compatibility shape of `/query`.
- `src.chain.retrieve_with_metadata()` returns retrieved context refs,
  source/page completeness, and the citation guard without calling the LLM;
  `POST /query/retrieve` exposes the same diagnostics for authenticated no-key
  retrieval checks.
- `scripts/evaluate_rag.py --retrieval-url ...` can call a deployed
  `/query/retrieve` endpoint and score expected-source retrieval coverage plus
  source/page completeness without model generation.
- `scripts/evaluate_rag.py --live-url ...` can call a deployed `/query/inspect`
  endpoint, score citation validity, expected-source retrieval coverage, and
  answer-term coverage, while reading the API token from an environment variable
  instead of the repository.
- `scripts/evaluate_rag.py --json-report ...` writes a no-secret
  machine-readable summary of offline/retrieval-only/code-output/provider/
  recorded/live retrieval/live answer eval results plus quality-maturity target
  gaps for deployment records.
- `scripts/quality_readiness.py` turns the same quality-maturity targets into a
  no-secret readiness preflight for self-use, small-group, and community
  release gates. It can merge explicit live eval reports for live retrieval
  count/pass-rate and live answer count/pass-rate/term-coverage evidence. Its
  Markdown and JSON output include a target gap summary for each maturity target
  plus an evidence-request summary that labels each remaining gap as
  `corpus_manifest`, `eval_baseline`, or `live_eval_report` work, so community
  shortfalls can be planned from executable metric deltas. It exits nonzero
  under `--require-target community` until the community corpus/eval/live
  evidence gaps and live answer quality thresholds are closed. It also emits
  no-secret evidence collection plans for the next target and community target,
  converting those requests into placeholder operator commands while keeping
  report paths, prompts, answers, source content, and API tokens out of the
  readiness output.
- `GET /admin/quality-readiness`, `POST /admin/quality-readiness`,
  `GET /admin/quality-readiness/report`, `POST /admin/quality-readiness/report`,
  and the Streamlit admin panel expose that quality preflight as an explicit
  on-demand JSON/Markdown surface. The POST API variants and UI upload accept
  in-memory `evaluate_rag.py --json-report` evidence while keeping raw reports,
  filenames, paths, prompts, answers, and source content out of responses.
- `GET /admin/product-activation-rehearsal`,
  `GET /admin/product-activation-rehearsal/report`, and the Streamlit admin
  panel expose the local SQLite product activation rehearsal as an explicit
  on-demand JSON/Markdown surface with no raw token, path, prompt, answer, or
  external account export.
- `GET /admin/provider-runtime-rehearsal`,
  `GET /admin/provider-runtime-rehearsal/report`, and the Streamlit admin panel
  expose the local provider runtime rehearsal as an explicit on-demand
  JSON/Markdown surface with no raw path, prompt, content, credential, or
  external provider export.
- `eval/rag_baseline.json` includes aggregate `quality_gates` for minimum case
  count, retrieval-only case count, total retrieval-question count, expected
  source-ref count, provider fixture count, recorded-answer count/pass
  rate/average term coverage, code-output case count/language/template/pass
  rate/execution-mode coverage, PDF structure case count/kind/pass-rate,
  answer-mode coverage, topic-tag coverage, ontology-group coverage, eval-lane
  coverage, and optional live answer/retrieval pass-rate thresholds.
- The baseline now covers all answer modes: explanation, derivation,
  implementation, literature review, and code generation.
- The generation prompt now tells the model the valid numbered context-ref range
  for each answer so live answers are less likely to invent citations such as
  bibliography numbers.
- Generated answers neutralize out-of-range bracket numbers before validation so
  source-paper bibliography refs cannot masquerade as FluxMind context refs.
- Still planned: external/service reranking that would require a hosted model or
  new account; broader real Octave execution eval after an `octave` binary is
  installed in CI/runtime; a real-PDF algorithm-caption acceptance case (needs a
  curated paper containing a numbered `Algorithm N` block).

Acceptance:

- Evaluation command runs without network where fixtures are available.
- Citation regressions fail locally before deployment.
- Source/page fixture regressions fail locally before deployment.
- Recorded answer coverage regressions fail locally before deployment.
- Generated answer citation refs can be inspected against retrieved source/page
  context without reading logs or raw prompts.
- Retrieval source/page quality can be inspected without calling a model
  provider.
- Retrieval-only source/page regressions fail independently from recorded-answer
  quality fixtures.
- Code-output regressions fail when local execution misses expected stdout,
  runtime metadata, or generated plot/text artifacts.
- PDF structure regressions fail when representative source pages no longer
  expose required equation/table/figure/algorithm anchors.
- Live model answers can be scored through the deployed inspect endpoint without
  committing provider tokens.
- Eval breadth and aggregate answer-quality regressions fail through configured
  quality gates, not only per-case checks.
- Eval results can be exported as JSON for CI/deployment evidence without
  copying provider tokens.
- Provider errors surface as structured user-facing messages.

## WP2: Corpus and Storage Layer

Status: complete for the current no-key local corpus/storage baseline: local JSON corpus metadata baseline, reusable local corpus profiles,
bibliographic paper enrichment, uploaded-PDF metadata extraction with
first-page author/keyword fallback, SQLite current-state paper/chunk metadata
mirrors, checksum-based uploaded-PDF deduplication, active/deactivated
selection workflow, corpus lifecycle status, local paper metadata filtering, and
admin index freshness plus durable storage readiness checks implemented; durable
local storage inventory, local storage-schema inventory, no-secret runtime backup
manifest, and dry-run restore verifier implemented;
durable multi-user database/object storage migration remains planned

- The bundled seed corpus currently contains 30 open-access papers across SMC,
  PMSM sensorless control, sliding-mode observers, flux observers, adaptive
  parameter estimation, and MRAS flux-linkage observation.
- `src/metadata.py` stores local paper metadata in git-ignored
  `metadata/corpus.json`.
- Local corpus/profile JSON writes use same-directory temporary files followed
  by atomic replace, so concurrent local reads should not observe partial JSON.
- Paper metadata is mirrored into `metadata/corpus.sqlite3` as a local
  current-state index for faster inspection and future database migration.
- Indexed chunk metadata is mirrored into `metadata/chunks.sqlite3` with source
  path, page, chunk index, content hash, character count, and preview text.
- Paper records include checksum, title, authors, year, DOI, arXiv ID, venue,
  topic tags, source path, source kind, active flag, indexed status, chunk
  count, and parse/index error fields.
- Uploaded/unmanifested PDFs get best-effort no-key metadata extraction from
  embedded PDF metadata and first-page title, author, DOI/arXiv, year, and
  keyword/index-term text, while curated manifest values still take precedence.
- Uploaded PDFs pass a local pre-write scan before persistence. The scan checks
  PDF magic and PyMuPDF parseability, rejects encrypted PDFs by default, blocks
  common active-content markers, applies a configurable page cap, and records
  only metadata-only `upload_scan` runtime events with reason codes and counts.
- Uploaded PDFs are deduplicated by SHA-256 against the selectable local corpus
  before writing a new local file or adding duplicate chunks to FAISS.
- Selectable corpus discovery and active-paper persistence accept only
  project-local regular PDF files; symlinked PDFs are skipped, and broken
  upload symlinks are treated as occupied filenames instead of write targets.
- `GET /corpus/papers` lists the current local paper metadata without requiring
  manual filesystem inspection, with local filters for query text, active state,
  source kind, and indexed status.
- `GET /corpus/chunks` lists local indexed chunk metadata with optional
  `source_path`, `page`, and `q` filters.
- `GET /corpus/structure` lists no-key PDF layout markers from selectable
  PDFs with optional source, kind, page, text-query, and limit filters.
- `GET /corpus/structure/report` exports the same filtered PDF layout markers
  as a Markdown handoff report with kind/source summaries.
- `GET /corpus/status` exposes corpus lifecycle state as `queued`, `parsing`,
  `indexed`, `failed`, `stale`, or `empty` from index rebuild jobs, paper status,
  and index freshness.
- `GET /admin/status` reports corpus JSON/SQLite storage state without SSH.
- `GET /admin/status` reports whether the current FAISS/chunk metadata source
  set is fresh, stale, missing, or empty relative to the active corpus.
- `GET /admin/status` reports no-secret metadata database and object-storage
  readiness for future production storage backends. Local JSON/SQLite/filesystem
  storage remains active; external database URLs, buckets, and endpoints are
  reported only as configured booleans and are not connected or exposed.
- `GET /admin/status` reports local storage inventory for metadata, jobs,
  artifacts, uploads, and FAISS index files as paths, file counts, byte totals,
  and known-file existence flags without returning file contents.
- `GET /admin/status` reports local storage-schema readiness for corpus/chunk
  metadata, jobs, artifacts, the API-key registry, and runtime events by checking
  schema version, JSON/JSONL shape, and SQLite table/column presence without
  returning row contents, prompts, answers, filenames, owner IDs, request IDs,
  source paths, token values, or runtime file contents.
- `scripts/storage_schema.py` runs the same no-secret schema readiness check from
  the CLI with JSON/Markdown output, `--target-root`, and a nonzero exit code
  when drift is detected.
- `scripts/platform_migration_preflight.py` composes the schema check, runtime
  manifest, restore dry-run, local durable job-store contract, storage readiness,
  distributed job-store readiness, and platform blocker summary into one
  no-secret CLI. Default mode fails only when local migration evidence is
  incomplete; `--require-activation` also fails until external metadata
  database, object storage, and distributed job-store targets are configured.
- `scripts/platform_migration_rehearsal.py` performs the next local migration
  drill by copying required runtime state into a staging root, verifying the
  staged copy with the runtime restore-check, and checking staged storage schema.
  Default mode uses an auto-cleaned temporary staging root; retained staging
  requires `--staging-root`, existing staging data requires
  `--overwrite-staging`, and runtime dependency groups such as models are skipped
  unless `--include-runtime-dependencies` is supplied.
- The same rehearsal can emit an opaque object-storage migration manifest with
  `--include-object-manifest`: object keys, SHA-256 hashes, byte counts, group
  names, and source-path tokens are included, while source paths, filenames,
  bucket names, endpoints, credentials, `.env`, and file contents are excluded.
  The verifier rejects unsafe nested or camelCase path/filename/bucket/endpoint/
  credential fields instead of merely omitting them from its output.
- The same rehearsal can emit a no-secret durable job-store migration manifest
  with `--include-job-store-manifest`: job rows and idempotency claims are
  represented by SHA-256 tokens, status/kind/lease/deadline metadata, and
  aggregate counts while job payloads, owner IDs, request IDs, worker IDs,
  idempotency keys, logs, artifacts, stdout/stderr, and secrets are excluded.
- `scripts/platform_migration_rehearsal.py --verify-job-store-manifest` checks a
  job-store manifest, or a full rehearsal JSON containing one, against local or
  staged durable job state and reports only missing/mismatched/extra token counts
  and changed metadata fields. The verifier rejects unsafe nested or camelCase
  payload/owner/request/worker/idempotency/log/artifact/stdout/stderr/credential
  fields instead of treating them as harmless ignored metadata.
- `scripts/runtime_manifest.py` exports a no-secret runtime backup manifest for
  the local state trees that source deploys exclude, with file counts, byte
  totals, and SHA-256 hashes for known metadata/job/API-key-registry/index files
  without exporting file contents or `.env` values.
- `GET /admin/runtime-manifest`, `GET /admin/runtime-manifest/report`, and the
  Streamlit runtime status panel expose the same no-secret backup manifest
  without requiring SSH access to run the CLI by hand.
- Restore dry-run verification is available through
  `scripts/runtime_manifest.py --restore-check`, authenticated
  `POST /admin/runtime-manifest/restore-check`, and authenticated
  `POST /admin/runtime-manifest/restore-check/report`, plus the Streamlit
  runtime status panel upload/report controls. These surfaces verify a saved
  manifest against a target runtime root without copying, overwriting, deleting,
  or restoring files. They report manifest contract errors, missing or
  mismatched groups, known files, byte counts, and SHA-256 hashes.
- `PUT /corpus/active` persists activation/deactivation choices after validating
  project-relative source paths against the selectable corpus.
- `load_library_manifest()` and `load_active_paper_paths()` tolerate malformed
  or wrong-shaped runtime JSON, reject non-project/unknown active selections, and
  fall back to the bundled library selection when the active-selection file is
  unusable.
- `save_active_paper_paths()` deduplicates existing PDFs and writes
  `faiss_index/active_papers.json` through a same-directory temporary file plus
  atomic replace.
- `GET /corpus/profiles`, `POST /corpus/profiles`,
  `GET /corpus/profiles/{profile_id}/status`, and
  `POST /corpus/profiles/{profile_id}/activate` persist and inspect reusable
  no-key local active-paper selections under `metadata/corpus_profiles.json`,
  allowing multiple named corpus selections to coexist before real
  workspace/user storage exists.
- `GET /corpus/profiles/{profile_id}/report` exports the same no-secret profile
  status as Markdown for handoff or offline review.
- `POST /corpus/profiles/{profile_id}/rebuild` activates a saved profile and
  queues the selected-PDF FAISS rebuild through the local async job manager.
- Upload and selected-PDF index rebuild flows update paper metadata.
- Upload scan outcomes are visible in admin status/report and Streamlit runtime
  status without exposing filenames, uploaded bytes, checksums, or request
  bodies.
- Decide storage path: local volume first or object storage plus relational DB.
- Still planned: durable metadata for chunks, corpora, jobs, artifacts, users,
  ownership, and retention in a production database; object storage; richer
  durable deactivation/reactivation workflows across multiple users/corpora.

Acceptance:

- Rebuilding an index is a job with status and logs.
- A paper can be indexed and traced to source path/checksum/chunk count.
- Duplicate uploads reuse an existing selectable/indexed PDF instead of creating
  a second local file or duplicate vector chunks.
- Upload filename conflicts treat symlinks as occupied paths, so a broken
  symlink with the requested name cannot redirect the write outside the upload
  directory.
- Failed upload scans are blocked before local file write and leave only
  no-secret reason-code metadata for operators.
- Indexed chunks can be traced to source path, page, sequence, hash, and preview.
- Corpus lifecycle status can be inspected without inferring from job and index
  records by hand.
- Index freshness can be inspected without comparing active paper files and
  chunk SQLite rows by hand.
- Active/deactivated state can be managed without editing runtime files by hand.
- Named local corpus selections can be saved and reactivated without copying or
  editing runtime files by hand.
- Named local corpus selections can be inspected for paper availability,
  active-selection match, index freshness, and rebuild requirement before
  activation.
- Named local corpus selection status can be exported without reading raw
  profile JSON or activating the selection.
- Storage state can be listed without reading raw filesystem layout manually.
- Runtime backup scope can be inspected before manual backup or storage
  migration without copying runtime contents into source control.
- Future database/object-storage readiness can be inspected without committing
  credentials, opening external connections, or migrating runtime data.
- Paper metadata can be searched or filtered locally without fetching and
  scanning the full corpus list by hand.

## WP3: Job System

Status: complete for the current no-key local job/worker baseline: local JSONL
history plus SQLite current-state and idempotency-claim indexes, in-process
async queue, scheduled retry/backoff, restart recovery for queued jobs, queue
health, queue-level deadlines, bounded automatic retry and dead-letter state,
durable worker lease metadata, enabled local durable worker service foundation,
stable execution timeout diagnostics, running Python cancellation for in-process
and explicit durable local workers, and cancellable index-rebuild checkpoints
plus admin worker-lease visibility implemented; distributed multi-worker queue
and full running cancellation for every future worker type remain planned

- Local JSONL job records exist in `src/jobs.py`.
- JSONL fallback reads skip malformed/non-record lines with warnings, so a single
  bad append-only history line does not block job listing, single-job lookup, or
  SQLite mirror recovery.
- Job writes are mirrored into `jobs/jobs.sqlite3` for current-state lookups and
  migration toward durable worker storage.
- The job SQLite mirror is treated as a rebuildable current-state cache:
  malformed payload rows are skipped or refreshed from append-only JSONL instead
  of breaking list, get, idempotency lookup, or worker claim paths.
- Immediate and async job creation requests accept optional `idempotency_key`.
  Duplicate submissions with the same job kind and key return the existing
  persisted job through the durable SQLite `job_idempotency` claim table; missing
  keys continue to create new jobs.
- Async job creation requests accept `max_attempts` and `retry_backoff_s`.
  Failed attempts requeue the same durable job until the attempt cap is reached,
  then persist `dead_lettered` plus `dead_lettered_at`. The default is one
  attempt, preserving previous single-run behavior.
- Query and job-creation requests accept optional local `owner_id` and
  `owner_label` metadata. Omitted values normalize to `local-user` /
  `Local user`.
- Durable job records persist `owner_id`, `owner_label`, and
  `ownership_source`, mirror those fields into SQLite, and include them in
  transition logs. Retry and scheduled retry inherit the original job owner.
- `POST /jobs/image/mock` creates a mock image-generation job.
- `POST /jobs/code/python-local` creates a development-only Python execution
  job.
- `POST /jobs/index/rebuild` rebuilds the FAISS index from selected PDFs as a
  persisted job.
- `POST /jobs/async/image/mock`, `POST /jobs/async/code/python-local`,
  `POST /jobs/async/code/octave-local`, and `POST /jobs/async/index/rebuild`
  enqueue those local jobs through an in-process background worker.
- `GET /jobs` lists latest job summaries with local `q`, `status`, `kind`, and
  `owner_id` filters over a no-secret search projection; raw owner IDs/labels
  are reduced to `owner_id_present` and `owner_label_present`.
- `GET /jobs/{job_id}` returns persisted job detail status for an exact job ID.
- Job responses include the normalized `idempotency_key` when one was supplied.
- Job responses include normalized owner metadata for local inspection. The
  fields are metadata only, not authentication, tenant isolation, quotas, or
  billing.
- `POST /jobs/{job_id}/retry` retries failed/cancelled local jobs with a new
  job ID.
- `POST /jobs/{job_id}/retry-scheduled` queues failed/cancelled local jobs for
  delayed retry with `parent_job_id` and `not_before` metadata.
- Manual retry and scheduled retry can also start a fresh retry from
  `dead_lettered` jobs.
- Async job and scheduled-retry requests can set `queue_timeout_s`; expired
  queued jobs persist `job_deadline_exceeded` before execution.
- `AsyncJobManager.recover_queued_jobs()` rehydrates queued/scheduled jobs from
  SQLite/JSONL after service restart and returns them to the local worker queue.
- Workers claim queued jobs with `worker_id`, `leased_at`, and
  `lease_expires_at` metadata before provider execution. Expired queued leases
  can be reclaimed, and the current in-process worker uses this same durable
  claim path.
- `LocalDurableJobWorker`, `scripts/run_job_worker.py`, and
  `deploy/systemd/fluxmind-worker.service` provide an enabled no-key local
  worker-service foundation that can claim and run due queued jobs from durable
  state outside the API/Streamlit process.
- The explicit durable worker polls durable job state while a provider is
  running; if another process marks that job `cancelled`, local Python/Octave
  providers receive a cancellation event and terminate the child process.
- `GET /admin/status` exposes `queue_health` with queued, due, scheduled,
  expired, running, leased queued, expired lease, running lease, and oldest
  queued timestamps.
- `GET /admin/status` exposes `worker_leases` with no-secret worker IDs,
  active/expired lease counts, and latest leased job summaries.
- Admin status/report expose metadata-only local job-health advisory alerts for
  recent failed jobs, dead-lettered jobs, expired queued deadlines, and expired
  worker leases, controlled by `JOB_ALERT_FAILED_MIN_EVENTS` and
  `JOB_ALERT_EXPIRED_MIN_EVENTS`.
- `POST /jobs/{job_id}/cancel` records cancellation for queued/running job
  states. Running local Python jobs observe cancellation; index rebuild jobs
  check cancellation during PDF loading, splitting, and before committing rebuilt
  index state.
- Streamlit sidebar can trigger selected-PDF index rebuild jobs, mock image
  jobs, local Python jobs, and display filtered latest job status through a
  no-secret summary projection.
- Streamlit recent-job panel can cancel queued/running jobs and retry
  failed/cancelled jobs immediately or after a local backoff delay.
- Local execution timeouts persist as `execution_timeout` instead of generic
  execution failures.
- Job records preserve no-secret transition logs for queued, running, terminal,
  cancelled, automatic retry, and dead-letter states.
- Still planned: distributed database-backed worker beyond the local SQLite
  recovery/lease/service bridge and cancellation for future external workers.

Acceptance:

- Job records preserve request, result, artifacts, errors, attempts, request
  IDs, and transition logs.
- Failed code-execution jobs preserve stderr/error details.
- API and Streamlit can show running/succeeded/failed states through job status.
- Queued local job endpoints return without blocking request handlers.
- Queued delayed retries can recover after API service restart.
- Bounded automatic retry can requeue failed attempts and dead-letter exhausted
  jobs without creating duplicate job IDs.
- Remaining: production-grade long-running work still needs a distributed
  worker/storage backend beyond the local SQLite worker service, plus
  cancellation propagation for future non-local providers.
- Worker/lease activity can be inspected without SSH or raw SQLite reads.

## WP4: Image and Diagram Generation

Status: complete for the current no-key diagram/artifact baseline: provider-neutral plumbing, no-key mock provider, local SVG engineering
diagram templates, local artifact export, artifact metadata with SQLite
current-state mirror, and RAG artifact references implemented; real
image-provider activation remains disabled until a key/account is configured,
but the provider-readiness surface now reports that disabled state as explicit
activation blocker codes

- `MockImageGenerationProvider` implements the `ImageGenerationProvider`
  contract with deterministic local SVG output.
- Local SVG templates cover generic engineering diagrams,
  sliding-mode-observer blocks, PMSM control loops, and paper-figure redraft
  scaffolds without external image providers.
- `GET /artifacts` lists generated local artifacts from persisted jobs.
- `GET /artifacts/{artifact_id}` exports regular local file artifacts by stable
  ID and rejects symlink artifacts.
- `GET /artifacts` supports local `q`, `kind`, `job_kind`, and `owner_id`
  filters over the public no-secret artifact projection while keeping raw
  owner/path/prompt/reference values out of list responses and search.
- Artifact records inherit owner metadata from their source job so generated
  diagrams, plots, and files remain attributable in local no-key status
  surfaces.
- Artifact metadata is mirrored into `artifacts/artifacts.sqlite3` as a local
  current-state index for inspection and future durable storage migration.
- The artifact SQLite mirror is treated as a rebuildable cache: malformed
  payload rows do not block stable-ID lookup from persisted job history.
- `GET /admin/status` reports local artifact integrity counts by checking
  persisted byte-count and SHA-256 metadata against current files.
- Streamlit sidebar includes a local artifact gallery with stable IDs,
  provider-neutral public metadata, local filters, and download buttons that use
  artifact-ID-based filenames.
- Mock diagrams and execution artifacts store byte counts and SHA-256 checksums
  in provider-neutral metadata, with prompt/style/template/source-reference/
  model/cost fields layered on diagram artifacts without external keys.
- Local artifact writes validate target containment under the artifact root,
  atomically replace destination paths without following preexisting symlinks,
  and reject symlink/non-regular copy sources.
- RAG prompts include recent generated artifacts as stable `[Artifact:<id>]`
  references with safe flags/counts, not raw prompts, owner labels, paths, or
  source-reference values.
- Keep generated images as artifacts rather than inline chat-only blobs.
- `scripts/provider_readiness.py`, admin status/report, metrics, and Streamlit
  expose the external image-provider activation blocker without exporting keys
  or enabling real image calls.

Acceptance:

- A request can generate an artifact record with a stable URI.
- Generated mock diagrams produce persisted artifact URIs.
- Generated mock diagrams can use engineering-specific templates instead of only
  a generic placeholder.
- Generated diagrams and execution artifacts can be listed and exported.
- Generated local artifacts can be downloaded from Streamlit.
- Generated local artifact metadata can be inspected through public summaries
  without scanning raw job history manually.
- Generated local artifact metadata includes stable byte counts and SHA-256
  checksums for later durable storage migration.
- Local artifact integrity can be inspected without reading or exporting raw
  artifact contents by hand.
- Provider can be swapped without changing the UI flow.
- External image-provider activation can be checked without exposing keys or
  enabling real image calls.

## WP5: Code Execution

Status: complete for the current no-key local execution baseline: local request/result plumbing, Python execution, Octave-compatible
execution interface, file/plot artifact capture, workdir path containment,
input file size/count limits, no-secret execution environment/policy metadata,
bounded stdout/stderr output capture, bounded generated-artifact export, Unix
child-process memory/CPU limit metadata/enforcement, opt-in Docker
container execution, Docker readiness reporting, request-level execution policy
preflight, no-secret code-execution outcome events/admin summaries, and
local code-execution advisory alerts, and Streamlit control-engineering
execution templates implemented; real hosted execution and MATLAB activation
remain disabled until infrastructure and license/account decisions are made,
with activation blockers now exposed through provider readiness

- `LocalPythonExecutionProvider` implements the `CodeExecutionProvider`
  contract for development-only Python snippets.
- `LocalOctaveExecutionProvider` implements the same contract for GNU
  Octave-compatible scripts when a local `octave` binary is installed; when it
  is absent, jobs fail with a structured runtime-unavailable diagnostic.
- Python and Octave-compatible snippets can capture stdout, stderr, exit code,
  generated text/file artifacts, and generated image files as plot artifacts.
- Local execution providers reject absolute or escaping input paths/entrypoints
  and skip symlink or out-of-workdir artifact collection.
- The artifact store behind execution export rejects symlink/non-regular copy
  sources and symlink parent escapes, and its atomic writes do not follow
  preexisting destination symlinks.
- Local execution providers reject excessive input file count, per-file bytes,
  and total input bytes before materializing files into the temporary workdir.
- Local execution results persist no-secret reproducibility metadata for the
  language, entrypoint, input file counts/bytes, provider runtime, local Python
  or Octave runtime availability/details, temporary workdir isolation, and
  current network-policy enforcement state.
- Local execution results persist timeout, CPU-time, and memory metadata, and
  Unix child processes receive address-space and CPU-time limits where supported.
- Local and Docker execution cap captured stdout/stderr with
  `CODE_EXECUTION_MAX_STDOUT_BYTES` and `CODE_EXECUTION_MAX_STDERR_BYTES`;
  results persist observed byte counts plus `stdout_truncated`,
  `stderr_truncated`, and `output_truncated` metadata.
- Local and Docker execution bound generated-artifact collection with
  `CODE_EXECUTION_MAX_ARTIFACTS`, `CODE_EXECUTION_MAX_ARTIFACT_BYTES`,
  `CODE_EXECUTION_MAX_ARTIFACT_TOTAL_BYTES`, and
  `CODE_EXECUTION_MAX_ARTIFACT_CANDIDATES`; results persist scanned/exported/
  skipped counts, exported bytes, and `artifact_collection_truncated` metadata.
- `CODE_EXECUTION_BACKEND=docker` routes Python and Octave-compatible code jobs
  through `DockerExecutionProvider` instead of the child-process development
  providers.
- The Docker backend runs `docker run --rm` with network disabled, a
  bind-mounted per-run workdir, read-only root filesystem, memory, CPU, and PID
  limits, dropped capabilities, and `no-new-privileges`.
- `DOCKER_EXECUTION_IMAGE` selects the container image; Python uses `python
  <entrypoint>`, while Octave/MATLAB-compatible requests use `octave --quiet
  --no-gui <entrypoint>` and therefore require an image that contains Octave.
- Admin status reports whether the Docker backend is configured and whether
  Docker is accessible by the runtime user without running user code.
- `CODE_EXECUTION_POLICY=local-safe-v1` runs before local or Docker execution.
  It rejects disallowed Python imports, shell/package-manager command patterns,
  absolute-path literals in common file constructors, and Octave/MATLAB-compatible
  shell, network, or package-install calls.
- `CODE_EXECUTION_ALLOWED_IMPORTS` configures the Python import allowlist. Policy
  decisions persist no-secret metadata such as profile, checked-file count, and
  violation count on execution results.
- Policy failures return the stable `execution_policy_violation` job error code
  without starting the local child process, looking up Octave, or launching a
  Docker container.
- Each code-execution attempt appends a no-secret `code_execution` runtime event
  with job id, owner metadata, language, backend, status/error code, duration,
  artifact count, exit code, output/artifact limit metadata, and policy
  metadata. Source files, stdout, and stderr are not copied into the event.
- Admin status/report summarize recent code-execution events by code, status,
  backend, policy violations, output truncations, artifact collection
  truncations, exported artifact bytes, failure rate, duration, and advisory
  alert codes.
- Configurable local alert thresholds cover minimum recent event count, failure
  rate, and max duration, while policy/output/artifact truncation alerts are
  surfaced whenever those recent signals appear.
- Timed-out local executions return a stable `execution_timeout` job error code.
- `POST /jobs/code/octave-local` and `POST /jobs/async/code/octave-local`
  expose immediate and queued no-key Octave-compatible job flows.
- Streamlit includes a local Octave-compatible job panel.
- Streamlit includes editable no-key Python and Octave-compatible templates for
  control-engineering examples that produce local artifacts through the existing
  providers. Python templates: `smc_reaching_law` and `pmsm_current_step`; Octave
  templates: `pmsm_current_decay` and `smc_sign_switching`.
- Job records persist execution artifacts alongside the execution result.
- Still planned: hosted/distributed execution beyond the local Docker backend,
  production metrics/tracing/alerts beyond the local advisory baseline, deeper
  malware/abuse controls, and true MATLAB activation if that product path is
  chosen.
- The provider-readiness preflight reports hosted execution and MATLAB backend
  activation blockers without exporting sandbox URLs, credentials, or license
  data.

Acceptance:

- Current child-process development provider runs code in a child Python process
  and temporary workdir with path-containment checks, but it is not a production
  sandbox.
- Octave-compatible requests have stable API/UI/job behavior without enabling
  real MATLAB licensing.
- Local control-engineering examples can be launched without writing a blank
  script while still running through the same development providers.
- Execution results are reproducible from stored input files and environment
  metadata.
- Docker/container execution readiness can be inspected without granting Docker
  access or running user code.
- Hosted execution and MATLAB activation readiness can be inspected as blocker
  codes without configuring a sandbox URL, account key, or license in docs.
- When `CODE_EXECUTION_BACKEND=docker` and Docker is available, code jobs run
  inside a container and persist Docker runtime metadata plus generated
  artifacts through the same job/artifact contract.
- Unsafe Python/Octave requests are rejected by policy before execution and are
  persisted with an explicit policy-violation error code.
- A failed local execution returns structured diagnostics without breaking the
  app.

## WP6: Product Shell

Status: complete for the current no-key product-shell foundation: local/admin status foundation, reusable local corpus profiles,
no-secret Markdown status-report export,
Markdown query-report export, provider-failure event history, estimated
query-usage history, metadata-only API access audit summaries,
local API rate-limit status, local job/provider/query/retrieval/code advisory alerts,
metadata-only retrieval trace summaries, no-secret local metrics export,
local storage-readiness dashboard, and local storage
inventory dashboard plus no-secret runtime backup manifest, restore dry-run
verifier, local API-key lifecycle registry, local product registry, local
product quota guard, local product RBAC guard, local product registry
management API/UI, local product activation rehearsal, local share-link token
registry, collaboration-readiness preflight for private corpora/share links, and
provider-readiness preflight implemented; keep external identity providers,
identity-backed quotas, and external billing disabled until decisions are made

- Decide when to replace Streamlit with a real frontend.
- Add private corpora, external identity-backed API keys, external quotas, and
  share/export flows after identity decisions are made. Local users/workspaces
  role guards, and local hash-only share-link tokens already exist as the
  no-key contract.
- Local corpus profiles let multiple named paper selections coexist without
  introducing accounts, permissions, or a public share model.
- `GET /admin/status` exposes no-secret local runtime status for job counts,
  failed jobs, job/artifact owner-count and ownership-source summaries, corpus counts, artifact
  counts/bytes, runtime directory existence/writability/bytes, public model
  names, and disabled external provider/productization switches.
- Streamlit includes a local runtime status panel for common operational checks.
- The Streamlit runtime status panel displays durable storage readiness and
  local metadata/object storage paths from `/admin/status` without exposing
  external storage credentials.
- Admin status and the Streamlit runtime status panel display a no-secret local
  storage inventory with file counts and byte totals for metadata, jobs,
  artifacts, uploads, and FAISS index files.
- Admin status/report, metrics, and the Streamlit status panel display a
  no-secret local storage-schema inventory for JSON, JSONL, and SQLite runtime
  stores. It reports schema readiness for migration planning without returning
  stored content or identifiers.
- `scripts/storage_schema.py` gives the same storage-schema check a CLI preflight
  path for local or target-root use.
- `scripts/api_key_registry.py` manages the local hashed-token registry with
  create/list/verify/revoke/status commands. Created tokens are returned once;
  `create` requires JSON output so that one-time token is actually emitted, and
  list/status/verify/revoke outputs never include raw token values.
- FastAPI auth can accept tokens from the local registry when
  `FLUXMIND_API_KEY_REGISTRY_BACKEND=sqlite`; this is local API authentication,
  not multi-user tenancy, quota enforcement, or billing.
- `scripts/product_registry.py` manages the local SQLite users/workspaces/RBAC/quota
  limits/usage/billing-attribution ledger. It supports status, bootstrap-local,
  set-quota, record-usage, add-member, check-permission, and list-workspaces
  while keeping provider secrets, payment credentials, prompts, answers, and
  runtime file contents out of output. `check-permission` exits nonzero when a
  local role does not satisfy the requested action.
- `GET/POST /admin/product-registry/*` exposes the same local registry as an
  API contract for operator tooling: status, workspace list/detail,
  create/update workspace, add/update member, set quota, set local billing
  attribution metadata, and permission check. Mutations emit metadata-only
  `product_registry_admin` runtime events and require the local registry backend
  to be explicitly configured as SQLite. Workspace list/detail and
  permission-check reads also pass through local admin-write RBAC when the
  product RBAC guard is enabled.
- The Streamlit admin panel includes the local product-registry management
  surface for workspace, member-role, quota, and permission-check operations
  only when both the backend and
  `FLUXMIND_STREAMLIT_PRODUCT_REGISTRY_MANAGEMENT_ENABLED=true` are enabled.
  With the default disabled backend or disabled management flag, it only shows
  no-secret status.
- FastAPI query routes can enforce the local product registry's `requests`
  quota when `IDENTITY_QUOTAS_BILLING_ENABLED=true`,
  `FLUXMIND_PRODUCT_REGISTRY_BACKEND=sqlite`,
  `FLUXMIND_QUOTA_STORE_BACKEND=sqlite`, and
  `FLUXMIND_PRODUCT_QUOTA_GUARD_ENABLED=true` are all set. Over-limit requests
  return `429` before model generation and record only metadata-only quota
  events.
- FastAPI can enforce local product RBAC when
  `IDENTITY_QUOTAS_BILLING_ENABLED=true`,
  `FLUXMIND_PRODUCT_REGISTRY_BACKEND=sqlite`, and
  `FLUXMIND_PRODUCT_RBAC_GUARD_ENABLED=true` are all set. Query routes require
  active workspace membership, local job submit/manage routes allow
  member/admin/owner, and corpus/index/admin destructive writes require
  admin/owner. Denials return `403` before work starts and record only
  metadata-only `product_rbac` events.
- `scripts/platform_migration_preflight.py` gives production storage/worker
  migration a single no-secret CLI gate. It reports `preflight_ok` for local
  evidence and `activation_ready` for configured external backends, keeping
  external URLs, buckets, queue names, credentials, job payloads, and runtime
  contents out of the output.
- `scripts/platform_migration_rehearsal.py` gives the same path a local
  backup/restore drill: copy required runtime groups into staging, verify the
  staged restore against the no-secret manifest, then verify staged storage
  schema. Reports still exclude runtime contents, job payloads, `.env`, external
  URLs, bucket names, queue names, and credentials.
- `scripts/platform_migration_rehearsal.py --include-object-manifest` adds the
  object-storage pre-upload manifest for the staged tree. It proves object key,
  hash, byte-count, and group coverage locally without connecting to object
  storage or leaking source paths, filenames, buckets, endpoints, credentials,
  or contents.
- `scripts/platform_migration_rehearsal.py --verify-object-manifest` checks an
  opaque object manifest, or a full rehearsal JSON containing one, against a
  local/staged runtime tree. It returns only group, source-path-token, hash,
  byte-count, and status-code differences, without leaking source paths,
  filenames, buckets, endpoints, credentials, or runtime contents.
- `scripts/platform_migration_rehearsal.py --include-job-store-manifest` and
  `--verify-job-store-manifest` give the distributed-worker migration path an
  equivalent no-secret job-state manifest. They prove staged durable job rows and
  idempotency claims by token and aggregate status metadata without exposing job
  payloads, owner IDs, request IDs, worker IDs, idempotency keys, logs,
  artifacts, stdout/stderr, credentials, or runtime contents.
- `scripts/product_readiness.py` gives identity/quota/billing productization a
  no-secret CLI gate. Default mode passes when local foundations such as API
  access audit, owner metadata, rate-limit configuration, local API-key registry,
  local product registry, local product quota guard, local RBAC guard, and cost-estimation
  surfaces are present;
  `--require-activation` still fails until the chosen identity provider, quota
  store, billing provider, billing-attribution, quota-guard, and RBAC-guard
  targets are configured.
- `scripts/product_activation_rehearsal.py` gives the local product path a
  disposable SQLite activation drill. It creates local API keys, revokes and
  verifies one key, creates two workspaces, sets quota and billing attribution,
  verifies allowed/denied RBAC decisions, proves cross-workspace access is
  denied before private corpora/share links are enabled, proves an over-quota
  request would be limited, and runs `product_readiness` against those
  temporary stores. The report returns only counts, booleans, and reason codes;
  raw tokens, workspace/user identifiers, paths, prompts, answers, external
  accounts, and payment data are not exported.
- `scripts/collaboration_readiness.py` gives private-corpus and share-link work
  a no-secret pre-activation gate. The default local runtime passes the safe
  local foundation because both capabilities remain disabled; `--require-activation`
  fails until the product registry, RBAC guard, feature flags, and share-link
  token registry target are configured. The JSON/Markdown report includes a role
  policy matrix and blocker codes only, omitting workspace/user/corpus/share
  identifiers, raw URLs, tokens, paths, prompts, answers, and contents. FastAPI
  exposes the same check through `/admin/collaboration-readiness` and
  `/admin/collaboration-readiness/report`, and Streamlit exposes it as an
  explicit on-demand admin-panel check.
- `scripts/share_link_registry.py` gives share-link tokens a local no-secret
  lifecycle surface. `create` requires JSON so the one-time token is not hidden
  in Markdown output; list/revoke/resolve omit raw tokens, URLs, resource refs,
  creator user IDs, descriptions, paths, prompts, answers, and contents. The
  matching `/admin/share-links*` routes plus the explicitly enabled Streamlit
  operator panel use the same SQLite registry. API writes record metadata-only
  admin events.
- `scripts/provider_readiness.py` gives external image providers, hosted
  execution, MATLAB backend/licensing, and provider quota/cost guards the same
  no-secret CLI gate. Default mode passes when local provider foundations are
  available; `--require-activation` fails until the real external targets are
  explicitly configured.
- `scripts/provider_runtime_rehearsal.py` gives the local provider runtime a
  disposable no-secret drill. It generates a deterministic SVG artifact, runs a
  local Python snippet with artifact capture and policy metadata, checks the
  Octave-compatible branch as either real execution or structured
  `runtime_unavailable`, reports Docker readiness by reason code, exercises the
  provider quota/cost guard through one allowed and one over-limit local
  decision, confirms provider-readiness local foundation while leaving external
  activation blocked, and keeps raw paths, prompts, content, credentials, and
  external provider data out of the report. `GET
  /admin/provider-runtime-rehearsal`, `GET
  /admin/provider-runtime-rehearsal/report`, and the Streamlit admin panel
  expose the same drill for explicit operator runs without running the full
  activation suite.
- `scripts/_safe_cli.py` is used by the no-secret readiness/rehearsal CLIs so
  OSError failures preserve safe diagnostic messages while redacting paths,
  URLs, bearer/sk-style tokens, and token/secret-like assignments.
  `scripts/health_check.py` guards that wiring.
- `GET /admin/platform-migration-rehearsal`,
  `GET /admin/platform-migration-rehearsal/report`, and the Streamlit admin
  panel expose the temporary local migration rehearsal as an explicit on-demand
  JSON/Markdown surface. The response includes restore/schema/object/job-store
  summaries but omits raw object manifests, job manifests, local paths, runtime
  contents, buckets, endpoints, credentials, job payloads, and raw IDs.
- `scripts/activation_suite.py` gives those local activation checks one
  operator-facing no-secret entrypoint. It aggregates actual product readiness,
  the local product activation rehearsal, collaboration readiness, provider
  runtime, durable job-store migration manifest, quality readiness, and the
  generated OpenAPI contract summary into `local_foundation`, `small_group`,
  `community`, and `full_activation` gates when run through the CLI/API/UI
  entrypoints. The suite
  intentionally omits raw child reports, so tokens, local paths, job payloads,
  artifact URIs, prompts, answers, and external account details are not exported
  through the aggregate JSON/Markdown report. It also projects the next
  no-secret quality evidence target, evidence source, and gap list from the
  readiness summary, giving operators the immediate small-group or community
  evidence delta without exposing raw reports. The same report now includes a
  full activation action plan that groups product-readiness, collaboration
  activation, provider activation, platform migration activation, and community
  quality blockers into placeholder commands plus verification commands,
  without enabling external services or exporting concrete URLs, paths, prompts,
  answers, source content, or credentials.
- `GET /admin/activation-suite`, `GET /admin/activation-suite/report`, and the
  Streamlit admin panel expose the activation suite as an explicit on-demand
  check. It is not folded into every `/admin/status` refresh because it runs
  local rehearsals; operators choose when to run it and can download the
  no-secret Markdown report. The POST API variants and Streamlit uploader can
  accept an `evaluate_rag.py --json-report` payload as in-memory live evidence,
  but the response keeps only no-secret maturity counts, pass rates, and
  coverage summaries instead of echoing the report, filenames, paths, prompts,
  answers, or source content.
- `scripts/openapi_contract.py` gives frontend/API split work a no-secret local
  contract preflight. It checks the generated FastAPI OpenAPI schema for
  required route/method coverage, operation summaries and IDs, response
  declarations, protected-route auth header declarations, and route-group
  coverage while keeping the raw schema out of JSON/Markdown readiness reports.
  The report now includes a stable operation fingerprint and
  `--verify-snapshot ... --require-no-drift` can compare the current no-secret
  report with a prior no-secret JSON snapshot for contract drift. The verifier
  projects only whitelisted counts, booleans, fingerprints, and valid flags, so
  malformed or raw-schema-shaped snapshot input is not echoed into JSON/Markdown
  output. Snapshot count fields must be non-negative bounded JSON integers, so
  stringified, negative, or extreme count values are treated as invalid shape
  instead of being echoed. `GET /admin/openapi-contract`,
  `GET /admin/openapi-contract/report`, `POST /admin/openapi-contract/verify`,
  `POST /admin/openapi-contract/verify/report`, and the Streamlit admin panel
  expose the same checks for explicit operator runs.
- `src.provider_guard` gives RAG/provider calls a reusable no-secret
  quota/cost guard. It is disabled by default, but when
  `PROVIDER_QUOTA_GUARD_ENABLED=true` it checks estimated prompt tokens,
  requested completion tokens, and an optional cost ceiling before provider
  clients are constructed. Its decisions contain only counts, thresholds, and
  reason codes.
- Admin status/report, metrics, and the Streamlit status panel display a
  no-secret `platform_readiness` summary for production storage migration and
  distributed worker acceptance. It reports only booleans, counts, and blocker
  codes; the current local runtime shows schema/inventory and local worker
  bridge readiness while keeping production activation blocked on external
  metadata database, object storage, and distributed job-store configuration.
- Admin status/report, metrics, and the Streamlit status panel display a
  no-secret `provider_readiness` summary for external provider activation. It
  reports only safe backend names, booleans, and blocker codes; the current
  local runtime keeps external image providers, hosted execution, MATLAB, and
  provider quota/cost guards disabled.
- `GET /admin/runtime-manifest`, `GET /admin/runtime-manifest/report`, and the
  Streamlit runtime status panel expose a no-secret runtime backup manifest for
  the state trees that source deploys exclude.
- `POST /admin/runtime-manifest/restore-check` and its Markdown report route
  check a supplied no-secret manifest against local runtime state without
  restoring, copying, or deleting files. The Streamlit status panel exposes the
  same check for an uploaded manifest JSON.
- `GET /admin/status/report` and the Streamlit status panel can export the same
  no-secret status snapshot as a Markdown operations report. Status/report
  latest-event summaries sanitize runtime-event metadata and expose only
  `request_id_present` booleans instead of request IDs.
- `GET /admin/metrics` and the Streamlit status panel export the same local
  admin summaries as Prometheus/OpenMetrics-style text. The metrics are
  metadata-only local-window gauges and avoid owner IDs, request IDs, paths,
  prompts, answers, uploaded content, filenames, and artifact contents.
- Successful `/query`, `/query/inspect`, `/query/report`, and
  `/query/retrieve` calls emit metadata-only `retrieval_trace` runtime events
  with endpoint, answer mode, context count, source/page completeness counts,
  citation status when available, duration, and whether an LLM provider was
  called. Admin status/report, Streamlit, and the metrics export summarize the
  events without prompts, answers, retrieved text, source paths, owner IDs, or
  request IDs.
- Retrieval trace summaries also include local advisory alerts for empty
  retrievals, missing source/page metadata, and citation validation failures,
  controlled by `RETRIEVAL_TRACE_ALERT_MIN_EVENTS`,
  `RETRIEVAL_TRACE_ALERT_EMPTY_RATE`,
  `RETRIEVAL_TRACE_ALERT_SOURCE_PAGE_INCOMPLETE_RATE`, and
  `RETRIEVAL_TRACE_ALERT_CITATION_FAILURE_RATE`.
- `GET /admin/retention` returns the default preview of upload/artifact files
  matching local age-based retention thresholds.
- `POST /admin/retention/delete` deletes the same bounded local candidate set
  only when `RETENTION_DELETE_ENABLED` is explicitly true; disabled mode returns
  a guarded no-delete result. Deletion excludes artifact SQLite metadata files
  and symlinks, rechecks candidates as regular files before unlinking, and emits
  only aggregate `retention_delete` runtime-event counts.
- The Streamlit status panel exposes the same retention preview with local
  upload/artifact day thresholds and a candidate limit, and only shows the
  delete action when the same config flag is enabled.
- `GET /admin/events` lists no-secret runtime events with local `kind`, `code`,
  and `q` filters; the Streamlit status panel exposes the same event viewer.
  Both event viewers sanitize event metadata and apply search to the sanitized
  projection rather than raw JSONL. Code-execution outcomes are emitted as
  `code_execution` events and summarized in admin status/report. The
  explicit admin readiness/rehearsal routes emit metadata-only `admin_check`
  events when API access auditing is enabled; those events contain check names,
  ok/blocked state, counts, booleans, and blocker counts, not uploaded
  snapshots, raw reports, fingerprints, paths, prompts, answers, tokens, or
  child payloads. Admin status/report, metrics, and Streamlit summarize those
  events by check name, code, ok/blocked state, and blocker-count totals; latest
  admin-check events use a fixed-key projection so malformed legacy metadata is
  not exposed. Unsafe legacy admin-check code/check labels are grouped as
  `invalid`, and negative blocker counts are clamped before totals are emitted.
  Malformed runtime-event JSONL lines are skipped with warnings instead of
  breaking the event viewer.
- FastAPI middleware emits metadata-only `api_access` runtime events controlled
  by `API_ACCESS_AUDIT_ENABLED`. These classify token status as
  `not_configured`, `valid`, `missing`, or `invalid` and record method, path,
  status code, duration, credential type, and request ID when present. They do
  not copy token values, headers, request bodies, prompts, answers, client IPs,
  or uploaded/runtime file contents. Admin status/report and the Streamlit
  status panel summarize recent API access by token status, HTTP status code,
  and method.
- `API_RATE_LIMIT_ENABLED` can enable a local in-memory API rate-limit guard
  using `API_RATE_LIMIT_MAX_REQUESTS` over `API_RATE_LIMIT_WINDOW_S`. Over-limit
  requests return HTTP 429 before route handling and emit only metadata-only
  `api_access` rate-limit fields plus `X-RateLimit-*` response headers. Admin
  status/report and the Streamlit status panel summarize recent rate-limited
  access counts and the configured local threshold. This is not identity-backed
  quotas, billing, or distributed rate limiting.
- `POST /query/report` exports an answer, citation validation, and retrieved
  context refs as a Markdown research report. For implementation and
  code-generation reports it adds a paper-to-code handoff with source refs,
  assumption/parameter guardrails, fenced code blocks, cited artifact IDs, and
  validation checklist fields.
- `/query` provider failures are appended to a no-secret local JSONL event log
  under `metadata/` and summarized by `GET /admin/status`. Admin status/report
  now also expose metadata-only local provider-failure advisory alerts
  controlled by `PROVIDER_FAILURE_ALERT_MIN_EVENTS` and
  `PROVIDER_FAILURE_ALERT_RATE`.
- Provider quota/cost guard denials use separate metadata-only
  `provider_quota_guard` runtime events and are not counted as provider
  failures, so local over-limit enforcement does not pollute provider failure
  rate alerts.
- Provider quota/cost guard, query-cost parsing, and artifact public cost
  metadata reject non-finite or extreme-exponent cost/rate configuration as
  invalid local configuration, keeping status/readiness/artifact output bounded
  and no-secret.
- Successful `/query`, `/query/inspect`, and `/query/report` calls append
  no-secret estimated usage events with duration, character counts, and rough
  token estimates. When provider responses include token usage, the same local
  events also store sanitized non-negative provider prompt/completion/total
  token counts and admin status aggregates them separately from estimates.
  Malformed provider usage metadata is ignored or falls back to the next valid
  field instead of failing the completed query. Admin status/report
  summarize recent average and max query duration plus metadata-only local query
  latency advisory alerts controlled by `QUERY_ALERT_MIN_EVENTS` and
  `QUERY_ALERT_DURATION_MS`.
- Admin status and the Streamlit status panel expose optional no-secret
  provider/model pricing configuration through `QUERY_COST_PROVIDER`,
  `QUERY_COST_PROMPT_USD_PER_1M`, and
  `QUERY_COST_COMPLETION_USD_PER_1M`. When rates are configured, FluxMind
  estimates USD query cost from provider token counts when available and rough
  estimated tokens otherwise; external billing remains disabled.
- Admin status/report, metrics, and the Streamlit status panel expose the same
  no-secret product-readiness summary. It reports `local_foundation_ready=true`
  in the default local runtime and keeps `activation_ready=false` until product
  activation blockers are cleared.
- Admin status/report, metrics, and the Streamlit status panel expose the same
  no-secret provider-readiness summary. It reports `local_foundation_ready=true`
  in the default local runtime and keeps `activation_ready=false` until external
  provider, hosted execution, MATLAB, and quota/cost guard blockers are cleared.
- Still planned: production durable storage dashboards beyond the local
  inventory/readiness/platform-readiness and local rehearsal views, real
  production storage and distributed worker migration execution after backend
  choice, external job-store activation behind the existing readiness target,
  identity-backed deletion/audit controls, external billing/payment activation,
  production scrape/alert routing beyond the local metrics text, and a real
  product frontend/team admin backed by external identity.

Acceptance:

- Multiple corpora can coexist without leaking documents or generated artifacts.
- User-facing workflows are not tied to local server filesystem assumptions.
- Operational state is inspectable without SSH for common local runtime
  questions.
- No-secret local metrics can be exported without scraping logs or exposing
  prompts, answers, request IDs, owner IDs, tokens, paths, or runtime contents.
- Retrieval trace summaries can be inspected without reading retrieved chunks,
  source filenames, prompts, answers, owner IDs, or request IDs.
- Durable storage readiness is visible in the UI without activating external
  storage accounts.
- Local storage inventory is visible in the UI without reading file contents or
  activating external storage accounts.
- Local storage-schema readiness is visible in API/report/metrics/UI surfaces
  without reading stored content or exposing identifiers.
- Production storage and distributed-worker blockers are visible in
  API/report/metrics/UI surfaces without connecting to external services or
  exposing runtime contents.
- Object-storage upload manifests can be generated from a migration rehearsal
  without activating external storage or exposing source paths, filenames,
  buckets, endpoints, credentials, or runtime contents.
- Object-storage upload manifests can be verified against local/staged runtime
  state without activating external storage or exposing source paths, filenames,
  buckets, endpoints, credentials, or runtime contents.
- Local retention candidates can be previewed without deleting files or reading
  raw runtime directories by hand.
- Local retention candidates can be deleted only through the explicit guarded
  switch and authenticated API/UI path, with aggregate no-secret event evidence.
