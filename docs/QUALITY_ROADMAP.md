# FluxMind Quality Roadmap

Last updated: 2026-06-16

For current repo state, use `docs/REPO_STATUS.md`. For live deployment state,
use `docs/DEPLOYMENT_STATUS.md` and re-run the health checks before making
deployment claims. This document owns the staged quality bar for moving
FluxMind from personal use to small-group use and then to community use.

## Quality Principle

FluxMind should advance by trust, not by feature count. New UI surfaces,
provider keys, hosted execution, MATLAB licensing, accounts, quotas, or billing
should wait until the paper-grounded research workflow is measurably useful.

The product wedge is:

```text
Paper-grounded control-engineering work product:
traceable answers, source/page evidence, executable modeling snippets,
reproducible plots, and implementation notes.
```

## Staged Quality Bars

The executable target definitions live in `eval/rag_baseline.json` under
`quality_maturity_targets`. `scripts/evaluate_rag.py --json-report ...` exports
the current metrics and target gaps under `quality_maturity`.

```text
Stage        Product meaning                 Main bar
-----------  ------------------------------  --------------------------------
self_use     Personal research assistant      Current no-key baseline stays
                                               green and repeatable.
small_group  Lab or small-team pilot          Broader corpus plus live
                                               retrieval evidence before
                                               account/platform investment.
community    Public community release         Broad curated content, live answer
                                               evidence, and deeper
                                               paper-to-code/layout coverage.
```

Current status:

```text
Target       Status
-----------  ---------------------------------------------------------------
self_use     met by the current no-key/local baseline
small_group  met: 30 curated papers, 100 no-LLM retrieval questions,
             40 recorded answers, 11 code-output cases, 17 PDF structure
             cases, and 100/100 live retrieval pass in the latest deployment run
community    gap: mainly corpus size, live answer evidence, and coverage depth
```

Latest measured deployed quality snapshot on 2026-06-16 14:17 CST:

```text
Metric                         Current  Small-group target  Gap
-----------------------------  -------  ------------------  ---
seed_paper_count               30       30                  0
answer_case_count              40       40                  0
retrieval_only_case_count      60       60                  0
retrieval_eval_question_count  100      100                 0
recorded_answer_count          40       40                  0
live_retrieval_result_count    100      50                  0
code_output_case_count         11       8                   0
pdf_structure_case_count       17       15                  0
topic_group_count              4        4                   0
```

The offline rows come from `/tmp/fluxmind-status-refresh-local-report.json`; live
retrieval comes from `/tmp/fluxmind-live-corpus30-report.json` on the deployed
host and is recorded in `docs/DEPLOYMENT_STATUS.md`.

## Near-Term Quality Lane

The small-group lane is complete. The next quality lane should stay evidence-led:

```text
Order  Work
-----  ---------------------------------------------------------------------
1      Choose whether the next milestone is community-quality evidence or the
       production storage/distributed-worker foundation.
2      For community quality, expand from 30 toward 50 curated papers, 80
       recorded answers, 180 retrieval questions, and live answer evidence.
3      For platform foundation, migrate metadata/object/job state behind a
       durable backend while preserving the no-secret local contracts.
4      Keep running live /query/retrieve before trusting live /query/inspect
       answers.
5      Add PDF structure cases only when the source paper has useful equation,
       table, figure, or algorithm anchors, and convert repeated strong cases
       into paper-to-code templates or examples.
```

The community target should be reached, or the production storage/job boundary
should be made durable, before starting identity, quotas, billing, or a frontend
rewrite as a primary lane.

## Acceptance Discipline

- A new paper is not "in the product" until it has usable metadata, source URL,
  license/provenance, and at least one retrieval or answer-quality eval case.
- A new eval case must point at source/page snippets that can be verified
  locally without an LLM call.
- Live answer results should not replace offline evidence; they are an extra
  confidence layer.
- Provider activation should not be used to hide weak retrieval, weak source
  coverage, or missing domain examples.
- Quality progress should be recorded as target metrics and evidence, not only
  as prose in a status update.
