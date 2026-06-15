# FluxMind Quality Roadmap

Last updated: 2026-06-15

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
small_group  gap: corpus size and eval breadth remain; live retrieval evidence
             is now 50/50 passing in the latest deployment run
community    gap: mainly corpus size, live answer evidence, and coverage depth
```

Latest measured quality snapshot on 2026-06-15 09:23 CST:

```text
Metric                         Current  Small-group target  Gap
-----------------------------  -------  ------------------  ---
seed_paper_count               11       30                  19
answer_case_count              20       40                  20
retrieval_only_case_count      30       60                  30
retrieval_eval_question_count  50       100                 50
recorded_answer_count          20       40                  20
live_retrieval_result_count    50       50                  0
code_output_case_count         4        8                   4
pdf_structure_case_count       6        15                  9
topic_group_count              4        4                   0
```

## Near-Term Quality Lane

The next quality lane is intentionally content and evaluation first:

```text
Order  Work
-----  ---------------------------------------------------------------------
1      Expand curated papers from 11 toward 30, with topic tags and provenance.
2      Add retrieval-only and recorded-answer cases alongside each useful paper.
3      Run live /query/retrieve before trusting live /query/inspect answers.
4      Add PDF structure cases only when the source paper actually has useful
       equation, table, figure, or algorithm anchors.
5      Convert repeated strong cases into paper-to-code templates or examples.
```

The small-group target should be reached before starting identity, quotas,
billing, or a frontend rewrite as a primary lane.

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
