# Batch 1 historical trace recovery

This recovery starts from Desktop `de05855a45507c7330e6d3225731732ac6bff140`,
which retains the evidence-chain commit following the requested `b46e755` base.
It changes no historical scientific conclusion and grants no new execution,
scale-up, role-access, or policy authority.

The deterministic adapter is `experiments/_scripts/recover_desktop_batch1.py`.
It only reads retained metadata and local Git objects and writes this fixed
batch. It does not load checkpoints, predictions, labels, or a model. Its
canonical serializer performs the shape checks required to emit a trace; no
pytest, repository validation, RML acceptance, rebuild, or backtest was run.
Luna owns corpus regeneration and acceptance.

## Selected evidence

| Trajectory | Recovery | Scope |
|---|---|---|
| TB-gptrans-t-100k-v4-reference | 60 original trace rows | Historical reference |
| TC-gptrans-flow-ablation-100k-s42 | no_pair_to_node arm, 60 original trace rows | Historical candidate, original server ownership retained |
| TB-gptrans-t-500k-bridge | Embedded completion trace | CONTEXT_ONLY / scale |
| TB-gine-1m-gap-specialist-v7 | Retained training CSV | CONTEXT_ONLY / scale |

`batch1_report.json` inventories original source paths, exact byte SHA256,
retained copies, field mappings and unknowns. Each experiment has its own
`historical_recovery/migration.json`, canonical trace, source snapshot and
source-semantics excerpts. Each root `trace_manifest.json` binds its canonical
trace SHA and existing V5 evidence ID. The sidecars also bind the retained
acceptance/completion metadata. Source-byte snapshots use `-text` attributes to
avoid Git newline conversion invalidating provenance. Existing V5 envelopes
are unchanged; the raw reference trace was not an artifact in its old envelope,
so this sidecar records a new local content hash, not a fabricated old hash.

## Why cumulative 100K coordinates are justified

The exact reference source commit is `814d10466d98a583117a20d65329ff9f27836203`;
the candidate source is `fc329e0948810e5649823ab64e834c9b5d36b47a`.
Both retained implementations call `_optimizer_step` once per batch, perform
`optimizer.step()` inside it, count the observed graphs, and require the count
to equal the declared epoch exposure before appending a row. Resume restores
the saved trace and starts at the following epoch. Thus each stored
`optimizer_steps` and `sample_presentations` value is an executed epoch
increment. Recovery sums those recorded values in source order. It does not
multiply an epoch by a planned batch count. The sums reconcile with retained
completion totals of 46,860 steps and 5,998,080 presentations.

`train_mae_eV` is the live training metric; `development_mae_eV` is evaluated
with EMA weights and is mapped to `ema_dev_metric`, never `live_dev_metric`.
The original zero-based epoch, learning rate and per-epoch elapsed wall interval
are preserved. Device time, cumulative timing and per-observation checkpoint
identity remain null. No best-so-far series replaces an observed metric.

## Comparable-world scope

The candidate's retained acceptance binds the same immutable reference model,
dataset/rows/features/target, seed, precision, optimizer, schedule, loss,
transform, EMA selection, role and exposure. These shared fields were also
read from both retained frozen-reference records during generation. Its
contract explicitly inherits the GPTrans 100K parent contract. Architecture
identity intentionally differs: removal of direct pair-to-node readback was
the accepted intervention, so this pairing does not require identical models.
Both manifests carry the same scientific comparison identity and exact
reference evidence ID. The reference binds its own accepted terminal result;
the candidate retains its original state-at-start reference binding.

The selected canonical records are prepared for one candidate/reference replay
world. No rebuild was run, so an actual generated group count is not claimed.
All four records remain retrospective/historical partial. The old candidate
V5 scientific status uses lowercase `negative_under_contract`; the current
policy label mapping requires uppercase. That envelope remains byte-identical,
so policy confusion metrics may exclude this entry even when the comparable
trace world exists. No policy is installed or activated to hide that limitation.

The K1 induced-pair candidate uses the K1 optimization/selection/exposure
contract and was not selected. The 500K and GINE records are explicitly
ineligible for strict replay; missing observed step/sample axes remain null.
No 100K-to-500K architecture-causal interpretation is introduced.

## Historical import and unknowns

Only the flow candidate's contract, accepted comparison, terminal decision and
V5 envelope were taken from local Git commit
`76c8fd3ed713774cc155477e925fbce97b1418cb`; no branch was merged. The original
prospective trajectory is retained as `original_trajectory.json`, outside the
canonical discovery filename. Its Desktop corpus projection is explicitly
`retrospective_partial`, keeps `owner=server`, and retains the negative outcome.
This evidence import does not hand off an active experiment or merge corpora.

No new role/cost event is created. In particular, the original two-arm job cost
is not allocated to the selected arm or converted from elapsed wall time into
device time. Omitted cost-event links are documented in the imported trajectory;
the original remains available in the source snapshot. Missing is not zero,
and no absent role event is interpreted as untouched.

GINE's CSV dev series is the 82,240-row internal scaffold-dev metric, not the
5,000-row official-validation metric present in historical acceptance metadata.
Only retained aggregate metadata is read. Exact original GINE source identity
is unknown; retained implementation excerpts explain field meanings without
asserting that the current source was the historical executable. The CSV lacks
step/sample counts, and neither dataset sizes nor schedules fill those gaps.

Matched-500K, recurrent-100K and geometry-transfer traces are outside this batch.
No remote retrieval, training, inference or protected-role evaluation occurred.
