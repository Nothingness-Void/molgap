# Local RML operational reconciliation — 2026-10-01

## Scope

This was local bookkeeping maintenance, not a scientific experiment. The
review reconciled all nine ACTIVE entries in the local RML index against
their owning retained records. It did not discover other accounts, adopt
desktop work, delete files, change experiment contracts, execute models,
retry/cancel/submit jobs, or modify monitor bindings.

At 2026-09-30 18:03:33 UTC (2026-10-01 03:03:33 JST), a read-only Kaggle2
status query of `kaseichou/molgap-gptrans-author-inputs-dual-s42/1` returned
`KernelWorkerStatus.RUNNING`. The retained physical receipt bound kernel
136543794/version 1. Status did not establish training preflight acceptance,
an epoch result, or a scientific outcome.

The two retained ACTIVE trajectories were G1 degree scaling and G2 path-bond
mean input, both in that one physical dual-worker experiment group. They
remained distinct prospective scientific identities. Live authority is
[CURRENT_STATE.md](../CURRENT_STATE.md); physical identity authority is the
[submission receipt](../experiments/pcqm_gptrans_author_alignment/gpu/submission_v1.json).

## Seven terminal reconciliations

| Prospective trajectory | Retained outcome | Terminal publication |
|---|---|---|
| TC-gptrans-author-input-preparation | INFRASTRUCTURE_ONLY: CPU chunk rederivation failed | [Receipt](../experiments/pcqm_gptrans_author_alignment/preparation_rml/rml_finalized/finalization.json) |
| TC-gptrans-author-input-preparation-verification-recovery | INFRASTRUCTURE_ONLY: rejected failed-kernel mount, version 1 ERROR | [Receipt](../experiments/pcqm_gptrans_author_alignment/verification_recovery/rml_plan/rml_finalized/finalization.json) |
| TC-gptrans-author-input-preparation-verification-recovery-v2 | NO_TRAIN: independently accepted complete CPU inputs | [Receipt](../experiments/pcqm_gptrans_author_alignment/verification_recovery/rml_plan_v2/rml_finalized/finalization.json) |
| TC-k1-chem-local-post100k-audit-s42 | CLOSED: conditional audit was not released or launched | [Receipt](../experiments/pcqm_k1_chem_local_100k/audit/rml_plan/rml_finalized/finalization.json) |
| TC-k1-motif-hierarchy-100k-s42 | INFRASTRUCTURE_ONLY: version 1 device/software gate failed | [Receipt](../experiments/pcqm_motif_hierarchy_100k/gpu_rml_plan/rml_finalized/finalization.json) |
| TC-k1-motif-hierarchy-100k-s42-v2 | INFRASTRUCTURE_ONLY: version 2 device gate failed | [Receipt](../experiments/pcqm_motif_hierarchy_100k/attempt_v2/gpu_rml_plan/rml_finalized/finalization.json) |
| TC-metagin-2d-3hop-100k-s42 | INFRASTRUCTURE_ONLY: version 1 throughput gate stopped before epoch 0 | [Receipt](../experiments/pcqm_metagin_2d_100k/rml_plan/rml_finalized/finalization.json) |

Each publication used the existing `molgap.research_memory finalize` API and
its receipt-verified discovery overlay. Owning `rml_reconciliation/` records
retained dimensional V5 outcomes, exact source hashes, logical-to-physical
attempt mappings and metadata-only acceptance scope. Native worker evidence
was not manufactured. The MetaGIN preflight performed optimizer calibration;
closure did not imply that its worker had never executed a model.

All original prospective files, knowledge snapshots and existing scientific
terminals stayed unchanged. Historical status/submission text describing
pending bookkeeping was preserved as source evidence; terminal overlays and
the root live-state document superseded that pending interpretation.

## Index before and after

| Quantity | Before | After |
|---|---:|---:|
| Trajectories | 62 | 62 |
| ACTIVE trajectories | 9 | 2 |
| CLOSED | 3 | 4 |
| INFRASTRUCTURE_ONLY | 13 | 18 |
| NO_TRAIN | 8 | 9 |
| Validated V5 envelopes | 54 | 61 |
| Cost events | 82 | 82 |
| Explicit role events | 218 | 218 |
| Canonical replay entries | 21 | 21 |
| READY_FOR_DESKTOP | 0 | 0 |

The seven additional V5 envelopes describe administrative/input/failure
closure, not seven new accuracy experiments or confirmations. The unchanged
replay pool contains 20 complete candidate entries plus one historical-partial
reference. No new training replay eligibility was granted.

## Cost and role honesty

Terminal discovery replaced each relevant prospective estimated cost event
with the same event ID carrying its observed, missing or not-applicable fields;
the frozen originals remained available. CPU native wall times and MetaGIN's
full T4x2 allocation cost were translated from retained records. Approximate
motif log durations remained estimated. Unobserved allocation/device time,
CPU accounting and queue time remained missing, not zero. The unlaunched
conditional audit had no applicable worker allocation. Source CPU counts and
process-CPU seconds were not conflated with measured allocated core-hours.

Graph-only CPU preparation did not read labels according to its accepted
worker records. No formal internal-role event was inferred without complete
row-identity event metadata; the coarse internal-role declaration remained
unknown. Protected-role applicability stayed scoped to these attempts and did
not erase desktop's earlier official-validation selection history.

## Verification

The standard compiler rebuilt all derived outputs; they were not manually
edited. Repository validation and frozen-output checking passed. Original
prospective SHA256 values, receipt snapshots and the user's existing
`tests/test_research_memory.py` edit were checked for byte preservation.
Terminal re-ingestion was checked for idempotency. No large cache or model
was loaded on the local machine. See each receipt for immutable publication
hashes and each owning acceptance for remaining metadata limitations.

Focused metadata-only finalization/receipt tests passed: 37 tests in
`tests/test_terminal_trace_closure.py` and
`tests/test_research_memory_run_binding.py`, with `--noconftest`.
