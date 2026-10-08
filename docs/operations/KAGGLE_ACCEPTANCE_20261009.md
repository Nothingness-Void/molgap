# Kaggle reconciliation navigation - 2026-10-09

User requested acceptance of retained desktop Kaggle work. Exact owning-account
SDK queries, source/version reconciliation and selective artifact retrieval were
performed once; no training, inference, calibration, retry or successor submission.
The owner checkouts retain raw scheduler/source/inventory and retrieval evidence.

| Owner | Authoritative local acceptance | Remaining blocker |
|---|---|---|
| Consistency500K | [Version2 bounded acceptance](D:/w/k1-consistency-500k/experiments/pcqm_k1_consistency_ablation_500k/submission_v2/terminal_inspection_20261009/stage_acceptance.md) |14epochs per arm plus identical selected-state clean-BN and independent terminal closure |
| Spectral100K | [Completed paired endpoint inspection](D:/w/k1-spectral-night/experiments/pcqm_k1_spectral_100k/reconciliation_20261009/acceptance.md), [attribution](D:/w/k1-spectral-night/experiments/pcqm_k1_spectral_100k/reconciliation_20261009/attribution.md) | Frozen material gate fails; per-arm terminal packages and fresh-pair versus historical-reference binding unresolved |
| EMA100K | [Infrastructure reconciliation](D:/w/k1-ema-night/experiments/pcqm_k1_weight_ema/reconciliation_20261009/acceptance.md) |ERROR with no retained training/failure/cost discriminator |

These are retained-worktree locators, not portable artifact provenance.
[BRANCHES](../../BRANCHES.md#retained-checkout-map) and each remote handoff own
the matching Git refs. Pending experiments stay on their owners; no wholesale
implementation merge or temporary-ref deletion is warranted yet. Root state
was updated to avoid stale RUNNING observations, not to replace evidence.

Reuse: explicit-account credential adapter, family selective-output retriever,
retained-resume selector, launch RunContext, pinned target-encoding binding,
family output inspector, K1 native-preflight validator, paired-bootstrap helper
and existing finite-state/runtime primitives. No acceptance framework was added.
New artifacts are metadata reports and retained output snapshots only.

No final RML receipt, strict replay qualification or model promotion was produced
by this reconciliation. In particular, the family terminal interface consumes
caller-owned scientific terminal packages; missing packages are not permission
to fabricate reference/role/cost/comparability evidence or copy another validator.
No automatic continuation/heartbeat or server takeover was installed.

## Delivery checks

Focused synthetic regression/navigation tests:170passed (25retention/consistency,
89family/K1screen,56navigation); dependency deprecation warnings only.
Existing desktop `check --frozen --portable` passed. This checks the already
indexed desktop evidence, not finalization of these still-pending owner records.
Owner evidence is pushed in [consistency981c0753](https://github.com/Nothingness-Void/molgap/commit/981c0753),
[spectrald8271477](https://github.com/Nothingness-Void/molgap/commit/d8271477)
and [EMA667b68dc](https://github.com/Nothingness-Void/molgap/commit/667b68dc).
Ignored prediction/checkpoint binaries remain retained locally and in the
observed Kaggle outputs; they were not added to Git or claimed Git-portable.
