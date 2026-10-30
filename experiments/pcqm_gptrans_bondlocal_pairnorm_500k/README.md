# BondLocal plus pair-update normalization at 500K

Desktop owner: `codex/exp/gptrans-bondlocal-pairnorm-500k`, base `0945f0e5`.
This question concerns normalization of each fresh pair update, not accumulated
pair-state normalization (`pair_post_norm`). No training is released.

## Retained-output verification, 2026-10-28

The user authorized a CPU-only reuse audit of the two existing 500K outputs.
The [prospective record](trajectory.json) preceded prediction decoding.
[Actual observations](reuse_report.json) own hashes, dtype, metrics and runtime
scope; [the thin runner](verify_retained.py) reproduces the scoped inspection.

| Retained arm | Job/version | Recomputed internal-development MAE (eV) |
|---|---|---:|
| BondLocal + EMA | 138717853/v1 | 0.101951375613 |
| BondLocal + pair-update norm + EMA, LR 1e-3 | 141037251/v1 | 0.099987377815 |

Both prediction SHA256 pins match. Both contain the same 50,000 unique ordered
internal selection rows, 500000 through 549999, and finite predictions/targets.
Both stored targets are float32 and their values are exactly equal. The earlier
target-hash difference came from float64 versus float32 digest encoding, not
different values or different stored tensor dtypes. Neither original was changed.

The observed MAE difference is -0.001963997798 eV, a descriptive comparison on
selection-used development, not an isolated causal PairNorm gain. Both runtime
provenance objects match their certificate's canonical provenance digest.

## Reuse disposition

- **Saved-prediction analysis: reusable.** Preserve existing custody locations;
  do not copy weights or predictions into this checkout.
- **Strict causal training comparison: not qualified.** Runtime scopes differ
  (`6c704d88` / `60677044`) and frozen execution implementations differ. Existing
  speed evidence does not bridge these exact sources. Shared readiness records
  the observed mismatch; other unassessed fields stay missing.
- **Checkpoint/resume reuse: not certified here.** No model weights were loaded.
  Old original resume bytes were retired; prediction verification does not
  certify a replacement tensor-only endpoint as an original resumable checkpoint.

No reference retraining follows from this result. Before any training proposal,
resolve cross-run reference conditions and obtain separate platform/budget
authorization. Do not automatically rerun either arm or change comparator policy.
The prospective remains open for that unresolved reuse decision; this audit is
not a newly finalized training trajectory or replay-ready comparison.

## Evidence and implementation owners

- [100K norm comparison](../pcqm_gptrans_bondlocal_pairnorm_100k/terminal_acceptance/decision.md):
  0.142861 to 0.139718 eV; no automatic 500K adoption.
- [Earlier PairNorm-only 500K](../pcqm_gptrans_pair_norm_500k/decision.md):
  no BondLocal; below its material gate, not this conditional comparison.
- [500K learning-rate decision](../pcqm_gptrans_bondlocal_pairnorm_lr_500k/terminal_acceptance/decision.md):
  both arms include norm; that learning-rate decision is unchanged.
- [New-arm custody](../pcqm_gptrans_bondlocal_pairnorm_lr_500k/terminal_custody.md),
  [old retirement index](../../docs/operations/LOCAL_RETIREMENT_INDEX.json),
  [execution refinement](../../docs/operations/GPTRANS_EXECUTION_REFINEMENT.md).

The runner reuses shared prediction inspection, target encoding bindings, hashes,
trajectory-schema validation and comparison readiness. It never calls full-output
inspection, which would deserialize checkpoints. A generic planning helper needs
a registered policy; this bounded audit validates its prospective schema instead
of inventing a policy. No shared infrastructure was changed.

Checks: real retained CPU inspection passed; no training, model inference,
official/protected-role access, remote submission or model adoption. Initial
adapter errors (contract versus manifest context; raw versus canonical provenance
digest) were corrected before the successful final run.
