# Decision

Date: 2026-11-19 JST. Run `pcqm-grit-spse-100k-20261118-r2-s42` (attempt `v1`), seed 42, 100K split `pcqm_fixed_100k_v1`.

## Outcome

- `pair_update_norm` (same-job control): **CLOSED**.
- `grit_spse`: **NEGATIVE_UNDER_CONTRACT**.

Both arms finished 60 of 60 passes and were retrieved in full.

## Result

Selected EMA development MAE, 50,000 rows:

| Arm | MAE (meV) |
| --- | --- |
| `pair_update_norm` | 139.44 |
| `grit_spse` | 153.95 |

Candidate minus control is **+14.51 meV** (paired bootstrap 95% interval 13.33 to 15.67 meV). The chance that the candidate is better is 0.0. The frozen gate needed -3.0 meV with an upper bound below 0. The candidate is worse than the control and misses the gate by about 14.51 meV.

## Strict comparison status

Strict comparison is not ready: SCIENTIFIC_CONTRACT_MISMATCH on runtime_certificate_scope only (reference c7501151…, candidate f0b18183…). All recipe, data, row, target and seed fields match; missing fields are empty. This arm closes through the non-strict mechanical path. NEGATIVE_UNDER_CONTRACT here means the frozen endpoint gate failed. It is not causal evidence about the architecture, and the arm does not enter the strict replay or backtest pool.

The runtime gap is unlikely to explain +14 meV: every r2 candidate (demol_cin, cutlass_gt, grit_spse, tb_eigengnn) carries the same candidate hash, demol_cin was 4.8 meV better than the same control under it, and 100K init spread is about 1.6 meV. This is inference; the retained records hold only the hash.

## Failure attribution

- Mode: `endpoint_regression` at the 100K endpoint, against the same-job control.
- Cause: `insufficient_evidence`. The run holds one seed and one endpoint pair. It cannot separate a weaker design from slow convergence or a poor setting.
- Read this as: this recipe at 100K did not pass the gate. It is not proof that GRIT-SPSE is worse in general.
- Missing discriminator: a second init, or a longer or wider run of the same recipe.

## Next step

No 500K promotion from this result. The 500K package for this idea stays unsubmitted unless a new decision gives a reason.
