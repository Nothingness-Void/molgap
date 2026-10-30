# Terminal decision

2026-10-30: terminal closure of `pcqm-single-model-total-budget-500k-s42` (attempt v1, continuation
kernel version 3) under [protocol](protocol.md) and [authority](authority.json). This file holds the
outcomes and the numbers behind them. It does not replace [decision.md](decision.md)'s no-adoption statement.

## Outcome

- `a_12x384` is `CLOSED`. It is the same-run reference arm named in `authority.json`.
- `b_16x256` is `NEGATIVE_UNDER_CONTRACT`. It is 0.71 meV better than A, below the 3 meV gate.
- Descriptive only. There is no causal winner, no adoption and no scale-up. One seed per arm.
- Strict replay qualification does not apply: the purpose is delivery_experiment (noncausal), so both arms close through the mechanical path with comparison context only; this is not a causal comparison.

## Numbers

Recomputed from the retrieved `canonical_trace.json` and `development_predictions.pt`
(50,000 development rows, Gap, eV target, both arms aligned on `source_idx` and `target_eV`).

| | A 12x384 | B 16x256 |
|---|---|---|
| Parameters | 12,478,993 | 7,700,481 |
| Best EMA dev MAE (pass) | 100.722 meV (47) | 100.008 meV (47) |
| Last EMA dev MAE (pass 60) | 101.135 meV | 100.551 meV |
| Saved-prediction MAE | 100.722 meV | 100.008 meV |

- Both arms completed 60 passes, 234,360 optimizer steps.
- A minus B is +0.714 meV (paired row bootstrap, 2000 resamples, seed 42: 95% interval [+0.221, +1.188] meV).
  The 3 meV gate is not met.
- Reference run (earlier accepted GPTrans BondLocal+PairNorm LR1e-3 500K): 99.987 meV. A is +0.735 meV and B is
  +0.021 meV from it.
- Mean of A and B predictions: 97.611 meV. That is -3.111 meV versus A (interval [-3.370, -2.854]) and -2.397 meV
  versus B (interval [-2.668, -2.148]). Error correlation between arms is 0.919. This is a two-model average, not a
  trained model and not an arm outcome.
- Bootstraps resample the 50,000 development rows. They measure row noise only, not training-seed variation, and
  each arm has one seed.

## History and cost

- Segment 1 (kernel v1, 2026-10-28) stopped at pass 33 (A) and pass 31 (B) with `STOP_FOR_COST` when its session
  limit was reached: 25,024 s wall on 2 T4.
- Segment 2 (kernel v2) failed in resume preflight after about 88 s. The target-statistics verifier compared a
  decoded standard deviation with the frozen constant by exact float equality. The value differs in its last digits
  with CPU thread count. Repair commit `a0d5ca8c` (`family_training_runner.py`) adds an absolute tolerance of 1e-12 to
  that verifier only. The training transform, recipe, data and numerics are unchanged.
- Segment 3 (kernel v3) resumed both arms from their checkpoints and completed 60 passes: 23,066 s wall on 2 T4.
- Cumulative T4 device time: the v3 allocation ledger (v1 plus v3) sums to 96,180.6 device-s = 26.72 T4-hours. The
  failed v2 segment (646.9 device-s, 0.18 h) is recorded in `continuation_authority_v3.md`, not in a ledger; adding it
  gives about 26.90 h. Both are below the 30 T4-hour authority ceiling, so `STOP_FOR_COST` and `INCONCLUSIVE` no longer
  apply to the completed arms. The ledgers mark provisioning and queue time as `measurement_missing`.

## Protected roles

Structural, not observed: protected roles were not among this run's declared mounted inputs (frozen loader blob
6c6bc503, pinned manifest, two-mount kernel). Both arms are `structural_untouched` in
`roles_v4_structural/`. This does not state that file opens were watched.
