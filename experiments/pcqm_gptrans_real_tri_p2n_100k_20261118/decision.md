# Decision

2026-11-18 JST. Run `pcqm-gptrans-real-tri-p2n-100k-20261118-s42`, attempt `v1`, strictly retrieved and diagnosed.

## Facts
- Contract & validity: both arms (`pair_update_norm`, `real_tri_p2n`) completed 60 passes (46,860 optimizer steps, 5,998,080 sample presentations) on `pcqm_fixed_100k_v1` from identical `gptrans.pt` initialization (`aaeced3b845dc34ea931f13a6cb52438324d35b9`) and passed mechanical and strict-causal verification (`STRICT_CAUSAL`).
- Role audit: `official_validation`, `test_dev`, and `test_challenge` remained `untouched` (`data_open_provenance.json` verified).
- Metric: selected EMA `0.999` MAE on the 50,000-row `internal_development` split of `pcqm_fixed_100k_v1`.
- `pair_update_norm` (same-job control):
  - Selected EMA `0.999` dev MAE: **139.440 meV** (pass 42, step 32,802); final pass-60 EMA `0.999` dev MAE: **139.867 meV**.
  - Aux EMA `0.9999` dev MAE: **143.367 meV** (step 46,860).
  - Online train MAE: **90.379 meV** at selected step, **70.450 meV** at final step.
  - Measured process wall time: **9,818.28 s** (~2.73 h on Tesla T4; `device_seconds` missing).
- `real_tri_p2n` (candidate):
  - Selected EMA `0.999` dev MAE: **136.852 meV** (pass 40, step 31,240); final pass-60 EMA `0.999` dev MAE: **138.010 meV**.
  - Aux EMA `0.9999` dev MAE: **139.952 meV** (step 37,488).
  - Online train MAE: **90.142 meV** at selected step, **64.835 meV** at final step.
  - Measured process wall time: **12,759.96 s** (~3.54 h on Tesla T4; `device_seconds` missing).
- Paired comparison (`real_tri_p2n - pair_update_norm` on 50,000 aligned rows):
  - Effect: **-2.588 meV** (`-0.002588 eV`).
  - Paired 2000-bootstrap 95% CI: **[-3.491, -1.670] meV**.
  - Probability candidate better: **1.0000**.
- One seed (42), one init per arm.

## Frozen Gate (`protocol.md`)
Candidate minus control must be `<= -3.0 meV` with 95% upper bound `< 0`. Observed `-2.588 meV` with 95% upper bound `-1.670 meV` (`ci_meV[1] < 0`, but point estimate `-2.588 > -3.0 meV`). The `-3.0 meV` material gate is not met.

## Outcome
- `pair_update_norm`: `CLOSED` (`failure_mode: null`).
- `real_tri_p2n`: `POSITIVE_BELOW_GATE` (`failure_mode: below_gate`).

## Terminal Attribution
- **Convergence trajectory**: `real_tri_p2n` led `pair_update_norm` in every pass window: `P1-10 = -6.283 meV`, `P11-20 = -5.876 meV`, `P21-40 = -4.133 meV`, and `P41-60 = -2.009 meV` (`late_horizon_fade=true`).
- **Mechanism**: Adding pre-GPA triangle + real-atom `pair->node` vector feedback (excluding virtual node 0) improved over both `pair_update_norm` (`-2.588 meV`) and `real_tri_pair` (`-1.667 meV`, a further `-0.921 meV` gain) and produced `5.19 meV` oracle blend complementarity (`131.66 meV`), but late-horizon overfitting (`online_train_mae` dropping to `64.835 meV` vs `70.450 meV` for control; `best_minus_final = -1.158 meV`) compressed the final-pass margin to `-1.857 meV` and kept the selected EMA gain `0.412 meV` shy of the `-3.0 meV` gate.
- **Closeout**: Closed as `POSITIVE_BELOW_GATE` without hyperparameter grids or automatic retries.

## Scope
This result covers one seed (42) at 100K on the internal development set. Official validation and test roles remain unread.
