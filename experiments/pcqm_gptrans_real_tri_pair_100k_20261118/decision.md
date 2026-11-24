# Decision

2026-11-18 JST. Run `pcqm-gptrans-real-tri-pair-100k-20261118-s42`, attempt `v1` (`nvoid912/molgap-gptrans-real-tri-pair-100k-20261118-s42`, `version=1`, commit `b1d2af4486a3c869a1985e39be0d4e7df7fed7b6`, spec `b06317872947a2a3afac4fc53038fa9688df23bc80fbae76b2cf1ff925e08c81`), strictly retrieved and diagnosed.

## Facts
- Contract & validity: Both arms completed 60 passes (46,860 optimizer steps, 5,998,080 sample presentations) and are `MECHANICALLY_VERIFIED`. Python main-process open provenance (`data_open_provenance.json`) shows `official_validation`, `test_dev`, and `test_challenge` remain `untouched` for both arms.
- Metric: selected EMA (`0.999`) MAE on the 50,000-row `internal_development` split of `pcqm_fixed_100k_v1`.
- `pair_update_norm` (same-job control):
  - Selected EMA `0.999` dev MAE: `139.440` meV (`0.139440` eV, pass 42, step 32,802); pass 60 EMA dev MAE: `139.867` meV.
  - Aux EMA `0.9999` dev MAE: `143.367` meV (`0.143367` eV).
  - Online train MAE: `90.379` meV at selected pass 42; `70.450` meV at pass 60.
  - Process wall: `11,323.3` s (`3.145` h) on Tesla T4 (`device_seconds` missing, requires platform allocation accounting).
- `real_tri_pair` (candidate):
  - Selected EMA `0.999` dev MAE: `137.773` meV (`0.137773` eV, pass 43, step 33,583); pass 60 EMA dev MAE: `138.790` meV.
  - Aux EMA `0.9999` dev MAE: `142.011` meV (`0.142011` eV).
  - Online train MAE: `85.594` meV at selected pass 43; `67.511` meV at pass 60.
  - Process wall: `13,515.8` s (`3.754` h) on Tesla T4 (+19.4% wall ratio `1.194`, +2,192.5 s; `device_seconds` missing).
- Paired comparison (`real_tri_pair - pair_update_norm`, 50,000 rows, 2000-bootstrap seed 42):
  - Effect: `-1.667` meV (`-0.001667` eV).
  - 95% bootstrap CI: `[-2.615, -0.749]` meV.
  - `P(candidate better)`: `1.0`; row win fraction: `0.5028` (25,140 vs 24,860).
- One seed (42), one init per arm.

## Frozen Gate (`protocol.md`)
Candidate minus control must satisfy `<= -3.0` meV with 95% CI upper bound `< 0`. Observed `-1.667` meV with 95% CI `[-2.615, -0.749]` meV: the upper bound is strictly `< 0`, but the point improvement does not reach the `-3.0` meV material gate.

## Outcome
- `pair_update_norm`: `CLOSED` (`failure_mode: null`).
- `real_tri_pair`: `POSITIVE_BELOW_GATE` (`failure_mode: "below_gate"`).

## Attribution
- Trajectory: `real_tri_pair` led in 58/60 paired evaluations, with strong early/mid gains (passes 1-10 mean delta `-3.595` meV; passes 11-20 `-4.628` meV) that faded in late training (last 10 passes `-1.050` meV; last 5 passes `-1.000` meV; `late_horizon_fade: true`).
- Failure-mode note (`below_gate`): Real-atom asymmetric low-rank triangle multiplicative updates at layers 2, 4, 6, 8, 10 genuinely improved 100K dev MAE by `-1.667` meV (CI excludes zero) while lowering online train MAE (`67.511` vs `70.450` meV at pass 60) and slightly widening the pass-60 dev-minus-train gap (`71.278` vs `69.417` meV). At 100K, the early `-4.6` meV advantage partially faded into late-horizon overfitting and did not clear the `-3.0` meV material gate at +19.4% T4 wall cost.

## Scope
This result covers one seed at 100K on `internal_development`. It does not evaluate larger data scales, other seeds, or the sealed `official_validation` and test roles.
