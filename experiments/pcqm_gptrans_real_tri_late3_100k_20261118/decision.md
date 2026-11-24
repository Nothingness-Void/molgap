# Decision

2026-11-18 JST. Run `pcqm-gptrans-real-tri-late3-100k-20261118-r2-s42`, attempt `v1` (`kaseichou/molgap-gptrans-real-tri-late3-100k-20261118-s42`, `version=1`, commit `edaa6ac62a5f89ad40435c98ee9f1d301e884dc0`, spec `576f7c2b5f931ed6ef8e15718addeb9bb5ffee19c5cd828073f22cad4ffdf279`), strictly retrieved and diagnosed.

## Facts
- Contract & validity: Both arms completed 60 passes (46,860 optimizer steps, 5,998,080 sample presentations) and are `MECHANICALLY_VERIFIED`. Python main-process open provenance (`data_open_provenance.json`) shows `official_validation`, `test_dev`, and `test_challenge` remain `untouched` for both arms.
- Metric: selected EMA (`0.999`) MAE on the 50,000-row `internal_development` split of `pcqm_fixed_100k_v1`.
- `pair_update_norm` (same-job control):
  - Selected EMA `0.999` dev MAE: `139.440` meV (`0.139440` eV, pass 42, step 32,802); pass 60 EMA dev MAE: `139.867` meV.
  - Aux EMA `0.9999` dev MAE: `143.367` meV (`0.143367` eV).
  - Online train MAE: `90.379` meV at selected pass 42; `70.450` meV at pass 60.
  - Process wall: `9,754.1` s (`2.710` h) on Tesla T4 (`device_seconds` missing, requires platform allocation accounting).
- `real_tri_late3` (candidate, `LAYERS=(6, 8, 10)`):
  - Selected EMA `0.999` dev MAE: `140.462` meV (`0.140462` eV, pass 38, step 29,678); pass 60 EMA dev MAE: `140.665` meV.
  - Aux EMA `0.9999` dev MAE: `143.125` meV (`0.143125` eV).
  - Online train MAE: `97.604` meV at selected pass 38; `68.416` meV at pass 60.
  - Process wall: `10,842.3` s (`3.012` h) on Tesla T4 (`+11.2%` wall ratio `1.112`, `+1,088.1` s; `device_seconds` missing).
- Paired comparison (`real_tri_late3 - pair_update_norm`, 50,000 rows, 2000-bootstrap seed 42):
  - Effect: `+1.022` meV (`+0.001022` eV).
  - 95% bootstrap CI: `[+0.134, +1.891]` meV.
  - `P(candidate better)`: `0.0105`; row win fraction: `0.49322` (24,661 vs 25,339).
- One seed (42), one init per arm.

## Frozen Gate (`protocol.md`)
Candidate minus control must satisfy `<= -3.0` meV with 95% CI upper bound `< 0`. Observed `+1.022` meV with 95% CI `[+0.134, +1.891]` meV; the gate is not met.

## Outcome
- `pair_update_norm`: `CLOSED` (`failure_mode: null`).
- `real_tri_late3`: `NEGATIVE_UNDER_CONTRACT` (`failure_mode: "below_gate"`).

## Attribution
- Trajectory: `real_tri_late3` led in only `22/60` paired evaluations. Pass-window EMA deltas (`real_tri_late3 - pair_update_norm`): `P1-10 = +0.481`, `P11-20 = -0.610`, `P21-40 = +0.182`, `P41-60 = +1.052` meV.
- Failure-mode note (`below_gate`): Restricting real-atom triangle updates to late layers `(6, 8, 10)` removed the early/mid `-3.6` to `-4.6` meV geometric acceleration observed in 5-layer `real_tri_pair` (`LAYERS=(2, 4, 6, 8, 10)`), while still lowering pass-60 online train MAE (`68.416` vs `70.450` meV) and hurting dev MAE by `+1.022` meV (CI `[+0.134, +1.891]` meV). Early-layer `(2, 4)` triangle composition is essential for the `-1.667` meV gain rather than a late-overfitting source to prune.

## Scope
This result covers one seed at 100K on `internal_development`. It does not evaluate larger data scales, other seeds, or the sealed `official_validation` and test roles.
