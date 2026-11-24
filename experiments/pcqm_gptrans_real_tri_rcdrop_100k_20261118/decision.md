# Decision

2026-11-18 JST. Run `pcqm-gptrans-real-tri-rcdrop-100k-20261118-s42`, attempt `v1` (`nvoid912/molgap-gptrans-real-tri-rcdrop-100k-20261118-s42`, `kernel_id=145712574`, `version=1`, commit `5146fdeb9a13fae9d9d569047d7019b6d25a9b92`, spec `36ef975b30d478d6c8982a7760316410de4faa25d704ade7dda6466b0ebc3893`), strictly retrieved and diagnosed.

## Facts
- Contract & validity: Both arms completed 60 passes (`46,860` optimizer steps, `5,998,080` sample presentations) and are `MECHANICALLY_VERIFIED`. Python main-process open provenance (`data_open_provenance.json`) shows `official_validation`, `test_dev`, and `test_challenge` remain `untouched` (`false`) for both arms (`pair_update_norm` sha256 `c5ea4f46eee6376db3ee5a4b6ba23eca4ad65c82eb6d8cacb5532854bcfd7260`; `real_tri_rcdrop` sha256 `227cb80cbf3299ea0bbebf8cb1d88547c1280c220f5d8dfb6e61d44416cad1d7`).
- Metric: selected EMA (`0.999`) MAE on the 50,000-row `internal_development` split of `pcqm_fixed_100k_v1`.
- `pair_update_norm` (same-job control):
  - Selected EMA `0.999` dev MAE: `139.440` meV (`0.139440` eV, pass 42, step 32,802); pass 60 EMA dev MAE: `139.867` meV.
  - Aux EMA `0.9999` dev MAE: `143.367` meV (`0.143367` eV).
  - Online train MAE: `70.881` meV at selected pass 42; `55.251` meV at pass 60.
  - Process wall: `9,628.5` s (`2.675` h) on Tesla T4 (`device_seconds` missing, requires platform allocation accounting).
- `real_tri_rcdrop` (candidate):
  - Selected EMA `0.999` dev MAE: `138.444` meV (`0.138444` eV, pass 45, step 35,145); pass 60 EMA dev MAE: `138.583` meV.
  - Aux EMA `0.9999` dev MAE: `141.690` meV (`0.141690` eV).
  - Online train MAE: `66.129` meV at selected pass 45; `55.028` meV at pass 60.
  - Process wall: `11,741.5` s (`3.262` h) on Tesla T4 (+21.9% wall ratio `1.219`, +2,113.0 s; `device_seconds` missing).
- Paired comparison (`real_tri_rcdrop - pair_update_norm`, 50,000 rows, 2000-bootstrap seed 42):
  - Effect: `-0.996` meV (`-0.000996` eV).
  - 95% bootstrap CI: `[-1.902, -0.133]` meV.
  - `P(candidate better)`: `0.9880`; row win fraction: `0.5046` (25,230 vs 24,770).
- Pass-window EMA (`0.999`) deltas (`real_tri_rcdrop - pair_update_norm`):
  - `P1-10`: `-0.602` meV (`[-3.298, +3.348]`)
  - `P11-20`: `-2.154` meV (`[-2.713, -1.824]`)
  - `P21-40`: `-1.091` meV (`[-2.018, -0.543]`)
  - `P41-60`: `-1.019` meV (`[-1.284, -0.379]`)
  - Candidate led in 56 / 60 passes.

## Frozen Gate (`protocol.md`)
Candidate minus control must satisfy `<= -3.0` meV with 95% CI upper bound `< 0`. Observed `-0.996` meV with 95% CI `[-1.902, -0.133]` meV: the upper bound is `< 0`, but the point improvement does not reach the `-3.0` meV material gate.

## Outcome
- `pair_update_norm`: `CLOSED` (`failure_mode: null`).
- `real_tri_rcdrop`: `POSITIVE_BELOW_GATE` (`failure_mode: "below_gate"`).

## Terminal Attribution
- Trajectory: Adding post-gate structured row/column dropout (`p=0.20`, row at layers 2, 6, 10; col at layers 4, 8) onto `real_tri_pair` damped the mid-training advantage (`P11-20` mean `-2.154` meV vs `-4.628` meV for unregularized `real_tri_pair`) without improving late-horizon retention (`P41-60` mean `-1.019` meV; last 10 passes of `real_tri_pair` were `-1.050` meV). At pass 60, online train MAE converged to `55.028` vs `55.251` meV (`-0.223` meV) while dev MAE improved by `-1.284` meV (`138.583` vs `139.867` meV), yielding `-0.996` meV overall at +21.9% T4 wall cost.
- Failure-mode note (`below_gate`): Structured row/col dropout (`p=0.20`) attenuated the useful triangle signal faster than it reduced late-horizon overfitting at 100K, leaving `real_tri_rcdrop` weaker than unregularized `real_tri_pair` (`-0.996` vs `-1.667` meV) and below the `-3.0` meV material gate.

## Scope
One seed (42) at 100K on `internal_development`. Official validation and test roles remain unread.
