# Decision

2026-11-18 JST. Run `pcqm-gptrans-real-tri-vread-100k-20261118-r2-s42`, attempt `v1` (`nvoid912/molgap-gptrans-real-tri-vread-100k-20261118-s42`, `kernel_id=145712650`, `version=1`, commit `3f28e2bb4`, spec `afa8cf9c76cc04cf8dbe8d539bb21a26734ffb648ac082e6b7e5cbce82a52049`), strictly retrieved and diagnosed (superseding initial `...-s42` which failed local `dataset_create_new` slug-length validation `51 > 50` chars with zero remote resources created).

## Facts
- Contract & validity: Both arms completed 60 passes (`46,860` optimizer steps, `5,998,080` sample presentations) and are `MECHANICALLY_VERIFIED`. Python main-process open provenance (`data_open_provenance.json`) shows `official_validation`, `test_dev`, and `test_challenge` remain `untouched` (`false`) for both arms (`pair_update_norm` sha256 `830c082f609c288df9995f907c54a4ec9c68cabf411e1f7ab25e2fa7a5796375`; `real_tri_vread` sha256 `2dd05d8f4663344ea97387ad83b1eb96f0a7266dcdef3b72fa3b7610138194f8`).
- Metric: selected EMA (`0.999`) MAE on the 50,000-row `internal_development` split of `pcqm_fixed_100k_v1`.
- `pair_update_norm` (same-job control):
  - Selected EMA `0.999` dev MAE: `139.440` meV (`0.139440` eV, pass 42, step 32,802); pass 60 EMA dev MAE: `139.867` meV.
  - Aux EMA `0.9999` dev MAE: `143.367` meV (`0.143367` eV).
  - Online train MAE: `70.881` meV at selected pass 42; `55.251` meV at pass 60.
  - Process wall: `9,818.9` s (`2.727` h) on Tesla T4 (`device_seconds` missing, requires platform allocation accounting).
- `real_tri_vread` (candidate):
  - Selected EMA `0.999` dev MAE: `137.580` meV (`0.137580` eV, pass 42, step 32,802); pass 60 EMA dev MAE: `138.021` meV.
  - Aux EMA `0.9999` dev MAE: `140.139` meV (`0.140139` eV).
  - Online train MAE: `67.720` meV at selected pass 42; `51.698` meV at pass 60.
  - Process wall: `12,092.2` s (`3.359` h) on Tesla T4 (+23.2% wall ratio `1.232`, +2,273.3 s; `device_seconds` missing).
- Paired comparison (`real_tri_vread - pair_update_norm`, 50,000 rows, 2000-bootstrap seed 42):
  - Effect: `-1.860` meV (`-0.001860` eV).
  - 95% bootstrap CI: `[-2.783, -0.909]` meV.
  - `P(candidate better)`: `1.0000`; row win fraction: `0.5088` (25,442 vs 24,558).
- Pass-window EMA (`0.999`) deltas (`real_tri_vread - pair_update_norm`):
  - `P1-10`: `-2.412` meV (`[-5.391, +2.797]`)
  - `P11-20`: `-4.142` meV (`[-4.904, -3.134]`)
  - `P21-40`: `-2.577` meV (`[-3.704, -1.863]`)
  - `P41-60`: `-1.828` meV (`[-2.006, -1.560]`)
  - Candidate led in 58 / 60 passes.

## Frozen Gate (`protocol.md`)
Candidate minus control must satisfy `<= -3.0` meV with 95% CI upper bound `< 0`. Observed `-1.860` meV with 95% CI `[-2.783, -0.909]` meV: the upper bound is `< 0`, but the point improvement does not reach the `-3.0` meV material gate.

## Outcome
- `pair_update_norm`: `CLOSED` (`failure_mode: null`).
- `real_tri_vread`: `POSITIVE_BELOW_GATE` (`failure_mode: "below_gate"`).

## Terminal Attribution
- Trajectory: Adding L12 virtual-query pair pooling readout onto `real_tri_pair` improved over `real_tri_pair` (`-1.860` vs `-1.667` meV; P60 `138.021` vs `138.790` meV; aux EMA `0.9999` `140.139` vs `142.011` meV) and retained stronger late-horizon lead (`P41-60` mean `-1.828` meV vs `-1.050` meV for `real_tri_pair`). However, P60 online train MAE dropped to `51.698` vs `55.251` meV (`-3.553` meV), widening the P60 dev-minus-train gap (`86.323` vs `84.616` meV).
- Failure-mode note (`below_gate`): At 100K, L12 virtual-query pair pooling + real-atom triangle pair updates genuinely improved dev MAE (`-1.860` meV, CI excludes zero) and beat `real_tri_pair` alone, but 100K overfitting still damped the `-4.142` meV mid-training lead down to `-1.860` meV, missing the `-3.0` meV material gate at `+23.2%` T4 wall cost.

## Scope
One seed (42) at 100K on `internal_development`. Official validation and test roles remain unread.
