# Decision

2026-11-18 JST. Run `pcqm-gptrans-rrwp-pairffn-100k-20261118-s42`, attempt `v1`, strictly retrieved and diagnosed from Kaggle 2 (`kaseichou/molgap-gptrans-rrwp-pairffn-100k-20261118-s42`, `kernel_id=145653182`, `version=1`).

## Facts
- **Contract & Provenance**: Both arms completed 60/60 passes (`46,860` optimizer steps, `5,998,080` sample presentations) from shared `gptrans.pt` under source commit `5575cfaa201a6a9255de7134d0b4db2a3777789d` (`package_identity=820a66052bdaaf3ea97456bd0e7656b3b47d2f3162542c22f945ad9145be2d2d`). Both arms are `MECHANICALLY_VERIFIED`.
- **Metric**: Selected EMA (`0.999`) MAE on the 50,000-row `internal_development` split of `pcqm_fixed_100k_v1`.
- **`pair_update_norm` (same-job control)**:
  - Selected EMA `0.999` dev MAE: `139.440 meV` (pass 42, step 32,802); final pass 60 EMA `0.999`: `139.867 meV`.
  - Aux EMA `0.9999` dev MAE: `143.367 meV` (step 46,860).
  - Online train MAE: `90.379 meV` at selected pass 42; `70.450 meV` at final pass 60.
  - Process wall: `9,660.95 s` (`2.684 h`).
- **`rrwp_pairffn` (candidate)**:
  - Selected EMA `0.999` dev MAE: `140.079 meV` (pass 46, step 35,926); final pass 60 EMA `0.999`: `140.578 meV`.
  - Aux EMA `0.9999` dev MAE: `144.980 meV` (step 46,860).
  - Online train MAE: `82.433 meV` at selected pass 46; `69.990 meV` at final pass 60.
  - Process wall: `12,202.18 s` (`3.389 h`, `1.263x` control).
- **Paired Comparison (`rrwp_pairffn - pair_update_norm`)**:
  - Effect: `+0.639 meV` (`+0.0006385 eV`).
  - Paired 2000-bootstrap 95% CI: `[-0.265, +1.569] meV`.
  - `P(candidate better)`: `0.083`; candidate better on `24,796 / 50,000` rows (`49.592%`).
- **Native Cost & Role Audit**:
  - Allocation ledger: `12,516.83 s` wall (`3.477 h`, within `14,400 s` ceiling), `25,033.66 s` allocated T4 device-seconds (`6.954` T4 device-hours, within `8` T4h ceiling).
  - Role audit (`data_open_provenance.json`, `roles_*.json`): `official_validation`, `test_dev`, and `test_challenge` remain `untouched`.

## Frozen Gate (`protocol.md`)
Candidate minus control must satisfy `<= -3.0 meV` with paired 95% bootstrap upper bound `< 0`. Observed `+0.639 meV` with 95% CI `[-0.265, +1.569] meV`. The gate is not met.

## Outcome
- `pair_update_norm`: `CLOSED` (`failure_mode: null`, same-job control arm).
- `rrwp_pairffn`: `NEGATIVE_UNDER_CONTRACT` (`failure_mode: below_gate`).

## Terminal Attribution
- **Pass-by-pass dynamics (`late_horizon_fade: true`)**: `rrwp_pairffn` led early (`passes 1-10` mean delta `-2.760 meV`, `passes 11-20` mean delta `-1.044 meV`, ahead on 23/60 paired evaluations through pass 24), then crossed over at pass 25 (`+0.463 meV`) and trailed over `last_10_passes` (`+0.688 meV`).
- **Mechanism / generalization**: With 16-step RRWP injection at block 0 and Pre-LN SwiGLU `PairTransition(32->64->32)` across all 12 blocks, `rrwp_pairffn` achieved slightly lower final online train MAE (`69.990` vs `70.450 meV`) and a larger selected-step dev-minus-train gap (`57.645` vs `49.061 meV`) without improving generalization on `internal_development`, while adding `+26.3%` wall time (`+2,541.22 s`). Stratum-wise, it gained slightly on `<4.0 eV` (`-1.488 meV`) but degraded on `4.0-6.0 eV` (`+0.484 meV`), `6.0-8.0 eV` (`+1.555 meV`), and `>=8.0 eV` (`+4.422 meV`). Post-hoc constant shift only accounts for `27.5%` of the delta (`+0.463 meV` shift-corrected).
- **Disposition**: Close `gptrans_rrwp_pair_transition` at 100K without hyperparameter grids or automatic scale-up.

## Scope
This result covers one seed (`42`) at 100K on the internal development split. Official validation and test roles remain untouched.
