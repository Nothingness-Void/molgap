# Decision

2026-11-18 JST. Run `pcqm-gptrans-hl-gauss-100k-20261118-s42`, attempt `v1`, strictly retrieved and diagnosed.

## Facts
- Contract & validity: both arms (`pair_update_norm`, `hl_gauss_512`) completed 60 passes (46,860 optimizer steps, 5,998,080 sample presentations) on `pcqm_fixed_100k_v1` from identical `gptrans.pt` initialization (`b7ec45911db1fd83cdb35d67e0b96f039a2fde0f`), and passed mechanical verification.
- Role audit: `official_validation`, `test_dev`, and `test_challenge` remained `untouched` (`data_open_provenance.json` verified).
- Metric: selected EMA `0.999` MAE on the 50,000-row `internal_development` split of `pcqm_fixed_100k_v1`.
- `pair_update_norm` (same-job control):
  - Selected EMA `0.999` dev MAE: **139.440 meV** (pass 42, step 32,802); final pass-60 EMA `0.999` dev MAE: **139.867 meV**.
  - Aux EMA `0.9999` dev MAE: **143.367 meV** (step 46,860).
  - Online train MAE: **90.379 meV** at selected step, **70.450 meV** at final step.
  - Measured process wall time: **10,611.4 s** (~2.95 h on Tesla T4; `device_seconds` missing).
- `hl_gauss_512` (candidate):
  - Selected EMA `0.999` dev MAE: **150.151 meV** (pass 29, step 22,649); final pass-60 EMA `0.999` dev MAE: **159.868 meV**.
  - Aux EMA `0.9999` dev MAE: **153.984 meV** (step 37,488).
  - Online train MAE: **99.095 meV** at selected step, **44.873 meV** at final step.
  - Measured process wall time: **10,755.9 s** (~2.99 h on Tesla T4; `device_seconds` missing).
- Paired comparison (`hl_gauss_512 - pair_update_norm` on 50,000 aligned rows):
  - Effect: **+10.711 meV** (`+0.010711 eV`).
  - Paired 2000-bootstrap 95% CI: **[+9.487, +11.910] meV**.
  - Probability candidate better: **0.0000** (row win fraction: 0.4710).
- One seed (42), one init per arm.

## Frozen Gate (`protocol.md`)
Candidate minus control must be `<= -3.0 meV` with 95% upper bound `< 0`. Observed `+10.711 meV` with 95% CI `[+9.487, +11.910] meV`. The gate is not met.

## Outcome
- `pair_update_norm`: `CLOSED` (`failure_mode: null`).
- `hl_gauss_512`: `NEGATIVE_UNDER_CONTRACT` (`failure_mode: below_gate`).

## Terminal Attribution
- **Convergence trajectory**: `hl_gauss_512` trailed at passes 1–2 (`+148.1`, `+20.9 meV`), led `pair_update_norm` across passes 3–19 (peaking at `-22.6 meV` at pass 4; mid-window passes 11–20 mean `-2.310 meV`), and crossed over at pass 20 (`+1.08 meV`), worsening to `+19.983 meV` over the last 5 passes.
- **Failure mechanism**: Overfitting / objective-metric mismatch on 100K. Under pure 512-bin HL-Gauss cross-entropy, the model continued fitting the training distribution aggressively (`online_train_mae` fell from `99.1 meV` at pass 29 to `44.9 meV` at pass 60, vs `70.4 meV` for scalar L1 control), while development MAE degraded by `+9.717 meV` after pass 29 (`150.151 -> 159.868 meV`, dev-minus-train gap `+115.0 meV` vs `+69.4 meV` for control). Degradation appeared across all gap strata (`+9.18 meV` on 4–6 eV, `+21.54 meV` on <4 eV, `+36.99 meV` on >=8 eV).
- **Closeout**: Closed without hyperparameter grids or retries.

## Scope
This result covers one seed (42) at 100K on the internal development set. It does not establish behavior at larger data scales or other seeds, and official validation/test roles remain unread.
