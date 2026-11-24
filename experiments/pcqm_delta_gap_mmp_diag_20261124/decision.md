# Single-point substitution ΔGap diagnostic: do not train

2026-11-24, Track B, local RTX 5060. Close this zero-training diagnostic as RML
`NO_TRAIN` under pre-registered verdict `INCONCLUSIVE_DO_NOT_TRAIN` by the
[frozen protocol](protocol.md). **Do not train a signed-ΔGap auxiliary loss
on these backbones.** Real single-point substitution MMP pairs clear `3 meV`
whole-block recoverable MAE in isolation (`13.64` / `13.66 meV`), yet recover
significantly **less** error than the covariate-balanced non-MMP rewiring
control (`14.61` / `14.81 meV`; excess `−0.967` / `−1.147 meV`). Zero training
arms are consumed from the 2-slot / 4-arm `kaggle2` budget, and no protected
role is touched.

## Result

500K GPTrans-T PairNorm `init_s43` and `init_s44` pass-60 EMA weights (`ema60`),
scored once across `N = 190,450` untouched ordinary rows (`Shards 58–61 \
spread-fit fit_idx`, `[2900000, 3100000)`). Stage 1 formed `M = 70,065` disjoint
1-cut Hussain–Rea MMP pairs (`molecule_coverage = 73.58%`, `75,918` hierarchical
series clusters, `max_series_cluster_share = 3.32%`). All metrics in meV except
correlation `r`; primary intervals use 2,000-draw series-clustered ratio
bootstrap (seed `20261124`).

| Arm | Whole MAE | Paired MAE | `R_real` [95% CI] | `R_rew` [95% CI] | **`R_real − R_rew` [95% CI]** | `res corr` (real / rew) | Pair ΔGap MAE (`r`) (real / rew) |
|---|---:|---:|---:|---:|---:|---|---|
| **`s43`** | 97.901 | 89.840 | 13.644 [13.182, 14.145] | 14.611 [14.176, 15.090] | **−0.967 [−1.210, −0.732]** | +0.130 / +0.074 | 122.696 (`0.9716`) / 128.283 (`0.9697`) |
| **`s44`** | 98.289 | 90.210 | 13.664 [13.197, 14.159] | 14.811 [14.385, 15.279] | **−1.147 [−1.397, −0.909]** | +0.134 / +0.069 | 122.602 (`0.9715`) / 129.192 (`0.9692`) |
| `2-init ens` (desc.) | 94.440 | 86.927 | 13.184 [12.722, 13.702] | 14.202 [13.780, 14.675] | −1.018 [−1.244, −0.791] | +0.127 / +0.069 | 118.876 (`0.9730`) / 124.766 (`0.9710`) |

- **Murcko-clustered sensitivity (`56,752` clusters, descriptive):** `R_real − R_rew`
  equals `−0.967 [−1.415, −0.580] meV` (`s43`), `−1.147 [−1.659, −0.692] meV`
  (`s44`), and `−1.018 [−1.476, −0.614] meV` (`2-init ens`).
- **Stage 3 Balance Gate (passed all 9 checks, `selected_lambda = 0.08`):**
  `unmatched = 0`, `mmp_leak = 0`, `same_Murcko = 0.50235 vs 0.50235` (`0.0` diff),
  `Tanimoto SMD = +0.0886` (`KS = 0.0441`), `|ΔHA| SMD = 0.0` (`KS = 0.0053`),
  `|ΔGap| SMD = −3.6e−7` (`KS = 0.0280`).
- **Pre-registered verdict:** Both `R_real` point estimates exceed `3.0 meV`, so
  `STOP_NO_TRAIN` (`CI upper < 3.0`) does not trigger, while `PASS_DIAGNOSTIC`
  fails decisively because both `(R_real − R_rew)` lower bounds sit well below
  `0.0`. Per Section 5 of the [protocol](protocol.md), verdict is
  `INCONCLUSIVE_DO_NOT_TRAIN` and training is withheld.

## Why (attribution)

- **Valid and verified.** In Stage 2 on `train_shard_0010.pt` (`50,000` dev
  rows), `selected_model.pt` replayed saved first-128 predictions within
  `7.6e−7 eV` (`s43`: `7.58e−7`, `s44`: `6.54e−7`), and `ema60` full-dev MAE
  matched `canonical_trace.json` pass 60 within `4.8e−9 eV` (`0.101540940 vs
  0.101540938` for `s43`; `0.101365407 vs 0.101365402` for `s44`). Stage 1
  excluded 28 mega-buckets (`> 300` entries) from both real pairing and Stage 3
  rewiring (`mmp_leak = 0`). All three `.npz` archives match their receipt SHA256s.
- **Failure mode:** `below_gate` on the pre-registered excess-over-rewiring
  criterion (`R_real − R_rew` lower bound `> 0.0` failed in the opposite
  direction by `−0.97` / `−1.15 meV`).
- **Mechanism:** By `ΔE_uv = |r_u| + |r_v| − |r_u + r_v|`, independent or
  negatively correlated residual errors yield large `ΔE_uv`, whereas positively
  correlated sibling errors leave `|r_u + r_v|` large and shrink recoverable
  MAE. Fixed 500K GPTrans-T PairNorm already predicts single-point substitution
  `ΔGap` more accurately than matched non-MMP pairs (`122.7 / 122.6` vs `128.3 /
  129.2 meV`, `r = 0.972`) and exhibits nearly twice the residual correlation on
  real MMP siblings (`+0.130 / +0.134` vs `+0.074 / +0.069` for rewired pairs).
  Therefore the `~13.6 meV` raw `R_real` reflects generic error cancellation across
  paired molecules rather than excess local `ΔGap` failure over common-series
  bias.
- **Limits and reopen condition:** Holds for fixed 500K GPTrans-T PairNorm
  `EMA@60` under 1-cut Hussain–Rea MMPs (`core_ha >= 6`, `1 <= r_ha <= 12`). Any
  future follow-up must target a distinct error structure (e.g., series-level
  common bias or explicit extreme activity-cliff regimes) under a new
  prospective protocol.

## Execution, remote status, and cost

- **Remote note:** Private dataset `kaseichou/molgap-delta-gap-diag-20261124-v2`
  and kernel `kaseichou/molgap-deltagap-mmp-diagnostic-20261124` (`v1`,
  `kernel_id=146939480`, `NvidiaTeslaT4`) were submitted to `kaggle2`, where the
  job remained `KernelWorkerStatus.QUEUED` for >2h48m with `0` GPU seconds used
  and no output log.
- **Local execution:** Using the identical frozen 4-stage pipeline added as
  `--mode local-full` in [probe_delta_gap.py](probe_delta_gap.py) over the exact
  SHA-verified `s43`/`s44` checkpoints and `pcqm_full_train_v1` shards, the
  run finished cleanly on `NVIDIA GeForce RTX 5060` in `270.8 s` total stage
  wall time (Stage 1 `56.51 s`, Stage 2 `132.47 s`, Stage 3 `75.42 s`, Stage 4
  `6.36 s`; parent CPU `6.70 s`, `132.63 s`, `73.67 s`, `6.31 s`). Exclusive GPU
  device-hours and Stage 1 worker-pool CPU hours were not measured and stay
  `measurement_missing`.
- **Roles:** Ordinary rows `Shards 58–61 \ spread-fit fit_idx` (`190,450` rows)
  were consumed as the scoring block; `train_shard_0010.pt` was re-verified for
  deterministic dev replay. Protected roles (`tail_holdout_v1`,
  `official_validation`, `test_dev`, `test_challenge`, `common`, `ood`) remain
  untouched.

See [results](results.json) and [acceptance](acceptance.json) for unrounded
numbers. Binary row arrays remain outside Git in
`D:/molgap-artifacts/diagnostics/pcqm-delta-gap-mmp-diag-20261124/local_rtx5060/`.
