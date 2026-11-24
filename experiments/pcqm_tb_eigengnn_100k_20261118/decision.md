# Decision

2026-11-18 JST. Run `pcqm-tb-eigengnn-100k-20261118-s42`, attempt `v1` (`nvoid912/molgap-tb-eigengnn-100k-20261118-s42`, `kernel_id=145744979`), closed as `STOP_FOR_COST` without submitting a `v2` continuation.

## Facts
- Contract & validity: both arms (`pair_update_norm`, `tb_eigengnn`) ran on Kaggle T4x2 (`pcqm_fixed_100k_v1`, 100K train / 50K `internal_development`, FP32, `batch_size=128`, seed 42). Control `pair_update_norm` completed 60/60 passes (`output_manifest.json` verified). Candidate `tb_eigengnn` paused at `optimizer_steps=19,516` (`sample_presentations=2,498,048`, 24 completed passes) when the frozen allocation ceiling (`16,022.80 s` window) exhausted (`pause_receipt.json` status `PAUSED`).
- Role audit: `official_validation`, `test_dev`, and `test_challenge` remained `untouched` (`data_open_provenance.json` sha256 `872e43ff61393ce125033c1ce406bcb91ed9f7bbde44753a24ae1e444790e99f` for `pair_update_norm`, `7acae294f493c02f2392b03d09a587aa3c478f5048fbde675f04ab79211e41c1` for `tb_eigengnn`).
- `pair_update_norm` (control, 60/60 passes):
  - Pass-24 EMA dev MAE: **146.154 meV** (`step=18,744`, `live_train=101.981 meV`).
  - Best EMA dev MAE: **139.440 meV** (pass 42, `step=32,802`, `ema_0p9999=174.884 meV`, `live_train=70.881 meV`).
  - Final pass-60 EMA dev MAE: **139.867 meV** (`ema_0p9999=143.367 meV`, `live_train=55.251 meV`).
  - Training wall time: **9,455.50 s** (2.63 h on Tesla T4; ~157.6 s/pass average).
- `tb_eigengnn` (candidate, paused after 24/60 passes):
  - Pass-24 / latest retained EMA dev MAE: **158.562 meV** (`step=18,744`, `ema_0p9999=180.254 meV`, `live_train=101.897 meV`).
  - Candidate invocation wall time: **15,709.06 s** (4.36 h on Tesla T4; ~654.5 s/pass average across 24 passes, 629 s/pass logged; ~4.1x slower than control; whole-job window `16,022.80 s`).
- Matched-pass comparison (`tb_eigengnn - pair_update_norm` on EMA dev MAE):
  - Pass 24 delta: **+12.408 meV** (`158.562 - 146.154 meV`).
  - Passes 1–5 led (`-1.446, -14.589, -16.868, -6.775, -1.126 meV`), crossed over at pass 6 (`+4.463 meV`), and trailed monotonically on passes 6–24 (`+4.463` to `+13.910 meV`).

## Outcome
- `pair_update_norm`: `STOP_FOR_COST`
- `tb_eigengnn`: `STOP_FOR_COST`

## Terminal Attribution
At pass 24, `tb_eigengnn` matches `pair_update_norm` on online training MAE (`101.897` vs `101.981 meV`) while trailing development EMA MAE by `+12.408 meV` (`158.562` vs `146.154 meV`) at `>4x` T4 wall cost per pass (`15,709.06 s` for 24 passes vs `9,455.50 s` for 60 passes). Per-graph FP32 dense 5-orbital tight-binding Hamiltonian diagonalization (`eigvalsh`) adds prohibitive CUDA kernel/synchronization latency while generalizing worse than the BondLocal+PairNorm GPTrans-T baseline. Closed permanently at `STOP_FOR_COST` without `v2` continuation.

## Scope
Partial 24/60-pass run on seed 42 at 100K (`pcqm_fixed_100k_v1` internal development); candidate predictions beyond pass 24 are absent and `tb_eigengnn` is closed without full 60-pass execution. Official validation and test roles remain untouched.
