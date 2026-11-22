# C3 checkpoint-space probe: MC-dropout and retained pass-60 weights

Track B, local only, 2026-11-23. Authorized local execution of item 4 (C3) from
the supplied `final_ruling.md`. No training, remote job, tail, official
validation/test, 3D input or model promotion is released.

## Question and snapshot inventory

Item C3 in `final_ruling.md` has two NO_TRAIN halves:

1. **Multi-epoch snapshot averaging** (`{54, 57, 60}` and `{48..60}`): local
   artifact inspection shows neither the full-scale GPTrans runs
   (`pcqm-gptrans-b1024-minus-tail-20261114-s42`, `-s43-20261115`, `-s44`,
   `-s45`, where `ema_model.pt` and `selected_model.pt` have identical weights)
   nor the 500K GPTrans runs (`pcqm-gptrans-pairnorm-init-noise-500k-s43`,
   `-s45`) retained intermediate epoch snapshots. Per `final_ruling.md`
   ("快照未保留则取消"), the multi-epoch snapshot-averaging half is
   **cancelled (`snapshots_not_retained`)**.
2. **MC-dropout** (dropout layers only, `K = 8, 16, 32`) versus deterministic
   `EMA@60`, plus a descriptive check of the only retained pass-60 weight pair
   (`ema60` vs `live60` and their 50/50 weight and prediction averages).

Full-scale GPTrans trained on all non-tail rows `[0, 3250000)`, so it has no
non-protected unseen rows; `tail_holdout_v1` is never touched for probing or
selection. This study therefore runs on the strict-verified 500K GPTrans-T
PairNorm pair (`init_s43` and `init_s44`).

## Frozen design

- Backbones: 500K `init_s43` and `init_s44`, pinned by their immutable
  `output_manifest.json` (`9f5131f3...`, `2a439ae4...`), `selected_model.pt`,
  and `last_checkpoint.pt` (`resume` artifact hashes `af709cc6...`,
  `170b62f8...`).
- Numerical verification before scoring: replay each `selected_model.pt` on the
  first 128 saved development predictions (max abs error <= 0.0001 eV), and
  verify each `ema60` full-dev MAE matches pass 60 in `canonical_trace.json`
  (<= 0.0001 eV) at scoring.
- Weight states per init:
  - `ema60`: `last_checkpoint.pt["ema"]` (pass-60 EMA 0.999 weights; not
    selected on development rows).
  - `live60`: `last_checkpoint.pt["model"]` (pass-60 live weights).
  - `wavg60`: 50/50 weight average of `ema60` and `live60` floating tensors.
  - `sel`: `selected_model.pt` (best-dev EMA weights).
- MC-dropout on `ema60`: put model in `eval()`, then enable `.train()` on
  `nn.Dropout` modules only (`dropout=0.1`; `LayerNorm` stays in eval). Run
  `K = 32` stochastic forward passes per batch (`batch_size=128`, FP32 without
  TF32, deterministic seed `20261123 + pass_idx * 100000 + batch_idx`). Save
  cumulative means at `K = 8, 16, 32` and the 32-pass std.
- Score blocks (ordinary OGB train rows only; zero protected roles):
  - **dev (primary): all `[500000, 550000)`** (shard 10, 500K development
    role; matches `final_ruling.md` / Prosecution §3).
  - **unseen (secondary, selection-free): all `[3100000, 3150000)`** (shard 62,
    50,000 rows unseen by 500K training and checkpoint selection, never scored
    by any prior study in this series).
- Seal all predictions with row IDs and SHA256 before reading score labels.
  At scoring, align archived CSV labels with packed shard labels.

## Endpoints and verdict

Paired row bootstrap (2,000 draws, seed 20261122, models fixed). Primary
endpoint is `dev` `[500000, 550000)` comparing `mc32` against deterministic
`ema60` for `s43`, `s44`, and their equal-weight 2-init ensemble `ens`:

- **PASS:** both `s43` and `s44` have `MAE(mc32) - MAE(ema60) <= -0.5 meV` with
  95% upper bound `< 0`, AND `MAE(ens_mc32) - MAE(ens_ema60) <= 0`.
- **FAIL:** either init has 95% lower bound `> -0.5 meV`, OR the ensemble 95%
  lower bound is `> 0`.
- Otherwise **INCONCLUSIVE**.

`K = 8, 16`, the `unseen` block, `live60` / `wavg60` / `pavg60` / `sel`
comparisons, and MC-std error correlations are descriptive only.

## Custody and cost

Publish the prospective trajectory before any data read. One local RTX 5060
run (~20–35 min expected for 32 passes x 2 inits x 2 blocks). No training, no
retry on a scientific miss, no protected role. Row outputs stay outside Git in
`D:/molgap-artifacts/diagnostics/pcqm-c3-checkpoint-space/`.
