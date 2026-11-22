# C3 passes for MC-dropout (K=32); multi-epoch snapshot averaging cancelled

2026-11-23, Track B, local RTX 5060. Close this NO_TRAIN probe as `NO_TRAIN`
by the [frozen protocol](protocol.md). **MC-dropout (K=32) verdict: PASS.**
**Multi-epoch snapshot averaging: CANCELLED (`snapshots_not_retained`).** No
training, remote job, tail, official validation/test or model promotion is
released.

## Result

500K GPTrans-T PairNorm `init_s43` and `init_s44`, pass-60 EMA weights
(`ema60`) from `last_checkpoint.pt`. Dropout-only `train()` enabled all 49
`nn.Dropout` modules (`p=0.1`); all `LayerNorm` modules stayed in `eval()`.
MAE and deltas are in meV (negative is better); intervals are the paired row
bootstrap (2,000 draws, seed 20261122) with models fixed.

| Block | Arm | `ema60` | `mc8` (Δ vs `ema60`) | `mc16` (Δ) | **`mc32` (Δ [95% CI])** |
|---|---|---:|---:|---:|---:|
| **dev `[500000, 550000)`, primary** | **s43** | 101.541 | 101.099 (−0.442) | 100.884 (−0.657) | **100.789 (−0.752 [−0.856, −0.646])** |
| | **s44** | 101.365 | 100.959 (−0.407) | 100.811 (−0.554) | **100.673 (−0.693 [−0.802, −0.587])** |
| | **2-init ens** | 98.461 | 98.161 (−0.300) | 98.060 (−0.401) | **97.985 (−0.476 [−0.563, −0.392])** |
| unseen `[3100000, 3150000)`, secondary | s43 | 99.572 | 99.402 (−0.170) | 99.115 (−0.457) | 98.975 (−0.597 [−0.735, −0.456]) |
| | s44 | 99.391 | 99.331 (−0.060) | 98.980 (−0.411) | 98.838 (−0.553 [−0.723, −0.388]) |
| | 2-init ens | 95.506 | 95.580 (+0.074) | 95.390 (−0.116) | 95.292 (−0.215 [−0.345, −0.088]) |

Every primary gate passes: both `s43` (−0.752) and `s44` (−0.693) clear
`<= -0.5 meV` with upper 95% bounds below zero, and the 2-init ensemble also
improves (−0.476 [−0.563, −0.392]). Crucially, the gain carries cleanly to the
selection-free unseen block 2.6M rows away: −0.597 / −0.553 single-model and
−0.215 [−0.345, −0.088] on the 2-init ensemble.

### Retained pass-60 weight checks and snapshot inventory (descriptive)

- **Multi-epoch snapshots (`{54, 57, 60}` / `{48..60}`):** not retained on
  disk for either full-scale (`b1024-minus-tail` s42–s45, where `ema_model.pt`
  and `selected_model.pt` are weight-identical) or 500K (`s43`–`s46`). Cancelled
  per `final_ruling.md`.
- **Pass-60 `live60` / `wavg60` / `pavg60` vs `ema60`:** effectively tied on
  both blocks (within 0.00 to +0.04 meV) because cosine LR decayed to `1e-6`
  by pass 60.
- **Best-dev `sel` vs `ema60`:** on `dev`, `s43` `sel` (pass 43) beat `ema60`
  by −0.454 meV, **but on the selection-free `unseen` block `sel` is +1.439
  meV worse than `ema60` (101.011 vs 99.572)**, confirming selection bias on
  shard 10 and supporting full-scale's `final-ema` rule.
- **Uncertainty calibration:** the 32-pass MC std averages 27.5–31.2 meV and
  correlates with `|ema60 - y|` at r = 0.447 / 0.454 (`dev`) and 0.472 / 0.389
  (`unseen`).

## Why it works and limits (attribution)

- **Valid and verified.** `selected_model.pt` replayed saved dev predictions
  within 1e-6 eV (`7.6e-7` / `6.5e-7`). Full-dev `ema60` MAE matched pass 60 in
  `canonical_trace.json` within 1e-4 eV, and `last_checkpoint.pt` SHA256
  matched the pass-60 `checkpoint_identity`. No protected role was touched.
- **Mechanism (supported by monotonicity and transfer):** `OGBGPTransTiny`
  trains 49 dropout sites (`p=0.1`) under L1 loss; deterministic `eval()`
  evaluates `f(E[mask])` rather than `E[f(mask)]` across nonlinear softmax and
  LayerNorm layers. Averaging `K = 32` stochastic passes uses zero labels or
  fitting, so it cannot overfit a block and transfers directly to unseen rows.
- **Limits and cost:** inference costs 32× forward passes (about 6.3 min per
  50K rows per model on an RTX 5060). Ensemble gains shrink versus single
  models (−0.476 on `dev`, −0.215 on `unseen`), because multi-init averaging
  already removes part of the variance. Full-scale 4-init ensemble impact is
  unmeasured and could be smaller (~0.1–0.3 meV); evaluating it once on
  `tail_holdout_v1` requires explicit tail-role authority and a frozen
  protocol.

## Execution and cost

One execution, no retry. Predict wall time: 1,654.3 s (~27.6 min on RTX 5060
for 4 × 50K × 32 passes plus deterministic baselines); score wall time: 25.6 s.
Exclusive GPU device-hours were not measured and stay missing. Tail, official
validation and test sets were untouched.

See [results](results.json) for unrounded numbers. Row predictions stay outside
Git in `D:/molgap-artifacts/diagnostics/pcqm-c3-checkpoint-space/`.
