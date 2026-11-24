# Decision: C3 full-scale tail, MC-dropout on four B1024 GPT-T models (2026-11-23)

Rules: [protocol.md](protocol.md), committed (1cff895d7) before any tail row was scored. Each model was scored once on `tail_holdout_v1` (128,606 rows). Official valid/test, sealed 20K and the production registry were not touched. Numbers: [results.json](results.json).

## Verdict: NO EFFECT
The 4-model equal-weight MC-dropout (K=32) ensemble scores **83.784 meV**. The deterministic 4-model ensemble scores 83.992 meV (recorded) and 83.992 meV on the same code path.
**Delta = -0.208 meV** (95% paired bootstrap CI [-0.257, -0.155], 2000 draws). The gate says "no effect" above -0.5 meV, "real" at -1.0 meV or lower, "adopt-worthy" at -1.5 meV or lower. The result misses the "real" bar by 0.8 meV. MC-dropout is not an adopt candidate for the four-model GPT-T ensemble. The gain is statistically clear but small.

## Numbers (tail MAE, meV)
| Model | Deterministic (K=1) | MC K=8 | MC K=16 | MC K=32 | MC32 - det [95% CI] |
|---|---|---|---|---|---|
| s42 | 87.941 | 87.689 | 87.457 | 87.329 | -0.612 [-0.686, -0.538] |
| s43 | 87.846 | 87.501 | 87.267 | 87.172 | -0.674 [-0.749, -0.597] |
| s44 | 87.891 | 87.615 | 87.441 | 87.358 | -0.534 [-0.608, -0.461] |
| s45 | 88.075 | 87.797 | 87.606 | 87.529 | -0.546 [-0.618, -0.473] |
| 4-model ensemble | 83.992 | 83.888 | 83.820 | 83.784 | **-0.208 [-0.257, -0.155]** |

Single models gain 0.53-0.67 meV each (mean 0.59). All four are real in the paired sense, and the K curve is still falling at K=32 (ensemble 83.888, 83.820, 83.784), so more passes would add at most a few hundredths of a meV.
Code-path check: the deterministic path here matches the earlier scorer's saved tail predictions to 7e-6 eV (max over rows), and the same-path ensemble equals 83.992 meV. No code-path confound.

## Attribution (descriptive)
- **Most of the single-model gain is the same variance an ensemble already removes.** The single-model gain of 0.59 meV shrinks to 0.21 meV for the 4-ensemble (35% kept). The 500K C3 probe kept 64% for a 2-init ensemble (-0.75/-0.69 single, -0.476 pair on dev). The fraction falls as members are added: 4 members leave less variance to remove.
- MC averaging makes members agree more: mean spread between members drops 26.46 to 25.06 meV, and pairwise residual correlation rises 0.908 to 0.919. Averaging dropout passes pulls each member toward a common smoothed function, which is what the ensemble was already doing across inits.
- **MC std tracks error.** Pearson 0.41-0.42, Spearman 0.46 between MC std (mean 22.3-23.0 meV) and the deterministic absolute error, for each model. The gain sits in the high-std rows: by MC-std quartile (low to high) the single-model gain is about -0.1 to -0.2, -0.3 to -0.4, -0.4 to -0.5 and -1.2 to -1.5 meV. The top quartile carries most of it.
- MC noise left in the 4-member K=32 mean is about 2.6 meV in std per row (from the per-row MC std), small against the 84 meV error.
- The shrink against dev is not a failure of transfer: the single-model gain on the full-scale tail (0.59) is close to 500K dev (0.69-0.75). The ensemble is what cuts it.

## Limits
- One tail, one pass, row bootstrap only. The CI covers row sampling, not model or mask randomness. Masks differ per model and chunk; one mask seed set was used.
- The four members differ only by init (data order and RNG seed 42 in training), so their errors are highly correlated (residual correlation 0.91). A more diverse ensemble might leave more room for MC, or less.
- Cost: K=32 is 32x the inference cost (about 47 min for 4 models on 128,606 rows on the RTX 5060) for 0.21 meV.

## Process notes
- Predict stage (label-free) started before the RML prospective trajectory was published; the trajectory was committed (8121b8465) while predictions ran and before the score stage read any label. The tail graphs had been read as inputs by then, labels had not.
- The tail scorer module and recipe code were imported read-only from worktree `gptrans-full-minus-tail-20261110` (sha 756227a3, the one that produced the recorded scores).
- Smoke before the real run: rows [0, 512), K=16 (the reused `run_mc` returns K=8/16 snapshots, so K<16 crashes). It reproduced the saved development predictions to 1.4e-6 eV. In-train rows, so its MC numbers mean nothing.
- Leaderboard: added `gptrans_minus_tail_s4{2..5}_mc32`, `gptrans_minus_tail_ens4_mc32`, `gptrans_minus_tail_ens4_det` with `tail_leaderboard_add.py` (ensemble rows carry a composite checkpoint hash: sha256 of the four `ema_model.pt` hashes joined in s42..s45 order). Board file hash after: ff139b8d...cfcf5b5. Rows saved in `leaderboard_eval.json`.
- Row outputs: `D:/molgap-artifacts/diagnostics/pcqm-c3-full-tail-20261123/tail/` (64 chunk files, `models.json`, `results.json`), not in Git.
- Wall time of the predict stage: 2808 s.
