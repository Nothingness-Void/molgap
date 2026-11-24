# C3 full-scale tail: MC-dropout on four B1024 GPT-T models

Frozen 2026-11-23, before any tail row is scored. Local only (RTX 5060). No training, no remote job.
Parent: `codex/exp/pcqm-c3-checkpoint-space` (500K C3 probe, MC-dropout K=32 PASS on dev rows).
Question: does MC-dropout still help the full-scale models on the protected tail?

## Models and rows
- Models: GPT-T B1024 minus-tail s42, s43, s44, s45 (final EMA weights, `ema_model.pt`, key `model`; equal to `selected_model.pt`). All four are local under `D:/molgap-artifacts/retrieved/`. Trained on rows [0, 3,250,000).
- Rows: `tail_holdout_v1`, source rows [3,250,000, 3,378,606) = 128,606 rows. Graphs come from the stored topology shards 65-67.
- Recipe: rebuilt from each run's `run_config.json` by the same scorer code that produced the recorded scores (`pcqm_tail_holdout_eval.py`, sha 756227a3, in worktree `gptrans-full-minus-tail-20261110`).

## Method (code: `mc_tail.py`)
- MC: reuse `probe_c3.run_mc`. Only `nn.Dropout` modules (49, p=0.1) in train mode; LayerNorm stays in eval. K=32 passes, batch 128, FP32, no TF32. Seeds differ per model and chunk (`20261123 + model*10M + chunk*1000 + k*100000 + batch`), so masks are not shared between models.
- Deterministic (K=1): `probe_c3.run_det` on the same prepared batches, same weights, same process. This is the paired baseline for each single model.
- The predict stage never reads labels. It writes one file per model and 8,192-row chunk, and resumes after a stop. The score stage reads labels once and writes `results.json`; both stages refuse to run if `results.json` exists. The tail is scored once per endpoint.
- Before the tail run: smoke on non-tail rows [0, 512), K=16 (the reused code needs K>=16 for its K=8/16 snapshots), checking the deterministic path against the saved development predictions of the earlier scorer. Smoke output is not evidence.
- Not touched: official valid/test labels, sealed 20K, production registry. num_workers stays 0 (no spawn).

## Endpoints and gate
Primary: tail MAE of the equal-weight 4-model MC (K=32) ensemble minus 83.992 meV (the recorded 4-model deterministic ensemble, master leaderboard and `decision_A.md`).
Delta = primary difference in meV (negative is better).

| Delta | Verdict |
|---|---|
| <= -1.5 | ADOPT-WORTHY |
| <= -1.0 (and > -1.5) | REAL |
| > -0.5 | NO EFFECT |
| between -1.0 and -0.5 | WEAK (not real, not null) |

The same-path deterministic ensemble is also computed and must agree with 83.992 within 0.01 meV; if it does not, the bootstrap comparison on the same path decides and the gap is reported.

Secondary (descriptive, same gate wording not applied): each single model, MC K=32 minus its own deterministic score (same code path); paired row bootstrap, 2000 draws, seed 20261123, 95% CI over rows, for each single model and for the ensemble (MC minus same-path deterministic). K=8 and K=16 curves. Attribution: MC-std versus absolute error (Pearson, Spearman), gain by std quartile, residual correlation between members with and without MC, MC noise left in the 4-member mean, and comparison with the 500K C3 dev gains (-0.752 / -0.693 single, -0.476 two-init ensemble).

## Rules
- No change of K, members, weights, seeds or thresholds after any tail number is seen. No rerun on a scientific miss.
- A crash is resumed from the chunk files with the same code; a code fix needs a note in `decision.md`.
- Row outputs stay outside Git in `D:/molgap-artifacts/diagnostics/pcqm-c3-full-tail-20261123/`. The tail score rows go to the master leaderboard with `tail_leaderboard_add.py` (new names `*_mc32`, `*_det_samepath`).
