# Decision: C4 full-scale weight decay 0.022 (2026-11-23)

Rules: [protocol.md](protocol.md), frozen before any result (commits fc6cdc2e7, 005c2ca13). The final-EMA weights were scored once on `tail_holdout_v1` (128,606 rows) with `platforms/local/pcqm_tail_holdout_eval.py` (sha 756227a3, same as s42-s45), through the guard `score_c4_once.py`. Official valid/test, sealed 20K and the production registry were not touched. Numbers: [results.json](results.json).

## Verdict: NO EFFECT (and slightly worse)
Tail MAE: C4 **88.102 meV**; reference s43 B1024 (weight decay 0.05) **87.846 meV**.
**Delta = +0.256 meV** (positive is worse). Paired row bootstrap, 2000 draws, seed 42, 95% interval [-0.074, +0.584]; C4 is better on 49.8% of rows.

| Gate | Needs | Result |
|---|---|---|
| Real effect | Delta <= -1.0 and upper bound < 0 | Not met (Delta +0.256, upper +0.584) |
| Adopt-worthy | Delta <= -1.5 | Not met |
| Inconclusive | -1.0 < Delta <= -0.5 | Not in this band |
| No effect | Delta > -0.5 | **Met. Closed.** |

Lowering weight decay from 0.05 to 0.022 does not help at full scale. Do not adopt it.

## Acceptance of the retrieved run
Directory `D:/molgap-artifacts/retrieved/gptrans-c4-wd022-s43-20261123/`.
- All 10 fetched files match the remote sha256 (`remote_sha256.txt`, `local_sha256.txt`). EMA weights sha e19c0a01...2aac.
- `training_complete.json`: COMPLETE, 60 passes, 190,380 steps, selection final-ema. 60 log records, nonfinite_params 0 in all.
- `run_config.json` (sha 51b99b3c...) and `run_provenance.json` are byte-identical to the s43 reference. Rows [0, 3,250,000); tail, official valid and test not read. Same init (file sha 2ccd1eb7...5f3c, state sha c95da412...5f5b), same runner sha 120d5571...5b37.
- Weight decay is not in `run_config.json`. `arm_record.json` records `weight_decay 0.022`. The reference value is 0.05, hard-coded in `train_full_b1024.py` (line 203; the reference root has no `arm_record.json`). Nothing else differs in the saved recipe.
- `selected_model.pt` and `ema_model.pt` hold identical tensors (337). The scorer read `ema_model.pt` with `--weights-key model`, as for s42-s45.

## Attribution (descriptive, one run)
Same init, data order and dropout seed, so the pair isolates weight decay. The change in the training curve matches the change in the setting:
| | C4 (0.022) | Reference (0.05) |
|---|---|---|
| Train MAE, pass 60 | 69.4 meV | 72.4 meV |
| In-train probe (20,480 rows), pass 60 | 63.1 meV | 67.0 meV |
| Dev MAE, rows 0-50,000 (in-train), final EMA | 62.0 meV | 66.7 meV |
| Tail MAE (held out) | 88.102 meV | 87.846 meV |
| Tail minus dev | 26.1 meV | 21.2 meV |
| Live weight norm, pass 60 | 314.6 | 176.1 |

- Less decay fits the training data better by 3 to 4.7 meV on every in-train measure, and weights grow 1.8 times larger. The held-out tail does not follow: it is 0.26 meV worse. This reads as a small widening of the train-to-tail gap (more fitting, no more generalisation), not as underfitting at 0.05. A weight-decay gain from under-regularisation would need the tail to fall with the train error; it did not.
- Weight decay alone is a plausible cause of the in-train shift (it is the only changed setting). It is not a plausible cause of any tail gain, because there is none. The tail shift (+0.26) is inside the 95% interval and about 2.6 times the init-only noise std (0.099 meV), so it is also compatible with run noise. No cause is claimed for it.
- The 0.022 value came from matching decay per pass to the 500K recipe. The result says that match does not carry to full scale. Other values were not tested; only "0.022 versus 0.05" is closed.

## Limits
- One init, one run. Init-only noise std (0.099 meV) used a fixed data order and dropout seed, so true run noise is larger. The bootstrap covers row sampling only.
- The tail was used to choose among earlier levers, so absolute tail numbers are optimistic. The paired difference is not affected.
- The bootstrap seed (42) follows the levers protocol; this protocol did not name one. The reference per-row predictions are the s43 file saved by the same scorer (`tail_eval_full_minus_tail_20261115/preds`); the reference was not rescored. Its mean over rows is 87.84585 meV, equal to the recorded 87.84585.

## Cost
Measured A100 time from `sessions.jsonl`: 8.257 h (one session, 2026-11-23 05:11:53 to 13:27:18 UTC). Reference s43: 8.410 h. Queue and CPU time not measured.

## Evidence (under `D:/molgap-artifacts/`)
| File | sha256 |
|---|---|
| tail_eval_c4_wd022_20261123/gptrans_c4_wd022_s43.json | 0f7a14b5ff6160dee58ff3938135ec9a583903d1f4b8a925cbc20fd312fba8bf |
| tail_eval_c4_wd022_20261123/preds/gptrans_c4_wd022_s43_tail_holdout_v1.pt | 515a7a182456bf809c523b5c1bb94432d7268ac6c4e8db0f1384995958fc31d4 |
| tail_eval_full_minus_tail_20261115/preds/gptrans_minus_tail_s43_tail_holdout_v1.pt (reference) | 38add7b00b8e806834b516edb81427c1f321aa748459ab49478a0b38a0d96671 |
| tail_holdout_master_leaderboard.json after adding `gptrans_c4_wd022_s43` (from ff139b8d...cfcf5b5) | 98aa8c1722ded06ca7fe0d6f488386bdc044b209ccfd3e5cf1ae3f8d2f1ef541 |

The leaderboard row was added with `platforms/local/tail_leaderboard_add.py`; the eval file is copied to `leaderboard_eval.json`. Scripts: `score_c4_once.py`, `bootstrap_c4.py`.
