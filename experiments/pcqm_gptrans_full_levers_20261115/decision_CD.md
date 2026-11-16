# Arms C (20 layers) and D (90 passes): tail verdict (2026-11-16)

Rules: [protocol.md](protocol.md) (frozen). Verdict code `verdict_levers.py`, committed (cf25ee6c) before either number below was computed. Evidence: `D:/molgap-artifacts/tail_eval_full_levers_20261116/` (`gptrans_c.json`, `gptrans_d.json`, `verdict_CD.json`, `preds/`).

## Retrieval accepted
Both runs COMPLETE (`training_complete.json`: C 190,380 steps / 60 passes, D 285,570 steps / 90 passes, selection final-ema, batch 1024). Remote sha256 equals local for every retrieved file, including `ema_model.pt` and the init (C `8ee2d6aa...` 9,529,761 params / layers=20; D s42 `27590c1a...`). Rows [0, 3,250,000). Scored once each, `ema_model.pt`, `--weights-key model`, `--run-config`; C with `MOLGAP_GPTRANS_LAYERS=20`. Scorer sha `756227a3...` (same as s44/s45).

## Result (tail_holdout_v1, 128,606 rows)
| Arm | Tail MAE (meV) | vs M 87.938 | Bar M - 1.5 = 86.438 | Paired vs B1024 s42 (99% upper) | Outcome |
|---|---|---|---|---|---|
| C, 20 layers | 87.048 | -0.89 | not met | -0.893 (-0.517) | FAIL |
| D, 90 passes | 87.130 | -0.81 | not met | -0.811 (-0.419) | FAIL |

sigma 0.099 meV (bar = max(1.5, 3 sigma) = 1.5). Both gains are real against the paired bound but below the pre-registered 1.5 meV bar, so neither lever is adopted. Dev MAE (50K, not comparable with tail): C 63.49, D 61.42 meV.

## Cost (measured, `sessions.jsonl`)
C 13.82 A100 h; D 12.93 A100 h. Credits not measured per job.

## Still open
B256 s42/s43 hit walltime at 59/60 passes (steps 754,256 and 751,388 of 761,700); resubmitted unchanged 2026-11-16 (jobs 2016619/2016620, no copy was running). B and E pending. Official valid: not read; one read of the final best model after every lever result is in.
Trajectories `c_l20_s42`, `d_p90_s42` and CURRENT_STATE are closed together with B/E (one RML rebuild).
