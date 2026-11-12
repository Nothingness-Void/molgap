# Status

2026-11-12: protocol frozen before any scoring. Training on Colab L4 started by the user
(seed 42 run runs/gptrans-b1024-500k-s42-20261112 at least at pass 10 when the rule was written;
seed 43 run runs/gptrans-b1024-500k-s43-20261112). Nothing scored, no B1024 result seen.
Next: retrieve both ema_model.pt, score each once on tail_holdout_v1, apply the rule.

2026-11-13: both runs COMPLETE (60 passes, 29,280 steps, batch 1024, EMA 0.992028 per run_config; torch 2.11.0 not recorded in run_config, unverified).
Erratum commit 1a45b228 (protocol EMA 0.999 -> 0.999^8) made before any scoring.
Tail scoring, one per ema_model.pt (--weights-key model), tail_holdout_v1, 128,606 rows:
- s42: 111.672 meV (D:/molgap-artifacts/tail_holdout_eval_gptrans_b1024_500k_s42.json)
- s43: 110.958 meV (D:/molgap-artifacts/tail_holdout_eval_gptrans_b1024_500k_s43.json)
- mean 111.315 meV; anchor 107.94; thresholds 109.44 (mean) and 110.94 (each).
Epoch-60 train MAE: s42 0.07109 eV, s43 0.068894 eV (batch-128 60-pass train logs not found under retrieved/).
Residual report (descriptive): D:/molgap-artifacts/tail_residual_gptrans_b1024_500k.json; Pearson b1024 s42/s43 0.875; vs batch128 s43/s44 0.857-0.864.
Rule result: REJECT. Next per protocol: B512 on the same two seeds.
