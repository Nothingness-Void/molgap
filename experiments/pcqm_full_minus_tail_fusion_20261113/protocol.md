# Full-scale minus-tail K1 + GPT-T fusion (frozen 2026-11-14, before any result)

Authorization: user, 2026-11-13/14. Multi-seed is NOT affordable at full scale (user decision), so no second K1 seed and no homogeneous baseline are trained.

## Members
- K1 Muon, B1024, lr 2.4e-3, 60 passes, EMA 0.999, rows [0, 3,250,000), seed 42; endpoint = pass-60 live (`selected_model.pt`).
- GPT-T B1024 (to be built later), rows [0, 3,250,000), seed 42; endpoint = pass-60 EMA. Recipe frozen separately before its run.
- Runner: platforms/colab/pcqm_k1_b1024_production/train_full_b1024.py (commit 5e6f12c7 or later). `--max-shards 65`.

## Data rules
- tail_holdout_v1 (rows 3,250,000..3,378,605) is never trained on and never used for selection; each endpoint is scored on it exactly once.
- No official valid/test is read for selection. Snapshot curves are descriptive only; no snapshot is chosen.
- Probe subset (20,480 in-train rows, seed 12345) is a training-fit diagnostic, not generalisation evidence.

## Fusion criterion (fixed before results)
Weights 0.5/0.5, not fitted. PASS requires all of:
1. tail MAE(fusion) - tail MAE(best single member) <= -3 meV;
2. paired row bootstrap (2000, seed 42) 95% upper bound of that difference < 0.

Dropped criterion: "fusion beats a homogeneous (same-architecture, other-seed) fusion by >= 3 meV". It cannot be evaluated without a second seed. Consequence: a PASS shows an ensembling gain, NOT that heterogeneity is the cause. Prior evidence at 500K: heterogeneous fusion beat the best single by 4.38 meV but beat two-GPT homogeneous fusion by only 1.22 meV (failed the 3 meV criterion). Interpret accordingly.

## Outcome vocabulary
PASS / FAIL on the two conditions above; a missing member checkpoint or incompatible scoring = INCONCLUSIVE. No re-tuning and no re-run after seeing tail scores.
