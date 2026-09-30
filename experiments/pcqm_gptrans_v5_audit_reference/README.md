# GPTrans 100K V5 audit reference

This is the user-authorized evidence-complete rerun of the frozen V4 GPTrans-T
100K recipe, not a new architecture screen or replacement of the historical
V4 result. The 100K train / 50K internal-development fixed graph cache,
seed 42, 60 epochs, physical batch 128, FP32, optimizer, schedule, and
EMA-selected model are unchanged. The additive observation is live-model
development MAE at every epoch. The existing online training MAE and EMA
development MAE are retained. No official validation or test role is read.

`kaggle_t4/run.py` is a thin mount, T4-isolation, preflight, and segmented
resume adapter. Each job runs at most 10 epochs, emits an atomic checkpoint,
canonical RML trace, bounded checkpoint chunk, and SHA manifest. The second
allocated T4, if present, remains idle and is still counted as allocated cost.

The historical reference remains in
[`../pcqm_gptrans_t_100k_v4/decision.md`](../pcqm_gptrans_t_100k_v4/decision.md).
Its completed independent acceptance and RML closure are in
[results/terminal/decision.md](results/terminal/decision.md). Even after
acceptance, this observation-only rerun is not candidate gain.
