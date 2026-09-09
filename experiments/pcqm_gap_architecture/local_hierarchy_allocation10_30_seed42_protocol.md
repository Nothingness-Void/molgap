# EdgeState Local-Hierarchy 10/30 Allocation Seed-42 Protocol

## Question

Does reallocating the fixed 40-epoch encoder-exposure budget from 20/20 to 10
local-hierarchy pretraining epochs and 30 direct-Gap epochs turn the observed
convergence acceleration into a material final improvement?

The preceding 20/20 result is closed in
`results/local_hierarchy_pretraining_seed42/decision.md`. This is a new training
schedule question, not an implementation retry.

## Single changed variable

- previous candidate: 20 pretraining + 20 Gap epochs;
- new candidate: 10 pretraining + 30 Gap epochs.

Scratch remains 40 direct-Gap epochs. Every arm therefore retains exactly 40
encoder passes over the 100,000-row train role. Model architecture, seed-42
initialization, 100,000/10,000 row identities, label sidecar, OGB atom/bond
features, RWSE16, EdgeState64, 9x192 encoder, dropout 0.1, mean pooling, FP32,
physical batch 48, AdamW, learning rate `1.6e-4`, weight decay `1e-6`, cosine
schedule, normalized-L1 Gap loss, evaluator, and best-checkpoint rule are
unchanged. Official validation, test-dev, and shadow remain unread.

The accepted local-label aggregate is
`e7d0557d7ade3469d25d91512fbe0900c7441db94427bce3ed5bc1faa74218a6`.

## Execution and gate

One Kaggle T4x2 job isolates fresh scratch and 10/30 candidate workers. The
candidate must improve the paired scratch validation Gap MAE by at least
`0.003 eV` to earn nomination. Nomination alone does not authorize shadow,
extra seeds, 1M/full training, official roles, desktop handoff, or molecular-
research-server access. Failure closes local-hierarchy budget allocation.
