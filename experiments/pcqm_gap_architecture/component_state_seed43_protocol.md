# Persistent ComponentState seed-43 Kaggle confirmation

Protocol date: 2026-09-07.

## Question

Does the seed-42 ComponentState gain survive one independent seed and a change
from Kunshan DCU to Kaggle T4, when compared inside the same paired job?

The control retains the deterministic eight-channel conjugated-component
descriptor. The candidate changes only one mechanism: it adds a persistent
32-channel state per conjugated component, with atom-to-component mean updates
and component-to-atom returns after blocks 3, 6, and 9. Both retain the same
no-attention GraphState9 body, real-bond EdgeState, sparse wedge state,
RWSE16, ETKDGv3/MMFF94s geometry and direct Gap head.

## Frozen contract

- Official-PCQM-train-derived 100,000 train / 10,000 internal-validation rows.
- Seed 43, FP32, batch 48, workers 0, AdamW `1.6e-4`, weight decay `1e-6`.
- Train-standardized L1, cosine decay to `1e-6`, 40 epochs, patience 8.
- Random initialization; no checkpoint reuse, warm start, pretraining,
  residual target, prediction fusion, or official-role access.
- One private Kaggle T4x2 kernel; each model has one isolated GPU, RNG,
  optimizer, checkpoint directory and output.
- The deterministic component cache is built by a private CPU kernel from the
  already accepted geometry cache and must pass no-model acceptance before GPU
  submission.

The paired candidate must be strictly better than its fresh descriptor control
to retain confirmation status. Record the `0.001 eV` material threshold
separately; this one job does not authorize seed 44, full-data training,
official validation/test-dev, production changes or molecular-research-server
access.
