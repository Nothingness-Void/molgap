# Degree-balanced K1 — terminal decision (2026-09-25)

The fixed Kaggle1 seed-42 arm completed 40 epochs with 3,660,545 parameters,
FP32/no-TF32 and physical BS128 on one Tesla T4. The source, runtime, selected
checkpoint, recovery files, aligned 50K development payload and trace passed
hash-bound no-inference acceptance. Official validation and test roles stayed
sealed.

The selected development Gap MAE was **0.1400965452 eV** (epoch 39), versus
immutable K1-v4 at **0.1413736343 eV**. Its **0.0012770891 eV** gain missed the
prospectively frozen **0.003 eV** gate, despite a favorable paired row-bootstrap
interval. This is `POSITIVE_BELOW_GATE`, not promotion.

The separate NO_TRAIN audit reproduced the original predictions within
0.001 eV, then evaluated frozen weights on the predeclared fixed500K internal
development rows. The candidate scored **0.1434239894 eV**, versus K1-v4 at
**0.1412533075 eV**: a **0.0021706820 eV regression**. The paired row-bootstrap
interval for candidate-minus-reference error was [0.0013697955, 0.0028863831]
eV. This role had been used elsewhere in the project, so this is a portability
diagnostic, not a newly independent held-out test or 500K training claim.

The zero-start mechanism was trainable; this closes only this graph-relative
degree multiplier. No seed, scale or protected-role release followed.
Method: [`../../protocol.md`](../../protocol.md). Compact saved-prediction
analysis: [`../../results/portability_analysis.json`](../../results/portability_analysis.json).
