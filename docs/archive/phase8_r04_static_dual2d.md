# Archive-r04: Dual-2D Static Blend

## Hypothesis

After removing the weak Geometry expert, independently initialized Local GINE6
and Global GPS9 might have sufficiently complementary errors for frozen
target-wise static blending to improve beyond the better single expert.

## Internal Signal

On the 30k scaffold-disjoint pilot, static blending improved the best single
expert's Gap MAE for seeds 42/43/44 by
`0.001303/0.012029/0.003400` eV. Equal averaging, embedding concat Fusion, and
both dynamic gates did not pass. This justified one external transfer audit,
not model promotion or a larger training run.

## External Gate

Frozen internal weights and each seed's predeclared internal Local reference
were applied unchanged to common, OOD-1000, P8-hard, and PCQM-like sets. The
gate required non-regression for every seed and set.

The rule failed: seed 42 regressed on OOD-1000 by `0.000698 eV`, seed 44
regressed on P8-hard by `0.000571 eV`, and seed 43 regressed on PCQM-like by
`0.000228 eV`. These are small but directly contradict stable transfer. The
30k experts also remain far behind routed-v4 in absolute Gap MAE, for example
common `0.228-0.236 eV` versus v4 `0.121896 eV`.

## Decision

**STOP.** Do not scale this architecture, joint-finetune a Router, open sealed
sets, or allocate a production version. Routed dual-GPS v4 remains the B3LYP
accuracy predictor.

Exact records: `results/phase8/archive/archive-r04-static-dual2d/`
`dual2d_decision.md` and `external_transfer_decision.md`.
