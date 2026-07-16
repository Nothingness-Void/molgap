# Archive-r04 Static Dual-2D Decision

## Internal Pilot

Local GINE6 and Global GPS9 were initialized from scratch and evaluated on a
30k scaffold-disjoint pilot. Target-wise static blending improved the best
single expert's Gap MAE for seeds 42/43/44 by
`0.001303/0.012029/0.003400` eV. Equal averaging, embedding concat Fusion, and
both dynamic gates did not pass.

## Frozen External Transfer

The three seed-specific weights and each seed's internal Local reference were
frozen before evaluating common, OOD-1000, P8-hard, and PCQM-like data. The
gate required Gap non-regression for every seed and block.

| failure | static minus reference Gap MAE |
|---|---:|
| seed42, OOD-1000 | +0.000698 eV |
| seed44, P8-hard | +0.000571 eV |
| seed43, PCQM-like | +0.000228 eV |

The 30k experts are also not competitive with routed-v4 in absolute terms:
common Gap MAE is `0.228-0.236 eV` for the static blends versus `0.121896 eV`
for routed-v4.

## Decision

**STOP.** The internal complementarity does not transfer stably. Do not scale
this architecture, joint-finetune a Router, open sealed sets, or allocate a
production version. Routed dual-GPS v4 remains the B3LYP accuracy predictor.

Exact transfer metrics: `external_transfer_metrics.json` and
`external_transfer_decision.md`.
