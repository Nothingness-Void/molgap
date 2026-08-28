# Round 4 Matched-Cost Attribution

After the frozen mixed-architecture ensemble passed its single-control gate,
the three already accepted control predictions were aligned and averaged as a
zero-training matched-cost comparator.

| Three-pass candidate | Development MAE | Radical MAE | Non-radical MAE |
|---|---:|---:|---:|
| Three independent control seeds | **0.142329** | **0.238701** | **0.134357** |
| Mixed architecture, best replicate (seed 42) | 0.143317 | 0.240734 | 0.135258 |
| Mixed architecture, three-replicate mean | 0.144552 | 0.241080 | 0.136567 |

The ordinary three-seed control ensemble is better on every reported subset
at the same three encoder passes. The Round 4 gate remains a valid statement
that the frozen mixed ensemble beats one matched control, but it is not
evidence that radical context is the source of the gain.

Therefore the radical-context architecture is not promoted. The reusable
finding is narrower: independent-seed ensembling is the strongest bounded
accuracy-mode option measured on this official-train-only split. No additional
architecture screen or full radical-context training is justified.
