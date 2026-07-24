# Phase 8 30K Common Evaluation

Question: does the replacement 30K data distribution generalize better than the
old 30K prefix when both are evaluated on the same molecules?

## Setup

- Models: `old30k` vs `replacement30k`
- Architecture: GPS 2D + SchNet 3D + standard single `FusionHead`
- Evaluation set:
  - Phase 7 OOD-1000: 999 valid molecules after ETKDG
  - P8 targeted hard slice: 981 valid molecules after excluding both 30K training prefixes
- Total valid evaluation molecules: 1,980

## Hybrid Results

Lower MAE is better. Delta is `replacement30k - old30k`; negative means the
replacement data is better.

| Eval set | old30k avg MAE | replacement30k avg MAE | delta avg | old30k Gap MAE | replacement30k Gap MAE | delta Gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 0.21828 | 0.21612 | -0.00216 | 0.26823 | 0.26721 | -0.00102 |
| OOD-1000 | 0.20010 | 0.20043 | +0.00033 | 0.23908 | 0.24121 | +0.00213 |
| P8 targeted hard | 0.23678 | 0.23210 | -0.00469 | 0.29792 | 0.29370 | -0.00422 |

## Component Check

| Model family | Predictor | all avg MAE | all Gap MAE | OOD avg MAE | P8 hard avg MAE |
| --- | --- | ---: | ---: | ---: | ---: |
| old30k | GPS 2D | 0.23409 | 0.28516 | 0.21150 | 0.25710 |
| old30k | SchNet 3D | 0.24027 | 0.30257 | 0.22813 | 0.25263 |
| old30k | Hybrid | 0.21828 | 0.26823 | 0.20010 | 0.23678 |
| replacement30k | GPS 2D | 0.23475 | 0.28364 | 0.21127 | 0.25865 |
| replacement30k | SchNet 3D | 0.24232 | 0.30371 | 0.23359 | 0.25121 |
| replacement30k | Hybrid | 0.21612 | 0.26721 | 0.20043 | 0.23210 |

## Decision

The replacement distribution is not a broad OOD win at 30K: OOD-1000 is a tie
within noise. It is, however, directionally positive on the exact hard chemistry
that Phase 8 targeted.

This is enough to justify one full replacement300K standard hybrid run if compute
budget is available, but expectations should be modest. The next full run should
remain the standard single `FusionHead`; MoE is still deprioritized.
