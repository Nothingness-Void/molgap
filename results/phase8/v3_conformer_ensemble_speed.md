# Phase 8 v3 Conformer Ensemble Speed

Date: 2026-07-06

- Input molecules: `100`
- Ensemble k: `8`
- Single conformer: `3.05` s total, `0.031` s/valid mol
- k-conformer ensemble: `20.70` s total, `0.207` s/valid mol
- Slowdown: `6.8x` per valid molecule

Decision: keep conformer ensemble as opt-in small/medium-batch inference, not the database-scale default.
