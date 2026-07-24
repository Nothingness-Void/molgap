# PCQM GIN Expert Pilot

This bounded Kaggle pilot trains a benchmark-specific Gap-only GIN virtual-node
expert using OGB molecular features.

- Training source: deterministic 250K sample from the official PCQM4Mv2 train
  interval in the public `piero0/pcqm4mv2` mirror.
- Development split: scaffold-disjoint hash split inside that sample.
- Final check: first fixed 5K rows of the official validation split.
- Exclusions: official test splits, MolGap future sealed 20K, and B3LYP
  production-model promotion.
- Durability: atomic graph shards, progress manifest, best/last checkpoints,
  train log, metrics, and completion manifest.

Version 4 resumes the accepted version 3 epoch-29 optimizer, scheduler, scaler,
and model state from the private complete-output dataset. It reuses all 11 graph
shards and trains only epochs 30-49, subject to the existing early-stop rule.
The continuation passes only if fixed official-valid 5K Gap MAE is at most
`0.20 eV`. Scaling is a separate decision after artifact acceptance.
