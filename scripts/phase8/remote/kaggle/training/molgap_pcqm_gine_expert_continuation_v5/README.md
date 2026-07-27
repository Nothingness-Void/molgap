# PCQM GIN Expert Continuation V5

This bounded Kaggle continuation resumes the accepted benchmark-specific
Gap-only GIN virtual-node expert using OGB molecular features.

- Training source: deterministic 250K sample from the official PCQM4Mv2 train
  interval in the public `piero0/pcqm4mv2` mirror.
- Development split: scaffold-disjoint hash split inside that sample.
- Final check: first fixed 5K rows of the official validation split.
- Exclusions: official test splits, MolGap future sealed 20K, and B3LYP
  production-model promotion.
- Durability: atomic graph shards, progress manifest, best/last checkpoints,
  train log, metrics, and completion manifest.

Version 5 resumes the accepted version 4 epoch-49 optimizer, scheduler, scaler,
and model state from the immutable private dataset. It reuses all 11 graph
shards and trains at most epochs 50-69, subject to the existing seven-epoch
early-stop rule. It is retained only if it improves the frozen version 4
official-valid 5K Gap MAE of `0.1965981659 eV`; otherwise version 4 remains the
accepted PCQM task expert. Official test and the MolGap sealed 20K stay locked.
