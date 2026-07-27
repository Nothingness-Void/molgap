# PCQM GPS9 Architecture Pilot

This bounded Kaggle experiment compares an OGB-feature GPS9 encoder against
the accepted GINE v4 on exactly the same deterministic 250K sample,
scaffold-disjoint development split, and fixed official-validation 5K.

- Encoder: 9 GPS layers, 320 hidden channels, 8 attention heads (about 12M
  parameters, close to the capacity class of the published GPS baseline).
- Input: accepted GINE v4 graph shards; no graph rebuild.
- Training: at most 40 epochs with seven-epoch early stopping.
- Gate: improve the frozen GINE v4 official-valid 5K Gap MAE of
  `0.1965981659 eV`.
- Exclusions: official test, MolGap sealed 20K, and production promotion.

The output contains best/last checkpoints, an atomic epoch log, fixed 5K
predictions, metrics, and a SHA256 completion manifest. Passing this pilot
authorizes a separately staged 1M/full-PCQM scale experiment; it is not itself
a leaderboard result.
