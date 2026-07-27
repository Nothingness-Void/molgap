# 1M Replay-Weighted Fusion

This is the first controlled follow-up to the 1M PCQM component diagnosis. It
freezes all 1M encoder embeddings and trains only the dual-GPS `FusionHead`.
Rows with `source_idx < 500000` receive weight 2.0, yielding approximately
two-thirds old-500K replay draws per epoch. It must not be interpreted as an
encoder retrain or a model-version candidate until the resulting head passes
the shared external OOD/hard and PCQM public-valid checks.

The Kaggle input is the existing private `1m-full` dataset plus the small
`molgap-runtime-source` package. Outputs are a FusionHead checkpoint, train
log, and internal split metrics only. A separate external evaluation uses the
checkpoint with the fixed 1M encoder weights.
