# Audit Preparation (2026-09-06)

Local source review found that the official-full EdgeState configuration
defaulted to dropout 0.05 while the geometry factory fixed dropout 0.10. The
actual checkpoint setting required scheduled verification. The historical
warm-start used FP16, whereas the accepted 100K screen used FP32.

These are potential confounders, not established causes of the observed MAE
regression. The prior single-batch initialization test was insufficient to
exclude every input/evaluation discrepancy. The authorized P0 audit was
prepared to resolve these uncertainties before any scratch successor.

No new accuracy outcome was available at preparation. Historical negative
evidence remains in `../pcqm_geometry_warmstart/eight_epoch_result.json`.
