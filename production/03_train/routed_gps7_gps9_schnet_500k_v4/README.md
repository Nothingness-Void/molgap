# Routed GPS7 + GPS9 + SchNet 500K V4

This directory contains the selection and evaluation evidence for the routed
dual-GPS v4 B3LYP predictor. It is still registered and loadable, but it is the
previous production baseline; the recommended predictor is only in
`CURRENT_STATE.md`.

- Base path: GPS7 + SchNet 500K v3.
- Routed expert: GPS9.
- Rule: invoke the dual-GPS fusion only when the base v3 predicted Gap is below
  4 eV.
- Training data: the same expansion 500K labels and SchNet branch as v3.
- Decision: `gps_arch_routed_decision.md`.

Model checkpoints remain under `models/`; this directory owns metrics,
decisions, and locally ignored evaluation intermediates.

Large frozen embeddings are stored under
`data/cache/production/routed_gps7_gps9_schnet_500k_v4/embeddings/`.
