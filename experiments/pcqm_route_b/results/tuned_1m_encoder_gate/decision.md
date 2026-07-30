# Tuned 1M encoder gate

## Decision

Freeze a mixed Route B encoder set for development-only fusion:

- tuned GPS9;
- tuned GPS11-160;
- fixed-config primary SchNet;
- fixed-config augmented SchNet.

The resumed tuned GPS9 result is accepted and improves development Gap MAE by
`0.003282 eV`; tuned GPS11-160 is also accepted and improves it by
`0.002755 eV`. Both tuned SchNet artifacts are complete and aligned, but they
regress against their fixed-config 1M baselines by `0.000810 eV` and
`0.025365 eV`; they are rejected for fusion selection. The previously submitted
fixed-GPS9 Fusion remains a matched control, while the selected tuned-GPS9
Fusion runs in a separate output namespace.

## Evidence

- Kaggle SchNet acceptance:
  `platforms/_records/kaggle/training/pcqm_route_b_tuned_schnet_acceptance_20260729.json`
- IMS tuned GPS11 acceptance:
  `platforms/_records/ims/pcqm_route_b_migration/tuned_gps11_20260729/mixed_encoder_acceptance.json`
- SCNet-to-IMS tuned GPS9 acceptance:
  `platforms/_records/ims/pcqm_route_b_migration/tuned_gps9_20260729/mixed_encoder_acceptance.json`
- Machine-readable comparison: `metrics.json`

All accepted embeddings contain aligned `915012/81961/4981`
train/development/official rows across 401 parts. Official-validation labels,
official test, the future sealed 20K, and the production registry were not
used.
