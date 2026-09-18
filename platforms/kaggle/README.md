# Kaggle Adapter

Cross-experiment Kaggle packages are grouped by workload role:
`acquisition/`, `training/`, and `evaluation/`. Experiment-specific kernels
stay with their owning experiment.

Kaggle provides the server-side queue. Submit a durable kernel directly,
verify its remote state, and do not create a local slot-trigger process.
Retrieved outputs and acceptance evidence belong in
`platforms/_records/kaggle/`.

The two earlier Kaggle accounts hold accepted, byte-identical fixed OGB PCQM4Mv2
100K/500K graph datasets. Evidence is under
`platforms/_records/kaggle/pcqm_fixed_datasets_v1/` and
`platforms/_records/kaggle/pcqm_fixed_datasets_kaggle2_v1/`. Both accounts are
excluded from holding the 1M and full identities.

Kaggle3 (`nvoid912`) additionally holds accepted private mirrors at
`nvoid912/pcqm4mv2-ogb-fixed-100k-v1` and
`nvoid912/pcqm4mv2-ogb-fixed-500k-scnet-v1`, plus the reusable source layer
`nvoid912/molgap-v5-desktop-runtime`. Evidence is under the corresponding
`platforms/_records/kaggle/*kaggle3_v1/` directories. The Kaggle3 batch API did
not allocate TPU v5e-8 despite correct remote metadata; TPU remains gated by an
interactive hardware and framework preflight.

Kaggle CLI 1.7 prefers `%USERPROFILE%/.kaggle/access_token` over a legacy
`kaggle.json`, even when `KAGGLE_CONFIG_DIR` points elsewhere. For account
rotation, isolate both `USERPROFILE` and `KAGGLE_CONFIG_DIR` in the submitting
process, verify access to one private input owned by the intended account, and
only then push. Do not edit or delete the user's global OAuth token.
