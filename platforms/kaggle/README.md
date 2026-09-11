# Kaggle Adapter

Cross-experiment Kaggle packages are grouped by workload role:
`acquisition/`, `training/`, and `evaluation/`. Experiment-specific kernels
stay with their owning experiment.

Kaggle provides the server-side queue. Submit a durable kernel directly,
verify its remote state, and do not create a local slot-trigger process.
Retrieved outputs and acceptance evidence belong in
`platforms/_records/kaggle/`.

Kaggle1's fixed OGB PCQM4Mv2 100K/500K graph datasets are accepted under
`platforms/_records/kaggle/pcqm_fixed_datasets_v1/`. The account is explicitly
excluded from holding the 1M and full identities.
