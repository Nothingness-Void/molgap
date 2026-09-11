# Kaggle Adapter

Cross-experiment Kaggle packages are grouped by workload role:
`acquisition/`, `training/`, and `evaluation/`. Experiment-specific kernels
stay with their owning experiment.

Kaggle provides the server-side queue. Submit a durable kernel directly,
verify its remote state, and do not create a local slot-trigger process.
Retrieved outputs and acceptance evidence belong in
`platforms/_records/kaggle/`.

For accelerator-specific kernels, use the current project Kaggle CLI with
credentials supplied through `KAGGLE_USERNAME` and `KAGGLE_KEY`, and pass an
explicit `--accelerator` value. After submission, pull the remote metadata and
verify that `machine_shape` survived. Kaggle API 1.7.4.5 can silently omit that
field and fall back to P100 even when the local JSON requests T4; that client
is suitable only for legacy operations that do not depend on accelerator type.

Kaggle1's fixed OGB PCQM4Mv2 100K/500K graph datasets are accepted under
`platforms/_records/kaggle/pcqm_fixed_datasets_v1/`. The account is explicitly
excluded from holding the 1M and full identities.
