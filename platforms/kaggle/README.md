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
verify the returned job identity and actual runtime GPU. Active jobs request
`NvidiaTeslaT4` only; P100 is no longer available. Kaggle may expose two T4s
even when only one scientifically justified arm exists. Isolate that arm to
one device and count the entire allocation in native cost.

Both Kaggle accounts hold accepted, byte-identical fixed OGB PCQM4Mv2
100K/500K graph datasets. Evidence is under
`platforms/_records/kaggle/pcqm_fixed_datasets_v1/` and
`platforms/_records/kaggle/pcqm_fixed_datasets_kaggle2_v1/`. Both accounts are
excluded from holding the 1M and full identities.
