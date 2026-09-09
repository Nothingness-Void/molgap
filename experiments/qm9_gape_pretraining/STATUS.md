# Status

Kaggle2 CPU kernel `kaseichou/molgap-qm9-gape-lite-cache-prep`, version 1,
was submitted from source commit `3c0b48cca759f01d5a2df4ea445bf73cee7988c7`.
Remote metadata confirmed `enable_gpu=false` and `machine_shape=None`. The job
completed and its downloaded cache passed independent data-only acceptance:
30,000 train graphs, 3,000 validation graphs, 17 hash-matched shards, aggregate
SHA-256 `80a2d256ccbbde8de5862e7ad4fd61ea7c1255a19217e590033a94cabe1c6340`,
and no held-out/test role. The cache was uploaded unchanged as private dataset
`kaseichou/molgap-qm9-gape-lite-cache`.

The only GPU screen is Kaggle2 kernel
`kaseichou/molgap-qm9-gape-lite-s42`, version 1. Remote metadata confirms
`machine_shape=NvidiaTeslaT4`, both accepted datasets, and a 14,400-second cap;
the job was `RUNNING` at first verification. Its three arms share one task and
platform and each independently uses physical batch 128 on one visible T4.

Version 1 ended before training because PyTorch 2.6's new weights-only default
rejected the trusted PyG cache class. No metric or epoch exists. The bounded
compatibility diagnosis is in `results/gpu_v1_failure_diagnosis.md`; the frozen
scientific contract is unchanged and permits one repaired version.

Version 2 was submitted with only the trusted-cache deserialization fix at
source commit `56e5db105ad8c43809664c0e0f001ea9ac9104e1`. Remote metadata again
confirms `NvidiaTeslaT4`, the same two datasets, and the same timeout; it was
`RUNNING` at first verification. All scientific settings remain identical to
version 1.

Version 2 completed and passed no-inference acceptance. The same-task
comparability guard passed, but matched GAPE-lite did not beat either required
control by the frozen margin. The exact question is closed in `decision.md`;
no PCQM transfer or successor was submitted.
