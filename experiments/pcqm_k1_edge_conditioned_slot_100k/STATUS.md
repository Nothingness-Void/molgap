# Operational state

Kaggle2 kernel `kaseichou/molgap-k1-edge-conditioned-slot-s42` version 2 is
`RUNNING`. It is the user-authorized infrastructure-only retry of version 1,
submitted with the explicit CLI accelerator override
`NvidiaTeslaP100`. The source dataset, archive hash, model, data, seed, strict
FP32/no-TF32 mode, physical BS128, optimizer, schedule, role access, and all
scientific gates are unchanged.

Version 1 terminated before candidate execution because Kaggle allocated a
Tesla T4 to the metadata-only P100 request. It produced no epoch, metric,
checkpoint, or acceptance output and remains classified as infrastructure
evidence rather than a scientific failure. Its terminal log and downloaded
source archive are retained.

Version 2 keeps the one-device P100 runtime guard. Until terminal acceptance,
do not submit another retry, candidate, seed, scale bridge, or official/test
evaluation. Version-1 identity is in `results/submission.json`; version-2
identity is in `results/retry_v2.json`.
