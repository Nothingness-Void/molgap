# Operational state

Kaggle2 kernel `kaseichou/molgap-k1-edge-conditioned-slot-s42` version 3 is
`RUNNING`. It is the user-authorized infrastructure-only repair after versions
1 and 2 both received a Tesla T4 despite requesting a P100. Version 3 requests
T4 explicitly, exposes only the first assigned CUDA device, accepts the actual
assigned GPU model, and records its runtime identity through the existing
optimizer-inclusive certificate.

Version 1 terminated before candidate execution because Kaggle allocated a
Tesla T4 to the metadata-only P100 request. It produced no epoch, metric,
checkpoint, or acceptance output and remains classified as infrastructure
evidence rather than a scientific failure. Its terminal log and downloaded
source archive are retained.

Version 2 kept the one-device P100 runtime guard but Kaggle exposed a Tesla T4,
so it terminated before candidate execution. It produced no candidate output,
epoch, metric, checkpoint, or acceptance result. Its diagnosis is in
`results/failure_diagnosis_v2.md`.

Version 3 changes only resource binding and dependency compatibility. The
source dataset, archive hash, model, data, seed, strict FP32/no-TF32 mode,
physical BS128, optimizer, schedule, role access, and all scientific gates are
unchanged. Until terminal acceptance, do not submit another retry, candidate,
seed, scale bridge, or official/test evaluation. Submission identities are in
`results/submission.json`, `results/retry_v2.json`, and `results/retry_v3.json`.
