# Operational state

Kaggle2 kernel `kaseichou/molgap-k1-edge-conditioned-slot-s42` version 2
terminated `ERROR`. It was the user-authorized infrastructure-only retry of
version 1, submitted with the explicit CLI accelerator override
`NvidiaTeslaP100`. The source dataset, archive hash, model, data, seed, strict
FP32/no-TF32 mode, physical BS128, optimizer, schedule, role access, and all
scientific gates were unchanged.

Version 1 terminated before candidate execution because Kaggle allocated a
Tesla T4 to the metadata-only P100 request. It produced no epoch, metric,
checkpoint, or acceptance output and remains classified as infrastructure
evidence rather than a scientific failure. Its terminal log and downloaded
source archive are retained.

Version 2 kept the one-device P100 runtime guard but Kaggle exposed a Tesla T4,
so it terminated before candidate execution with
RuntimeError: Candidate requires one P100, got Tesla T4. It produced no
candidate output, epoch, metric, checkpoint, or acceptance result. The
terminal diagnosis is in `results/failure_diagnosis_v2.md`; the durable
terminal handoff and downloaded source/log remain under
`platforms/_records/kaggle/training/k1_edge_conditioned_slot_s42_v2/`.

Do not submit another retry, candidate, seed, scale bridge, or official/test
evaluation from this terminal error. Version-1 identity is in
`results/submission.json`; version-2 identity is in `results/retry_v2.json`.
