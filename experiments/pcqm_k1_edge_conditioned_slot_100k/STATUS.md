# Operational state

Kaggle2 kernel `kaseichou/molgap-k1-edge-conditioned-slot-s42` version 3 is
`COMPLETE` and passed saved-artifact acceptance without model inference. It is
the user-authorized infrastructure-only repair after versions 1 and 2 both
received a Tesla T4 despite requesting a P100. Version 3 requested T4
explicitly, exposed only the first assigned CUDA device, accepted the actual
assigned GPU model, and recorded its runtime identity through the existing
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
physical BS128, optimizer, schedule, role access, and all scientific gates were
unchanged. It completed 40 epochs and 31,240 optimizer steps with
3,671,105 parameters. The best epoch was 39, with development Gap MAE
`0.1410829425 eV`, 790.15 graphs/s, and 543.898 / 586 MiB peak allocated /
reserved memory on a 14,911.6875 MiB Tesla T4.

Against immutable K1-v4 (`0.1413736343 eV`), the candidate gain was only
`0.0002906919 eV`. The paired bootstrap 95% interval was
`[-0.0012159737, 0.0006187779] eV`; the material-gain and favorable-interval
gates therefore failed, so `selected_candidate=null`. The candidate was
mechanically valid and trainable, but scientifically sub-threshold and
inconclusive. The question is closed; no retry, candidate, seed, scale bridge,
or official/test evaluation is released. The full attribution is in
`decision.md` and the machine record is in `results/acceptance.json`.

Submission identities are in `results/submission.json`, `results/retry_v2.json`,
and `results/retry_v3.json`. Complete artifacts remain under
`platforms/_records/kaggle/training/k1_edge_conditioned_slot_s42_v3/`.
