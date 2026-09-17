# Xi'an cross-process replay probe

On 2026-09-15, job 67440607 completed with exit code 0 in 266 seconds on
c06r2n07. Downloaded model-replay JSON files matched remote SHA256SUMS.

Both processes reproduced initialization, predictions, and gradients exactly
within their three repetitions. Across processes, initialization, predictions,
and loss (5.2028045654296875) matched, but 269 parameter gradient tensors had
different hashes. Acceptance was false. The loss was an untrained diagnostic
value, not a model-quality metric.

This reproduced the earlier cross-process failure despite sorted CSR reductions
and ROCBLAS_DEFAULT_ATOMICS_MODE=0. MIOpen reported that disabling benchmark
mode was unsupported and forced benchmark mode on. That warning is a lead,
not proof that MIOpen caused the differing gradients. Hash differences alone
do not quantify numerical error or downstream MAE drift.

The probe did not authorize formal V4 training or short-training successors.
A useful follow-up would save aligned gradient tensors and localize the first
divergent backward operation across isolated processes before spending further
training hours. CPU workloads were not assessed by this probe.

Evidence: `model-replay/acceptance.json`, `first.json`, `second.json`, and
`SHA256SUMS`; execution logs are under `logs/`.

