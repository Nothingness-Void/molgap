# GPTrans local inductive bias 100K status

**Kaggle1 attempt 002 was submitted and was `KernelWorkerStatus.RUNNING` at
2026-09-27 07:30:37 UTC.** The authoritative live state is the Kaggle kernel
`nothingnessvoid/molgap-gptrans-rwse-local-bias-paired-100k-s42-v2`. This is a
paired job with independent `rwse16` and `rwse16_local_edge` arms. The
[retry decision](retry_decision_attempt_002.md),
[release gate](release_gate_attempt_002.json), and
[observed launch record](launch_attempt_002/kaggle_observation.json) pin the
source, data, Spec, and kernel identities. The local
[reconciled receipt](launch_attempt_002/reconciled_receipts/17faa1313b7d3d4637c914ccb22abfba0820948e36a219a76e8502922949e081.json)
binds the platform reference to the package; it does not qualify runtime or
accept scientific evidence. Wait for the remote terminal state before fetching
the minimal replay artifacts and independently accepting each arm.

**Kaggle1 attempt 001 ended `KernelWorkerStatus.ERROR` before training.** The
remote `rwse16` worker accepted its T4 runtime preflight. The
`rwse16_local_edge` worker failed the frozen initial-model-state identity check:
observed `96eefae91355e36eff3566e063bee88bceae903ed8cae56e8269cc984b156cc9`,
expected `9e2e58fa54f68c061f47637a9c406dd4c849ae9b0d49623881f63bb94655db0b`.
The paired runner stopped before launching either training worker. See the
[attempt record](../../platforms/_records/kaggle/training/pcqm_gptrans_local_inductive_bias_100k_s42_v1/attempt_001/README.md)
and retained remote log.

Attempt 001 has no completion manifest, selected model, aligned development
predictions, resumable training checkpoint, or 60-epoch trace. No MAE or
B-versus-A scientific decision is accepted for either attempt, and neither arm
is replay-ready. The failed v1 attempt is not negative mechanism evidence.
Attempt 002 remains on this experiment branch through terminal acceptance.
The [initial release decision](launch_decision.md) authorized attempt 001;
the user separately authorized the bounded attempt 002.
