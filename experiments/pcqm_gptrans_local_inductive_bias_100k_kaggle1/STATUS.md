# GPTrans local inductive bias 100K status

**Kaggle1 attempt 002 completed** on 2026-09-27. The authoritative remote
terminal state is `KernelWorkerStatus.COMPLETE` for the Kaggle kernel
`nothingnessvoid/molgap-gptrans-rwse-local-bias-paired-100k-s42-v2`. This is a
paired job with independent `rwse16` and `rwse16_local_edge` arms. The
[retry decision](retry_decision_attempt_002.md),
[release gate](release_gate_attempt_002.json), and
[observed launch record](launch_attempt_002/kaggle_observation.json) pin the
source, data, Spec, and kernel identities. Both arms independently passed
artifact acceptance for the frozen 60-epoch exposure, runtime, selected and
resumable checkpoints, aligned 50K predictions, trace, and protected-role
sealing. The paired gain was 0.0010816 eV, below the frozen 0.003 eV minimum;
the paired row-bootstrap upper bound for B−A was −0.0000293 eV. The scientific
outcome is therefore `NEGATIVE_UNDER_CONTRACT`; no scale-up or successor is
authorized. See [attempt 002 terminal decision](decision_attempt_002.md) and
the retained platform artifacts. RML dual replay-ready closure is pending the
strict same-job comparison-readiness transaction; do not describe this pair as
dual replay-ready until the pool reports two complete entries with no
exclusions.

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
B-versus-A scientific decision is accepted for attempt 001. The failed v1
attempt is not negative mechanism evidence. Attempt 002 has the accepted
paired result above; its RML replay qualification remains pending.
Attempt 002 remains on this experiment branch through terminal acceptance.
The [initial release decision](launch_decision.md) authorized attempt 001;
the user separately authorized the bounded attempt 002.
