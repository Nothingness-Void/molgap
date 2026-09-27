# Attempt 001 infrastructure disposition

The Kaggle1 v1 paired kernel ended `KernelWorkerStatus.ERROR` during remote
preflight, before either arm entered contracted training. RWSE16 completed its
runtime preflight and ran training-role optimizer probes; the local-edge arm
failed its frozen initial-state identity check. The paired runner stopped both
workers. The [retained observation](../../platforms/_records/kaggle/training/pcqm_gptrans_local_inductive_bias_100k_s42_v1/attempt_001/README.md)
records the status, log, preflight certificate, and exact identity mismatch.

Both v1 trajectories close `INFRASTRUCTURE_ONLY`. Neither has a training trace,
selected checkpoint, development prediction, MAE, or replay-ready entry. The
total native device time and B's exact data-role access remain unmeasured; no
official validation or test role use is evidenced. This disposition makes no
scientific claim about either mechanism. A new source and prospective pair were
separately authorized for attempt 002, whose
[terminal decision](decision_attempt_002.md) owns the scientific result.
