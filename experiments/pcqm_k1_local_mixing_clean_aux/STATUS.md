# Release status

BLOCKED_BEFORE_SUBMISSION. Kaggle1 was not provisioned.

Existing K1 canonical trace declares checkpoint_identity=false. Raw source
trace and best/last model files are absent from the checked retained local
records. The original Kaggle2 kernel kaseichou/molgap-pcqm-k1-v4-reference-s42
was queried with confirmed kaseichou credentials; status and file listing both
returned403 on 2026-10-01. No assumption of expiration, deletion or failure is
made. Historical scalar/prediction evidence remains valid under its owner.

SSMA and clean fingerprint trainer extensions are prepared. One concentrated
synthetic invocation passed215 tests; see [check metadata](qualification_check/run-metadata.json).
Both local qualification trajectories were published before execution through
the existing experiment CLI and independently closed NO_TRAIN through the
existing RML finalizer. Both pipelines returned COMPLETE/VALID and added no
replay-pool entries. See [SSMA](qualification/ssma/decision.md) and
[fingerprint](qualification/clean_fingerprint/decision.md). This is an evidence
release blocker, not a negative molecular result. The existing strict trace
validator's exact rejection is retained in [reference gate](reference_gate.json).
Runtime qualification,
native allocation cost binding and platform dispatcher integration remain
unverified and must be completed before any release. This branch stays owned
through resolution; it is not merged to desktop or archived as a negative model.
