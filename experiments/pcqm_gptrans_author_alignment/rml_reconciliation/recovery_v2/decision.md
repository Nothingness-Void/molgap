# Retained-attempt RML reconciliation — 2026-10-01

This record closed the bookkeeping for `TC-gptrans-author-input-preparation-verification-recovery-v2` as **NO_TRAIN**.

Independent CPU acceptance established 150,000 rows / 150 chunks, full source rederivation and the exact two-table degree transform. This was accepted input preparation, not model training or an accuracy comparison.

## Scope and authority

The review used only retained local metadata. It did not run training, model
inference, protected-role evaluation, remote cancellation, retry or submission.
The original prospective trajectory and its knowledge snapshot were preserved
byte-for-byte. Logical run `kaseichou/molgap-gptrans-author-inputs-verification-recovery` maps to physical `kaseichou/molgap-gptrans-author-inputs-verification-recovery:v2` (kernel 136535008).
No native training trace was manufactured; this terminal is not a training
replay entry. Later successful/negative attempts retain separate identities.

Source records:

- `experiments/pcqm_gptrans_author_alignment/verification_recovery/acceptance_v2.json`
- `experiments/pcqm_gptrans_author_alignment/verification_recovery/submission_v2.json`
- `platforms/_records/kaggle/training/gptrans_author_inputs_verification_recovery_v2/gptrans_author_inputs/native_cost.json`
- `platforms/_records/kaggle/training/gptrans_author_inputs_verification_recovery_v2/gptrans_author_inputs/startup.json`

Exact source hashes, dimensional V5 outcome, logical/physical binding and
cost translation are in [acceptance.json](acceptance.json). That acceptance
is a metadata bookkeeping review, not another scientific re-evaluation.
Measured, estimated, missing and not-applicable cost fields stay distinct.
The frozen estimated cost event is replaced in RML discovery by its observed
or explicitly missing terminal record; original files remain intact.

No further compute action was released. See `CURRENT_STATE.md` for live state.
