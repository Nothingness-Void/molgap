# Retained-attempt RML reconciliation — 2026-10-01

This record closed the bookkeeping for `TC-gptrans-author-input-preparation` as **INFRASTRUCTURE_ONLY**.

CPU path construction completed, but independent chunk source rederivation failed with a chunk-local/shard-coordinate IndexError. No successful preparation acceptance; not a scientific negative.

## Scope and authority

The review used only retained local metadata. It did not run training, model
inference, protected-role evaluation, remote cancellation, retry or submission.
The original prospective trajectory and its knowledge snapshot were preserved
byte-for-byte. Logical run `kaseichou/molgap-gptrans-author-inputs-preparation-v1` maps to physical `kaseichou/molgap-gptrans-author-inputs-preparation-v1:v1` (kernel 136492377).
No native training trace was manufactured; this terminal is not a training
replay entry. Later successful/negative attempts retain separate identities.

Source records:

- `experiments/pcqm_gptrans_author_alignment/results/preparation_failure_v1.md`
- `experiments/pcqm_gptrans_author_alignment/preparation_submission.json`
- `platforms/_records/kaggle/training/gptrans_author_inputs_preparation_v1/gptrans_author_inputs/failure.json`
- `platforms/_records/kaggle/training/gptrans_author_inputs_preparation_v1/gptrans_author_inputs/native_cost.json`

Exact source hashes, dimensional V5 outcome, logical/physical binding and
cost translation are in [acceptance.json](acceptance.json). That acceptance
is a metadata bookkeeping review, not another scientific re-evaluation.
Measured, estimated, missing and not-applicable cost fields stay distinct.
The frozen estimated cost event is replaced in RML discovery by its observed
or explicitly missing terminal record; original files remain intact.

No further compute action was released. See `CURRENT_STATE.md` for live state.
