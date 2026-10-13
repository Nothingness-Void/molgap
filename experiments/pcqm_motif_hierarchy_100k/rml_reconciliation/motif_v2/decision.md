# Retained-attempt RML reconciliation — 2026-10-01

This record closed the bookkeeping for `TC-k1-motif-hierarchy-100k-s42-v2` as **INFRASTRUCTURE_ONLY**.

Physical version 2 ERROR at the single-device runtime gate, with zero scientific epochs. The retained wall duration is approximate; accelerator identity and full allocation cost remain unknown.

## Scope and authority

The review used only retained local metadata. It did not run training, model
inference, protected-role evaluation, remote cancellation, retry or submission.
The original prospective trajectory and its knowledge snapshot were preserved
byte-for-byte. Logical run `kaseichou/molgap-k1-motif-hierarchy-s42:v2` maps to physical `kaseichou/molgap-k1-motif-hierarchy-s42:v2` (kernel 136356323).
No native training trace was manufactured; this terminal is not a training
replay entry. Later successful/negative attempts retain separate identities.

Source records:

- `experiments/pcqm_motif_hierarchy_100k/attempt_v2/failure.json`
- `experiments/pcqm_motif_hierarchy_100k/attempt_v2/submission_receipt.json`

Exact source hashes, dimensional V5 outcome, logical/physical binding and
cost translation are in [acceptance.json](acceptance.json). That acceptance
is a metadata bookkeeping review, not another scientific re-evaluation.
Measured, estimated, missing and not-applicable cost fields stay distinct.
The frozen estimated cost event is replaced in RML discovery by its observed
or explicitly missing terminal record; original files remain intact.

No further compute action was released. See `CURRENT_STATE.md` for live state.
