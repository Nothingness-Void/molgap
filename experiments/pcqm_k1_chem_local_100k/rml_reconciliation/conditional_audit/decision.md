# Retained-attempt RML reconciliation — 2026-10-01

This record closed the bookkeeping for `TC-k1-chem-local-post100k-audit-s42` as **CLOSED**.

Both chemistry-local candidates failed the nomination gate. The conditional fixed500K audit was not released or launched. Its frozen run ID was a prospective logical ID, not an observed physical job.

## Scope and authority

The review used only retained local metadata. It did not run training, model
inference, protected-role evaluation, remote cancellation, retry or submission.
The original prospective trajectory and its knowledge snapshot were preserved
byte-for-byte. Logical run `kaseichou/molgap-k1-chem-local-audit-s42:v1` was never a launched physical job.
No native training trace was manufactured; this terminal is not a training
replay entry. Later successful/negative attempts retain separate identities.

Source records:

- `experiments/pcqm_k1_chem_local_100k/decision.md`
- `experiments/pcqm_k1_chem_local_100k/STATUS.md`

Exact source hashes, dimensional V5 outcome, logical/physical binding and
cost translation are in [acceptance.json](acceptance.json). That acceptance
is a metadata bookkeeping review, not another scientific re-evaluation.
Measured, estimated, missing and not-applicable cost fields stay distinct.
The frozen estimated cost event is replaced in RML discovery by its observed
or explicitly missing terminal record; original files remain intact.

No further compute action was released. See `CURRENT_STATE.md` for live state.
