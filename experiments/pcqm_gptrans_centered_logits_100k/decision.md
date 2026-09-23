# Centered-logits 100K cancellation decision

Decision date: 2026-09-24 (Asia/Tokyo).

The desktop-owned, seed-42 single-arm Kaggle3 kernel
`nvoid912/molgap-gptrans-centered-logits-100k-s42-v1` was pushed as version 1.
The user then cancelled it. At the 2026-09-23 16:12:56 UTC reconciliation,
Kaggle returned `KernelWorkerStatus.CANCEL_ACKNOWLEDGED`; its output-file list
contained no entries. The exact training progress, actual hardware allocation,
role use, and device/wall cost are unobserved. The launch and cancellation
observations are retained under `launch/`.

Outcome: `INCONCLUSIVE`. There is no accepted runtime certificate, complete
training trace, candidate MAE, aligned predictions, or replay-ready artifact.
This observation cannot support a positive or negative claim about logit
centering. The accepted GPTrans-T reference remains unchanged.

The single-arm version-1 attempt is closed. A future run requires a separate
prospective contract and resource decision. Desktop submissions should prefer
two independently justified arms in one durable accelerator job when the
allocation and time limit permit; the operating rule is in
`docs/operations/EXPERIMENT_ADDON_GUIDE.md`.
