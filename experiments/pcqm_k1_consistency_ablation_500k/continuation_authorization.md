# Authorized continuation — 2026-10-08

The desktop user explicitly requested resubmission after accepting version1's
bounded stage. Continue the same two-arm question from verified epoch index23
(89,838 steps/11,499,264 presentations per arm); never reset or retrain a peer.
The original Spec, recipes, initialization, roles, optimizer, sampling, precision
and60epoch schedule remain frozen. Source changes only resolve resume retention
and bind the continuation to fresh private source/checkpoint datasets.

Reuse `pcqm_500k_preparation.prepare_continuation`, the original 500K trainer,
shared release/source/prospective helpers and the explicit Kaggle accelerator
submitter. `training_reproducibility.retained_resume_artifacts` supplies one
explicit `selected-and-resume-v1` policy to packaging, mounted verification,
GPU preflight and completed-peer retention. Preserve the producer stage manifest
byte-for-byte. Required selected/resume states and metadata remain hash-checked;
only exact `predictions_epoch_<number>.pt` outputs are omitted. Historical callers
without the policy keep their full-artifact behavior. No model code or loss change.

The next invocation is bounded to32,400seconds on Kaggle1/T4x2, with both native
optimizer preflights before training. It may require another bounded continuation
to finish37remaining epochs; no final-completion claim is made in advance.
Measured training allocation17.389569095 T4-hours leaves34.610430905 under52hours;
one next9hour paired allocation fits that remaining ceiling. Keep setup/native
qualification, training/idle allocation and unknown scheduler costs separate.

Private checkpoint dataset: `nothingnessvoid/molgap-k1-consistency-500k-resume-s42-stage23-v1`.
Fresh executable source dataset: `nothingnessvoid/molgap-k1-consistency-500k-source-s42-v2`.
Continue the existing kernel, binding its actual returned version/physical ID.
Do not create monitoring, transfer ownership to server or release protected roles.

Acceptance authority: [version1 stage record](submission_v1/terminal_inspection/stage_acceptance.md).
Submission identity and subsequent scheduler observations belong in `submission_v2/`.
