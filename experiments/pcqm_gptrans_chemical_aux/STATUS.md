# Chemical auxiliary pair status

Owner: desktop, `codex/exp/gptrans-chemical-aux`.

The authorized two-new-direction 100K pair was submitted to Kaggle1. The
authenticated scheduler observation is RUNNING; see
[submission_v2/remote_observation.json](submission_v2/remote_observation.json)
for its timestamp, actual kernel/version, source/Spec and reconciled receipt.
This is submission acceptance, not GPU qualification or scientific acceptance.

Arms: `descriptor_aux` (Gap L1 + 0.1 masked descriptor MSE) and
`fingerprint_aux` (Gap L1 + 0.1 fingerprint BCE). Each uses one isolated T4,
FP32, seed42, 100K train rows, 50K internal development rows and 60 epochs.
The retained GPTrans-T reference is reused; no reference training is launched.
Read [training_protocol.md](training_protocol.md) for the frozen recipe and gate.

Both complete train-only caches were accepted. Their CPU-stage result and
NO_TRAIN diagnostic closure are owned by
[cache_retry_observation/decision.md](cache_retry_observation/decision.md).
The unsubmitted v1 training plans closed NO_TRAIN after a source packager
filename guard blocked packaging; their decisions stay under
`training_prospective/`. The corrected source and active plans are bound in
`experiment_spec_v2.json` and `training_prospective_v2/`.

Local release bindings and both full-frozen-shard loaders passed. Remote
runtime/resume/optimizer-step overhead gates must pass before formal training.
No terminal training result, scientific promotion, replay-ready claim or
successor is authorized by this status. On return, query the actual Kaggle job
before retrieving the minimum acceptance artifacts for each arm. No automatic
monitor or server handoff was created.
