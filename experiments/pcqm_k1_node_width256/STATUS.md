# Status

The single K1 atom-width256 candidate was submitted to Kaggle3 (`nvoid912`)
as [kernel version 1](https://www.kaggle.com/code/nvoid912/molgap-k1-node256-100k-s42-v1),
kernel ID `136660752`. The authenticated scheduler observation at
`2026-10-01T12:46:32.729914+00:00` reported `running`, without a failure message.
This platform state does not establish that GPU qualification or training has
started; no runtime log or result was returned at that observation.

## Frozen launch

- One candidate, original pure-2D V4 100K training / 50K internal development,
  40 epochs, seed42, FP32, batch128, 9 layers; atom256, edge64, slot64.
- 6,035,201 parameters: 1.649495 times the 3,658,817-parameter width192 reference.
- Source `1d869a0474bb37cb4fb120da0d509c48a542b03b`.
- Spec `c5e0b09a3144566079baebb633fd8a3b3e69ad0e2c60449f2d46cb2aefa1d232`.
- Package `475390043d4c5600f55f1aaa888218a65ce83c74d85ae1cc1aae5b72a289bc95`.
- Archive `905c12f632605608886524ef38ecd96d2b60f176ace05d2a627827b61dbc0b1e`.

[Training protocol](training_protocol_kaggle3_v1.md) owns authorization and
comparison gates; [prospective record](kaggle3_v1/width256/trajectory.json)
was published before submission. The earlier [protocol](protocol.md) and
[preparation](preparation_report.json) retain the preceding blocked stage.

## Qualification and retained evidence

The completed, separately authorized K1/SSMA run supplied its width192 reference
arm. Its 40-epoch checkpoint-bound trace, runtime qualification, predictions,
roles and measured T4 cost were inspected. The
[prelaunch comparison](reference_binding/comparison_prelaunch.json) permits
the planned width-only comparison; this reuse does not decide the SSMA question.
Reference MAE is 0.1412944608205557 eV, with training-seed variability unknown.

[CPU qualification](qualification_result.json) verifies initialization and
parameter counts. [Real-shard loader qualification](cpu_loader_qualification.json)
verifies the original 100K/50K input using the family dataset and sampler.
The Windows CPU inspection uses zero loader workers; the frozen Linux run keeps
two workers and must pass its actual remote preflight.

The [release report](kaggle3_submission_v1/release_report.json) passed 69 checks,
including trusted real-shard pickle dependencies. All nine uploaded private
source files, including initialization, were downloaded and matched byte hashes
in [source acceptance](kaggle3_submission_v1/source_dataset_acceptance.json).
The [submission response](kaggle3_submission_v1/submission-response.json),
local launch receipt, and
[scheduler observation](kaggle3_submission_v1/scheduler_observation.json)
retain the actual platform identity. GPU qualification, predictions,
comparison, measured candidate cost and scientific acceptance remain pending.

Keep the experiment ACTIVE on `codex/exp/k1-node-width256`. Reconcile this exact
version and retrieve its durable outputs before deciding another action.
No baseline retraining, extra seed or scale-up is authorized by this submission.

The inherited portable RML check has unrelated historical artifacts declared
locally retained that are absent in this worktree. Their declarations were
preserved; no portable pass is claimed.
