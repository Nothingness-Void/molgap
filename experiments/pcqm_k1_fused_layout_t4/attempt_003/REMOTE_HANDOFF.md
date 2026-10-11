# Timing-IO retry handoff

2026-10-11 JST. Desktop-owned same question, same experiment branch
`codex/exp/k1-fused-layout-t4-100k`, checkout `D:/w/k1-fused-t4`.
No server takeover, heartbeat, local slot trigger or automatic retry.

## Exact launch

- Kaggle1 `nothingnessvoid/molgap-k1-fused-layout-100k-s42-v3`,
  kernelID138131630, platform version1. Script-version ID is unknown.
  [API push](submission/kernel_push.json) and
  [observed receipt](submission/observed_launch_receipt.json) own identity.
- [Scheduler](submission/status_after_push.json): RUNNING at the recorded UTC
  instant. [Single startup check](submission/startup_observation.json) has an
  empty log; native qualification and formal training are not yet verified.
- Frozen source `e4876d35fca740e35bdfb908836c7a2cda7aab0e`, archive
  `d9f09a783dd7b72e04f105fd7c7ec3c77b08a7f699b9acf3f2dd697132b37ced`;
  package `97fc370188d37c9a15027f64e18bc4fef17dacdbd5db61a081f5d4d2a1780f70`.
- Private source `nothingnessvoid/k1-fused-layout-100k-s42-source-v3`;
  [create](submission/source_dataset_create.json),
  [mount inventory](submission/source_dataset_layout.json) and
  [15-file SHA256 roundtrip](submission/source_dataset_roundtrip.json) retained.
  The SDK aggregate download returned404; authenticated per-file dataset REST
  retrieval verified the same files. No republishing occurred.

## Unchanged contract and local checks

[Protocol](protocol.md) and [user release](user_release.json) authorize one
fresh100K/40epoch paired run on T4x2. Reference uses native-default AdamW;
candidate enables fused AdamW and CPU layout only. FP32/no-TF32, BS128, original
loaders, objective and development selection stay fixed. No EMA or protected
roles. Both start from the original pinned seed42 artifact, not partial states.
[Input equivalence](submission/input_equivalence.json) verifies both declarations
and exact recipe bytes versus attempt002; accepted cache checks are reused,
not re-executed. Packaged actual initial loading (367 tensors) and object/array
IO roundtrips passed for both arms. The repair/lifecycle tests passed27 tests.

Per-arm prospective RML records were published before submission and committed.
RML rebuild/frozen check passed. Whole-checkout `--portable` did not pass:
unrelated historical locally-retained large artifacts are not materialized in
this isolated worktree. This run is not terminal or replay-ready; no historical
retention claims were rewritten to suppress that limitation.

## Next reconciliation

Use the platform skill for this exact returned kernel/version. Verify all-arm
native certificates before interpreting formal progress. Retain manifest-bound
best/last state, selected predictions, complete traces and per-epoch timing.
Report matched40epoch training-pipeline, development and entry/allocation costs
separately; the local24.13% diagnostic is not a T4 forecast. The frozen gates
remain >=10% training-pipeline reduction and selected candidate-minus-reference
MAE paired95% upper bound <=+0.001eV; row bootstrap is not seed variance.
No adoption,500K/full release, automatic continuation or further retry.
