# Initialized-layout native T4 pair handoff

## Submission observation

Desktop-owned branch `codex/exp/k1-fused-layout-t4-100k`, checkout
`D:/w/k1-fused-t4`. One authorized Kaggle1 submission, no automatic retry,
continuation, local monitor or server takeover.

- Kernel: [nothingnessvoid/molgap-k1-initialized-layout-100k-s42-v2](https://www.kaggle.com/code/nothingnessvoid/molgap-k1-initialized-layout-100k-s42-v2).
- Kernel ID `138206744`, platform version `1`; script-version ID unknown.
- [API response](submission/kernel_push.json) records the successful submission.
  [Scheduler observation](submission/status_after_push.json), at
  `2026-10-11T17:50:58.411326+00:00` (2026-10-12 JST), records `QUEUED`.
- [Bound receipt](submission/observed_launch_receipt.json) binds both arms to
  the observed job. Its generic local-core `SUBMIT_UNIMPLEMENTED` limitation
  does not replace the actual adapter response or scheduler evidence.
- [Startup observation](submission/startup_observation.json) records zero
  output files at that observation, not a failure. Native qualification and
  formal training progress are not yet verified; actual cost remains unknown.

## Frozen release

[Protocol](protocol.md) and the [parent contract](../protocol.md) own settings.
One T4x2 Notebook runs original K1 and initialized-layout K1 on separate GPUs:
fixed100K TRAIN, fixed50K reused internal development, seed42,40epochs per arm,
FP32/noTF32, physical BS128, unchanged native AdamW and training semantics.
This candidate is not fused AdamW. Native all-arm equality/resume qualification
must pass before either formal arm starts; a failure stops both arms.
No protected evaluation or promotion is released.

- Executable source: `cd615aa4716a43f984de8223ab50b137d4b45eab`.
- Spec: `2d77fa933ae9345f1591b71955f7c473faf8d67016ac94fdd917845757dfb1f7`.
- Package: `7fb07ae2a26d364fcafea73309cbfe3ba20f1862d2392a318aeb1b30d9bd42ad`.
- Source archive SHA256: `923743640e74447bea3ace18c3520f225c0b3ae112dbf5cb3ff7f32683394dd5`.
- Private source dataset: `nothingnessvoid/k1-initialized-layout-100k-s42-source-v2`.
- Fixed input: `nothingnessvoid/pcqm4mv2-ogb-fixed-100k-v1`.

[Release report](submission/release_report.json) and
[package manifest](submission/package_manifest.json) retain the release binding.
[Source roundtrip](submission/source_dataset_roundtrip.json) verifies all15
mounted source files by SHA256; [local TRAIN-only CPU check](submission/local_train_input_check.json)
verifies actual input hashes, membership, batch shapes and target statistics.
Later documentation commits do not change this executable source.

## Verification and return action

Local focused runs passed151 tests for runtime/recipe/lifecycle integration,
then111 tests including the corrected native entry whitelist; these overlap
and are not a combined test count. They do not qualify native T4 execution.
RML rebuild and frozen check passed for prospective records. Whole-owner
portable checking remains blocked by pre-existing missing historical artifacts
in this isolated checkout; neither this pending run nor historical evidence is
claimed terminal replay-ready.

On desktop return, use the Kaggle skill to reconcile this exact kernel/version
and independently retrieve source/config identity, best/last checkpoints,
native qualification, predictions, trace, roles and actual cost. Reuse existing
family acceptance and V5/RML helpers. Compare matched complete exposure and
the frozen paired quality/speed gates; row bootstrap is not seed variance.
Do not retry based on a stale local snapshot. Desktop shutdown leaves the cloud
queue independent and does not hand work to server.
