# Repaired T4 pair handoff

2026-10-11 JST. Same owning question/branch `codex/exp/k1-fused-layout-t4-100k`,
custody `D:/w/k1-fused-t4`. No server takeover, heartbeat or local submitter.

## Exact attempt

- Kaggle1 `nothingnessvoid/molgap-k1-fused-layout-100k-s42-v2`, kernelID138119131,
  version1, script-version ID unknown. [API receipt](submission/kernel_push.json)
  owns actual identity; v2 is the slug, not a claimed platform version2.
- [Scheduler observation](submission/status_after_push.json): RUNNING.
  Native GPU qualification, formal training state and scientific endpoint were
  not yet accepted; metadata confirms private/GPU-enabled/NvidiaTeslaT4 only.
- Frozen executable source `6b526e4333beca85008ddec085969430e073c4fc`;
  source archive SHA256 in [release report](submission/release_report.json).
- Private source `nothingnessvoid/k1-fused-layout-100k-s42-source-v2`;
  [mounted layout](submission/source_dataset_layout.json) and
  [15-file exact byte roundtrip](submission/source_dataset_roundtrip.json).
- [Observed launch receipt](submission/observed_launch_receipt.json) binds
  both arms, actual kernel/version and frozen package through the shared API.
  Submission acceptance is not runtime/scientific acceptance.

## Checks and unchanged contract

[Protocol](protocol.md) and [user release](user_release.json) own scope.
[Input equivalence](submission/input_equivalence.json) verifies identical
family, initial tensors, data, training declaration and recipe bytes versus
attempt001. Original100K/50K cache acceptance is explicitly reused, not claimed
as a newly executed role check. Fused AdamW plus CPU layout is still the only
scientific execution delta; original single-forward reference, FP32/BS128,
seed42,40epochs each and common internal development selection remain fixed.

Both uploaded initial wrappers passed the actual frozen K1 reader in the
package-only CPU release worker:367 tensors with the unchanged initial hash.
The repaired check also rejects both arms of the old frozen payload before
publication. No weight regeneration, model construction or CUDA initialization
occurred in those local checks. Tests:75 owner scoped,230 integration with1
host-limited skip, and75 scoped on the selectively repaired desktop branch.
Prospective records published; RML rebuild/frozen check passed.

[Local preparation blocker](submission/local_preparation_blocker.json) retains
the missing review-pointer failure before prospective publication; the pointer
was supplied and a fresh staging directory used. No remote retry was triggered
by that local issue. Attempt001 failure evidence and original source remain
unchanged; its terminal RML envelope gap is not repaired by this new submission.

## Next reconciliation

When desktop returns, use the platform skill for this exact kernel/version and
durable outputs. Verify all-arm native certificates before interpreting formal
trace, then inspect matched40epochs, saved predictions and train/dev/entry
timings through existing family acceptance/V5/RML helpers. The frozen10% speed
and paired upper+0.001eV quality gates remain unchanged; row uncertainty is not
training-seed variance. No model/default adoption, protected role access,
automatic continuation or further retry is released.
