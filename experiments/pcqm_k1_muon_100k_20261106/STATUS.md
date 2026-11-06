# Status

## Terminal acceptance:2026-11-06 JST

Both arms completed40passes/31240updates/3998720presentations. Existing strict
acceptance returned COMPLETE/closure_complete=true: both native and replay
VERIFIED with no exclusions; Muon comparison STRICT_CAUSAL. Both RML trajectories
were finalized through the shared closure, which owns its derived-index update.

AdamW CLOSED:0.14129446082055568eV. Muon POSITIVE_UNDER_CONTRACT:
0.1325206676620245eV; effect-8.773793159meV,95%row interval
[-9.708100418,-7.861417456]meV. See decision.md and attribution.md.
Actual allocation6.390410T4-device-hours; Muon process wall2.03x reference.
Source/config/receipt identities below are unchanged. No new training submitted.
Keep this owner branch pending an explicit adoption/scale decision.

Strict qualification directory:
`runs/pcqm-k1-muon-100k-20261106-s42/prospective/strict_qualification/f4005a7e77752152c375/`.
Role audit and reviewed closure input:closure_v1/. Only selected required
artifacts were retrieved to D:; no all-output download was used.

## Submission history

2026-11-06 JST: Kaggle2 two-fresh-arm100K submission authorized by the user.
Submitted to Kaggle2 at2026-11-06 03:01:51 JST, job142927073/version1:
`kaseichou/molgap-k1-muon-100k-20261106-s42`.
The existing submitter returned SUBMISSION_BOUND and scheduler RUNNING. This
does not establish native qualification, training entry or replay readiness.
Luna's one-shot check at03:02:27-03:02:31 JST observed latest version1 RUNNING,
matching receipt version1, with zero artifact downloads. No persistent monitor
or new cron was installed.
Shared Muon support0d6beda4 is applied as b36e541b following user approval of the
adaptation and measured budget; existing CPU update/recovery/output/replay checks
passed. Native qualification executes through the existing worker.

Frozen source62c70310d0528b0e8e1181c943ca10e9e6282bb4;
Spec975d4f6f1da6b0ee6d2d6545d17550b6e3cfb3b55b2efad3d5bc366419420a1f;
packagefddb6e19c9070b0e23d299727ab23c2be4a97d6ae0e1b379134de8886fb90118.
Runpcqm-k1-muon-100k-20261106-s42, attemptv1.
Durable launch receipt and submission journal are under
`D:/molgap-artifacts/runs/pcqm-k1-muon-100k-20261106-s42/v1/operations/`.
Prepared payload and frozen inputs remain beside operations. Prospective records
for both arms are published under this experiment's runs directory.

Owner branch: codex/exp/k1-muon-100k-20261106.
Owner checkout: D:/molgap-worktrees/k1-muon-100k-20261106.
Arms: adamw, muon; both new training in one Kaggle2 T4x2 session,40passes each.
The user requested computer shutdown after successful submission and retained
receipts; a one-shot Luna scheduler check precedes shutdown.
