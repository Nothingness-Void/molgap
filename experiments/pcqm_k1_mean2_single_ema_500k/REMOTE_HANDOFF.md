# Kaggle1 paired500K handoff - 2026-10-13 JST

Owner branch: `codex/exp/k1-mean2-single-ema-500k-20261013`.
Owner worktree: `D:/w/k1-mean2-single-ema-500k-20261013`.
[Protocol](protocol.md) and [config](config.json) own parameters and boundaries.

## Submission observation

- Actual ref: `nothingnessvoid/molgap-k1-mean2-single-ema-500k-20261013`.
- KernelID138365645, version1; script/session version ID unknown.
- API scheduler at2026-10-12T17:45:43.159225+00:00: **QUEUED**.
  No native allocation, preflight, training exposure, speed or MAE observed yet.
- Source dataset: `nothingnessvoid/molgap-k1-mean2-single-ema-500k-20261013-src`.
  API status ready; exact11-file layout verified. Source archive, launch config
  and initial tensor transport files downloaded and verified by SHA256 before POST.
  Dataset version number not returned; preserve raw creation/roundtrip receipts.
- Two T4s requested, one arm each: mean2/live and single-forward/EMA0.9999.
 60-epoch schedule,500K/50K, physicalBS128, FP32. Ceiling32400s includes setup,
 qualification and both arms; reserve180s. No automatic continuation/retry.

Frozen executable source: `4d4de9a706525d9d3136956e8a62c93e33587fd0`.
Source archive SHA256:
`e55e46fb43f7d3898502928f71728df6f67629badaaad36cc9807e525982f566`.
Package identity:
`9731ab24ae3b4867cb334ab55ade7a1a72afaf7dd22dcb0670d175702ee2a5de`.
Spec identity:
`7c88fd5dda8dc182732ca2a6c133bbf58d5ee0751fe49b389856089b9e8ced4f`.
Both arms use `molgap.family_training_runner`, with training-only native
repeat/resume qualification before the all-arm training barrier is released.

## Preparation reconciliation

release001 failed before prospective publication because its draft omitted the
required supporting evidence ID. No remote request occurred. release002 planned
both arms, then correctly blocked on global RML rebuild: ignored historical
artifacts were absent in the fresh worktree. The two published records were
preserved, not deleted/replanned. Existing custody receipts were used to copy149
opaque retained artifacts from primary desktop, with source/copy SHA equality;
no tensor decoding, model execution or historical evidence rewrite occurred in
that materialization. RML rebuild/frozen check then passed. Preparation resumed
from the exact existing package/prospective identities and passed its final
sealed-source release check. No new physical training attempt was created by
this local reconciliation.

Submission receipts are in `submission/`; canonical prospective records are in
`rml/mean2/` and `rml/single_ema/`. Durable private Kaggle outputs will be under
`/kaggle/working/experiment`, not `/kaggle/temp`. Source and configuration remain
in the private input dataset. Latest observation is not scientific acceptance.

## Next authorized action

No continuous monitor or local slot trigger. On return, query only the actual
ref above and reconcile the same version/run, source, hardware and durable
outputs before any decision. Retrieve/inspect with existing family-output
owners. Checkpoint/source/role/cost/trace acceptance and V5 comparison are still
pending. If the budget stops either arm, compare only common exposure and retain
STOP_FOR_COST/INCONCLUSIVE, not a complete60-epoch endpoint. Do not consume
official validation/test or promote/launch another job from this receipt.
