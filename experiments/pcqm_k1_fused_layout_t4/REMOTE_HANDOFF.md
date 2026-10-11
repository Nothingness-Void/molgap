# Native T4 pair handoff

The explicitly authorized [repaired attempt002](attempt_002/REMOTE_HANDOFF.md)
has its own source, prospective records and actual API receipt. Use that owner
entry for its dated state; the attempt001 observations below remain historical.

[Attempt001 acceptance](terminal_acceptance/decision.md) supersedes the queue
observation for this attempt: ERROR during initial-state loading, neither arm
trained. Reconcile retained failure evidence; no automatic retry is authorized.

2026-10-11 JST. Desktop-owned question on `codex/exp/k1-fused-layout-t4-100k`,
local custody `D:/w/k1-fused-t4`. No server takeover or heartbeat.

## Exact submission

- Kaggle1 / `nothingnessvoid`, kernel `molgap-k1-fused-layout-100k-s42-v1`,
  kernel ID138066440, version1; script-version identity unknown.
- [API receipt](submission/kernel_push.json) owns response identity and release pins.
  [Dated observation](submission/status_after_push.json) reports QUEUED, not
  native qualification or scientific acceptance. Reconcile this exact run first.
- Frozen executable source `fdcaee6a30496804315e5c8650d48ac1f41ffb7f`;
  subsequent handoff commits do not change executable source.
- Private source `nothingnessvoid/k1-fused-layout-100k-s42-source-v1`;
  [publication](submission/source_dataset_create.json),
  [mounted layout](submission/source_dataset_layout.json) and
  [15-file byte roundtrip](submission/source_dataset_roundtrip.json) retained.
- Fixed input `nothingnessvoid/pcqm4mv2-ogb-fixed-100k-v1`;
  [CPU cache acceptance](submission/cpu_input_acceptance.json) verifies real
  membership, shard hashes and targets without model execution or GPU use.

## Execution and acceptance

[Protocol](protocol.md) owns all settings: original single-forward K1 versus
fused AdamW plus CPU layout, original epoch loaders, FP32/BS128, seed42,
40 epochs each, one arm on each physical T4 in the same Notebook.
No worker reuse, EMA, official validation/test role, or protected evaluation.
User requested completion rather than a four-hour cutoff; platform limits remain
binding. Native all-arm qualification must pass before either formal arm starts.

Best/last checkpoints, saved predictions, timings and traces are atomic worker
outputs retrievable from this exact Kaggle version. Native session interruption
does not authorize retry or continuation. On return use the platform skill and
existing family acceptance / V5 / RML helpers; compare full matched exposure,
primary dev quality and measured train-pipeline cost separately. Bootstrap is
paired-row uncertainty, not training-seed variance. No runtime/model adoption yet.

Local checks:185 targeted tests passed,1 host-limited skip; real CPU input check
passed; RML rebuild and frozen check passed. Native T4 qualification and terminal
RML closure remain pending. Do not substitute a QUEUED status for either.

No local submitter or monitor is needed. Desktop shutdown leaves the Kaggle queue
independent; never submit a duplicate because a local snapshot is stale.
