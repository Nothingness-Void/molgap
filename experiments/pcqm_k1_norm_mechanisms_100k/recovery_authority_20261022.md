# Scope/gain continuation authority — 2026-10-22

The user explicitly requested detailed repair and continuation after repair.
This releases one Kaggle1 continuation of failed kernel 140063114/version1,
on the same owning question and branch. It supersedes the original no-retry
restriction for this infrastructure recovery only. Position kernel140062850
completed and is not released for retraining.

Both retained checkpoints passed the shared resume validator: Graph-RMS at
30 passes/23430 updates/2999040 presentations; AdaRMS at20 passes/15620
updates/1999360 presentations. Checkpoint hashes agree with retained traces;
selected models, optimizer state and aligned development predictions agree.
Durable validation is D:/molgap-artifacts/recovery/k1-norm-mechanisms-100k/v2/recovery_validation.json.

Keep the original source package, recipe, seed, data roles, initialization,
model mechanisms and160-pass total target. Only separately pinned, committed
infrastructure repair files may overlay the frozen package: atomic JSON
sidecar snapshotting, checksum error context and source-verifier support for
that explicit repair. Bootstrap records the executed repair provenance.

The retained failure is a resume-publication checksum mismatch, not a science
negative. Hash-then-copy permits an atomically refreshed sidecar to change
generation; the original exception does not identify which file raced.
Snapshot bytes fix that race without weakening model/checkpoint checksums.

Continuation ceiling:23500 entry-wall seconds, two T4 devices,180 seconds
cleanup reserve. Together with10435.946233888 measured T4-device seconds from
version1, the maximum allocation is15.9544295094 T4-hours, below the original
16-hour ceiling. Queue, CPU and wall units remain separate. No extra epochs,
protected roles, monitor, server takeover or further automatic retry released.

Publish an actual response/version receipt and source roundtrip before calling
the continuation bound. Native qualification and observed training remain
separate from submission; terminal scientific/RML acceptance remains pending.
