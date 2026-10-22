# Status

Reconciled exact Kaggle1 version1 jobs on2026-10-22JST: position COMPLETE,
scope/gain ERROR. Both stopped. The
[check report](reconciliation_20261022.json) retains observations and hashes;
[interpretation](reconciliation_20261022.md) records the result and failure.

Position arms passed the existing family mechanical inspector, including
160-pass trace/exposure, selected Live models, finite aligned50K predictions,
resume schema and artifact hashes. Best MAE: Post-RMS0.1401534744 atpass147;
Pre-RMS0.1456912256 atpass135. Pre minus Post is+5.537751meV, paired-row95%
bootstrap interval[+4.534687,+6.516362]meV (2000 samples, seed42).

Scope/gain arms retained30Graph-RMS and20AdaRMS observations, then both failed
while publishing a resume chunk: `Copied artifact checksum mismatch; original
retained`. This is an infrastructure failure, not a scientific negative.
Their output manifests are absent; recovery checkpoints are not yet qualified.
Source review identifies a race between ledger hashing/copying and the parent
refreshing it every30seconds. The exact failed artifact is unconfirmed because
the exception omits its filename/hash.

Only required completed-arm outputs and small metadata/logs were downloaded,
about143MiB onD:. No whole output, cache or chunk history was fetched.
No retry, training, inference, monitor or server handoff was launched.
Scientific terminal acceptance/RML and replay readiness are not established;
the incomplete scope/gain question remains open. Repair immutable sidecar
snapshotting and qualify recovery evidence before authorized continuation.

## Original submission observation (historical)

Both exact Kaggle1 version1 submissions were BOUND and observed QUEUED.
[Submission summary](submission_summary.json) owns immutable job/source bindings;
per-slot submission records retain the authenticated source roundtrip and receipts.
Remote source verification passed. Startup log streams timed out while queued;
no native repeat/resume certificate, TRAIN_STARTED, complete result or scientific
acceptance has been observed. Reconcile these exact jobs after desktop returns;
never submit a successor because this local observation is stale.

- Position: A Post-Node-RMS / B Pre-Node-RMS.
- Scope/gain: C Post-Graph-RMS / D Post-Node-AdaRMS.

All four prospective RML records are published. RML frozen check passed.
Portable check is blocked by inherited historical local-artifact claims absent
from this temporary checkout; no inherited catalog was hydrated. Strict replay
and scientific acceptance remain independent, unproven terminal gates.

The user requested shutdown after submission. No automatic monitor, retry,
server handoff or adoption is released. Prepared copies were retired by the
existing platform adapter after verified private source publication; original
initialization and immutable source packages remain on D:.
