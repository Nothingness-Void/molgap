# Status

Terminal acceptance completed on2026-10-23. Position140062850/version1 and
scope/gain140063114/version3 are COMPLETE. All four arms passed the existing
mechanical inspector at160 passes/124960 updates/15994880 presentations;
scope/gain preserved exact version1 trace prefixes after accepted native
actual-resume preflight. No new training was launched during acceptance.

[Decision and attribution](terminal_decision_20261023.md): Graph-RMS is best
at0.138508950eV,1.645meV better than Post-Node RMS, below the3meV material
nomination gate. Post-Node/Graph serialization references close CLOSED;
Pre-Node/AdaRMS close NEGATIVE_UNDER_CONTRACT. No model adoption or scale-up.

All four independent RML terminal pipelines completed; derived rebuild and
check --frozen passed. Strict replay readiness remains excluded: strict V5
comparison readiness is not established, native scratch tensor proofs were
not independently reverified locally, and large artifacts have explicit
machine-local custody. [Acceptance method](terminal_acceptance_20261023/method.md)
and [records](terminal_acceptance_20261023/locations.json) retain the details.

Total observed physical entry allocation is26.048143 native T4-device hours,
including failed scope/gain versions1/2; no inherited ledger double-counting.
Only142.80MiB of new final scope/gain outputs/metadata was retrieved toD:.
Owner branch retains accepted evidence; long-lived integration/archive routing
is pending reviewed custody. No monitor or successor has been created.

## Version3 submission observation (historical)

Kaggle1 continuation **140063114/version3** was accepted and source-bound on
2026-10-22. Last scheduler observation: **QUEUED**, 07:10:33 UTC. The user
requested ending the wait; native preflight and formal resumed progress have
not been observed. Do not treat submission as successful training.

[Version3 authority](recovery_authority_v3_20261022.md) preserves the original
30-pass Graph-RMS and 20-pass AdaRMS checkpoints and 160-pass total target.
Repair commit `e2038d8b` isolates optimizer state between scratch probes.
[Test cleanup](test_cleanup_20261022.md): 38 redundant/identity cases removed,
one causal regression added, 87 cases collected; the selected21 passed once
in55.63seconds. Scientific acceptance and replay readiness remain pending.

Exact receipt, source verification, remote file hashes and timestamped scheduler
observations are retained at
`D:/molgap-artifacts/recovery/k1-norm-mechanisms-100k/v3/operations/`.
Kernel: `nothingnessvoid/molgap-k1-rms-scope-gain-160ep-100k-s42`, version3.
No recurring monitor was created. Reconcile this exact version before any retry.

## Version2 terminal diagnosis (historical)

Kaggle1 scope/gain continuation **140063114/version2** is terminal ERROR.
It failed native actual-resume preflight before formal training; both worker
logs report optimizer update exposure outside acknowledged work. The loaded
optimizer state was reused across scratch probes; the first can mutate shared
CPU AdamW step tensors before the second restore. [Terminal diagnostic](recovery_v2_terminal_20261022.json)
owns exact source, log and native-cost observations. The original on-disk
version1 checkpoints remain retained. Version3 has not been submitted.

The [124-case audit](test_audit_20261022.md) explains why the passing local
component/orchestration suite did not cover populated native optimizer restore.
An additional regression batch was canceled at the user's request. Only one
small causal AdamW isolation check was executed afterward; it passed, and
native continuation remains unqualified.

[Recovery record](recovery_20261022.json) owns the historical accepted response
and initial RUNNING observation; it does not assert native training admission.
[Continuation authority](recovery_authority_20261022.md) bounded version2.
Original model/recipe/data identity and160-pass target remain.

The retained Graph-RMS checkpoint has30 completed passes/23430 updates;
AdaRMS has20 completed passes/15620 updates. Immutable sidecar snapshotting repairs the
resume-publication race. The one focused regression batch passed124 tests.
The original scientific package is preserved; three committed infrastructure
repair files are separately pinned and verified by the bootstrap/workers.

Version2 failed before formal resumed training and native qualification.
Terminal acceptance and per-arm RML/replay readiness remain pending. Position
version1 remains complete and was not resubmitted. No monitor/server handoff
or additional retry was released.

## Version1 reconciliation (historical, before repair)

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
