# Version 3 continuation authority — 2026-10-22

The user explicitly requested test reduction, correction of weak assertions,
repair of continuation, resubmission to Kaggle1, and confirmation of actual
training. This authorizes one version 3 continuation of kernel 140063114
(`nothingnessvoid/molgap-k1-rms-scope-gain-160ep-100k-s42`).

Version 2 failed during actual-resume qualification before any new formal
training. Its retained terminal record is
[recovery_v2_terminal_20261022.json](recovery_v2_terminal_20261022.json).
Restore the original version 1 checkpoints: Graph-RMS at 30 completed passes /
23430 updates and AdaRMS at 20 completed passes / 15620 updates, using the
already verified version 2 local resume bundles. The fix isolates loaded
optimizer state between scratch probes; it does not reset formal training.

Keep the original source package, data, roles, architecture, initialization,
FP32 recipe, optimizer/schedule and 160-pass total target. Bind the committed
infrastructure overlay separately through the existing repair attestation.
The completed position slot is outside this continuation.

The entry-wall ceiling is 23100 seconds on two native T4 devices, including a
180-second cleanup reserve. Prior measured cost is 10435.946233888 T4-device
seconds for version 1 and 845.2848590100001 for version 2. Even the full new
ceiling totals 57481.231092898 T4-device seconds (15.967008637 hours), below
the original 16 T4-hour ceiling. Queue time is not device time.

Release uses the existing resume preparation, source verification, accelerator
push and receipt reconciliation. Unknown publication outcomes must be
reconciled before retry. Success requires native preflight admission and
observed formal progress beyond each restored cursor, not RUNNING alone.
Scientific acceptance and per-arm replay readiness remain terminal gates.
