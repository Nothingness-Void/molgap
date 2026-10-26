# Submission state

2026-10-26 JST: both exact Kaggle1 version1 jobs are ERROR. Native preflight
completed and all four arms trained; the first10-pass portable resume export
failed when tensor-bearing Clean-EMA cursor state entered a JSON manifest.
The [failure inspection](failure_inspection/inspection.md) owns observed progress,
partial metrics, scope and costs. The scientific question remains unresolved.
No successor or retry was submitted during inspection.

| Slot | Kernel ID / version | Scheduler | Durable operations |
| --- | --- | --- | --- |
| single | 140826099 / 1 | ERROR | [pcqm-k1-init-forward-500k-single-s42](D:/molgap-artifacts/runs/pcqm-k1-init-forward-500k-single-s42/v1/operations/) |
| mean2 | 140826274 / 1 | ERROR | [pcqm-k1-init-forward-500k-mean2-s42](D:/molgap-artifacts/runs/pcqm-k1-init-forward-500k-mean2-s42/v1/operations/) |

Frozen execution source remains `64933302a449cd48961a610028a17bd6ae6d2e23`.
Both original v1 packages, run IDs, launch bytes and four published prospective
trajectories were reused. Recovery extracted only the pinned comparison declaration
from each verified source package into its acceptance attachments, then used existing
release binding and platform submission. Each workflow report retains the original
error and recovery facts. No source refreeze or prospective republication occurred.

The local staging fix is `b6f17d11` (cherry-pick of `776868f6`); it does not replace
the frozen worker source. Both source datasets are private version1, with complete
upload byte verification and actual kernel/version/source receipts in operations.
The initial post-create status calls returned HTTP403; later authoritative checks
confirmed the same datasets ready before journal reentry. Neither dataset was recreated.

Upload/roundtrip copies were retired by the existing bound-submission cleaner.
Canonical planning records and immutable source packages remain retained.
Initial startup log requests timed out. Subsequent terminal inspection retrieved
per-arm logs and canonical traces:10 completed observations/39060 steps per arm;
the logger stopped at9. Full checkpoint validity and native-proof verification
are pending; these are not complete replay-ready results.

## Authorized version2 continuation

The user authorized repair and continuation of both jobs. The
[continuation authority](continuation_authority.md) retains the original30-pass
target and lower per-slot allocation ceilings. All four real pass10 checkpoints
passed the owning resume validator; see
[checkpoint verification](continuation_checkpoint_verification.json).
No initial state, optimizer, scheduler horizon or scientific recipe is reset.

The existing resume preparation owner produced both successor payloads with a
SHA-bound two-module repair overlay. The original source archive and Spec stay
frozen. Both private source datasets are version2; all43 files per source passed
the existing byte-level roundtrip. Both kernel continuations are SUBMISSION_BOUND.

| Slot | Actual kernel / version | Accepted UTC | Retained receipt |
| --- | --- | --- | --- |
| single | 140826099 / 2 | 2026-10-26 05:16:54 | [receipt](continuation/v2/single/launch_receipt.json) |
| mean2 | 140826274 / 2 | 2026-10-26 05:17:59 | [receipt](continuation/v2/mean2/launch_receipt.json) |

The adjacent source verification records bind the actual remote entry bytes,
private T4 configuration and source/data mounts. Recovery manifests bind the
two reviewed replacement modules; original archive identities remain unchanged.
A selected-file transport failure interrupted source verification before POST.
Luna's exact-file probe succeeded; its cause remains unconfirmed. Sequential
journal reentry reused hash-verified files and completed without another dataset
version or kernel POST. The transport diagnostic is retained in single operations.

Operations: `D:/molgap-artifacts/recovery/k1-init-forward-500k/{single,mean2}/v2/operations/`.
Only Luna checked remote training progress. It confirmed formal TRAIN_LOG epoch11
for all four arms:42966 optimizer steps,5499648 sample presentations, next_batch0,
checkpoint_optimizer_steps42966 and full_pass timing. These are formal train-phase
observations, excluding the scratch resume-proof updates. Single was observed
at2026-10-26 05:36:30 UTC; mean2 at05:48:23 UTC. See the retained
[single observations](continuation/v2/single/training_progress_observations.json)
and [mean2 observations](continuation/v2/mean2/training_progress_observations.json).
Both exact version2 jobs were RUNNING without failure messages at those checks.
There was no separate restored-cursor log line; checkpoint validation and the
SHA-bound resume inputs own the pass10 origin. No healthy training artifacts
were downloaded. Luna stopped this startup check after all four confirmations.
The30-pass target remains unfinished; terminal acceptance and four independent
strict replay qualifications remain pending.
