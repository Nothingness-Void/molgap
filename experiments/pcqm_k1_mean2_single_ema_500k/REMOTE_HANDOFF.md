# Kaggle1 paired500K handoff - 2026-10-13 JST

Owner branch: `codex/exp/k1-mean2-single-ema-500k-20261013`.
Owner worktree: `D:/w/k1-mean2-single-ema-500k-20261013`.
[Protocol](protocol.md) and [config](config.json) own parameters and boundaries.

## Corrected version2

The user explicitly authorized correction and resubmission. Same Kaggle1 ref
below, kernel138365645/version2: authoritative scheduler RUNNING. Exact remote
entry bytes and the source/graph dataset membership are verified in
[version2 receipts](attempt_002/submission/source_verification.json).
Initial startup log was empty; native qualification and actual training were
not yet observed. No model acceptance follows from RUNNING.

Executable source `b7c137b23cec8483dba118de3ff2ddfd4a02dab2`,
Spec `a5c527d38864b91730153347b7e49d889e7a87c46b3f07c03cee92b9f9699734`,
package `708e59178a6c0d2bdd6ba87f31b2451f957a8cd54fa382132119fda6e6d0bd42`,
archive `00b2308a7debdcd44fd24abaecb267867b40c7142d3cc5d0e2ff91468ffd0147`.
Fresh private source dataset `nothingnessvoid/molgap-k1-mean2-single-ema-500k-20261013-src-v2`
has11 verified file paths/sizes; source archive and launch bytes passed server
roundtrip. The fixed graph dataset, both scientific recipes and initial tensors
are unchanged. Separate prospective records are under `rml/attempt_002/`.
The shared corrections were selectively integrated into desktop f35ae89f,
without adopting this experiment or moving its custody.

## Submission observation

Terminal reconciliation on2026-10-13 JST supersedes the queue observation:
kernel138365645/version1 is ERROR. Both preflight workers fail resolving the
frozen graph manifest path; neither starts formal training (0epoch/step/sample).
See [attempt decision](decision.md), [acceptance](terminal_acceptance/acceptance.md)
and [attribution](terminal_acceptance/attribution.md). No retry is authorized by
this closure; preserve the owner for the unresolved scientific question.

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

Final `check --frozen --portable` found six more existing retained artifacts
(four historical clean-second checkpoint files and two seed43 preflight logs)
absent in this worktree. Their primary-desktop originals were copied and SHA
checked without decoding. The final portable/frozen check passed. This validates
committed planning/custody/index state, not this pending run's training replay.

Shared budget/mirror implementation was selectively integrated into desktop
commit73f367af and routed in9817d9ef, without this experiment's unfinished
science or canonical trajectories. Both desktop and the owning experiment
branch were pushed; the source executing on Kaggle remains the frozen4d4de9a7
package, not whichever branch happens to be checked out later.

Submission receipts are in `submission/`; canonical prospective records are in
`rml/mean2/` and `rml/single_ema/`. Durable private Kaggle outputs will be under
`/kaggle/working/experiment`, not `/kaggle/temp`. Source and configuration remain
in the private input dataset. Latest observation is not scientific acceptance.

## Next authorized action

Version2 consumes the explicit retry authorization. Reconcile that exact run,
version, frozen identities and durable outputs on return. No continuous monitor,
automatic successor, official-role access or adoption is authorized. Version1
remains NO_TRAIN; do not treat its failure as model evidence or repeat its RML
finalization. Apply the original partial/completed outcome rules to version2.
