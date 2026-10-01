# Branch History

The [2026-10-01 archive custody record](https://github.com/Nothingness-Void/molgap/blob/b1435ba939c39d88b6af55aca04c0b62f82314c3/docs/operations/BRANCH_ARCHIVE_20261001.md)
preserves four retired temporary histories and an unreviewed pretraining snapshot;
the pending chemical-aux working ref remains active. This is administrative custody,
not scientific acceptance or a change to historical qualification gaps.

Historical audit records moved intact from BRANCHES.md. These dated snapshots
are not live routing or scheduler truth; use [BRANCHES.md](../../BRANCHES.md)
for checkout ownership and CURRENT_STATE.md for the desktop summary.

## Retained branch awaiting reconciliation

The 2026-09-30 local reconciliation is recorded in
`docs/operations/LOCAL_RECONCILIATION_20260930.md`. It preserves eight old
infrastructure drafts and the inconclusive frozen-transfer history in archive,
and identifies the geometry, precision and DSAR/DSMR branches still in use or
awaiting acceptance. Historical downloaded payloads remain retained locally.

The 2026-09-24 local branch audit found one retained desktop temporary ref:
`codex/fix/rml-500k-reference` at `b78b978d`. It contains prospective DSAR/DSMR
work and its checkout is clean. The committed tip is also at
`origin/codex/fix/rml-500k-reference`; reconcile its evidence before
integrating it into `molgap-desktop`. This routing statement does not assert a
remote job's current state.

The superseded shared experiment-infrastructure tips were patch-equivalent to
the desktop implementation. Archive merge `bbd98fb3` preserves their original
commit ancestry without changing the archive tree. Their local `codex/` refs
were retired after the archive push. Dirty worktrees were detached at their
original commits with their tracked and untracked changes retained. The four
closed GPTrans flow / K1 pair-screen tips were already in `archive`; their
remaining local refs were retired under the same preservation rule.

At the earlier PairNorm cleanup snapshot, no desktop temporary experiment
branch had an unresolved terminal job. The PairNorm 100K shortlist and
training-only 500K profile source history was Git-merged into
`molgap-desktop` at `43db4a9d`. The later standalone
PairNorm 500K experiment closed negative at `0460ba6`; its source history is
preserved by archive merge `234f511`. Its selected canonical evidence remains
indexed on `molgap-desktop`. The temporary refs were retired.

Inspect retained branches only when working on their question. Their old root
documents do not override the integration branch. Do not wholesale merge
server discovery history to obtain one desktop implementation.

The separate Noisy Nodes + Pair Update Norm 500K experiment was landed in RML
at `507ced7` and merged into `molgap-desktop` at `5e3c1ed`. The later tip of
`codex/exp/gptrans-noisy-pair-norm-500k` contains server DSAR/DSMR work; that
tip was not merged and does not add desktop acceptance tasks. The standalone
PairNorm bridge closed above is a different experiment.

The GPTrans-T/K1 convergence branch reached terminal acceptance at `cac804c`.
Its source tip `30872aa` was Git-merged into `molgap-desktop` at `91bccb87`;
the desktop IMS resource-repair tip `e777ebc` was merged at `430e4dc6`. The
positive distance-angle tip `86a4489` was Git-merged at `63876d32`. The
negative Xi'an audit tip `3fc1d80` remains in `archive`; its accepted evidence
is indexed on `molgap-desktop`. These temporary refs were retired.

## Cleanup evidence

The full-run branch through fb05260 was integrated with explicit resolution of
CURRENT_STATE.md and ROADMAP.md. Superseded scratch-control, submission-backup
and EdgeState304 tips were preserved as archive merge parents (b2f65e0); that
merge intentionally retained the archive tree rather than activating their
code. Earlier retired tips were already reachable from origin/archive.
Archive merge `234f511` similarly preserves the closed desktop PairNorm,
GPTrans-T/K1 convergence, feature-denoising, PairValue, Xi'an audit, and
distance-angle source tips without changing the archive tree. The PairNorm
profile and 100K normalization tips are ancestors of its 500K tip. Their eight
local and remote temporary refs were retired only after remote archive
reachability was verified. The positive/operational desktop tips listed above
were later Git-merged into `molgap-desktop`; the negative and inconclusive
500K, feature-denoising, PairValue, and Xi'an tips remain archived.
Archive merge `b0177970` also preserves the terminal desktop conditional-flow,
K1 structural-lite no-train, K1 sparse-pair, and 500K module-attribution tips.
Their four refs, plus the merged IMS resource-repair ref, were retired after
remote ancestry checks. Server-based research refs and the mixed, dirty
Noisy Nodes + PairNorm checkout were not merged into desktop or cleaned here.
Working directories and untracked payloads were retained when branch refs were
removed. Detached old worktrees are historical snapshots, not work entrypoints.
Completed historical ESGPS6-304 (03f5253) and GPTrans-T 500K (c86970d) histories
were also preserved in archive. Their positive results retain their original
contracts; archival does not reclassify them as rejected. GPTrans full execution
is integrated separately. Retrieve an old file with `git show <commit>:<path>`.

The old local master tip 36a8156 was already reachable from remote desktop.
The local master ref was aligned to origin/master after verifying that
preservation; no submission-process history was promoted to delivery.

Use a clean `molgap-desktop` worktree for desktop integration. Check branch and
status before using the primary local checkout; it may contain unrelated work.

The completed matched-500K V4, geometry-transfer, and IMS V4 infrastructure
branches were integrated or preserved in `archive` before their local and
remote branch refs were removed. Xi'an remains archived; the distance-angle
tip is also an ancestor of `molgap-desktop` through merge `63876d32`.
No remote scheduler was queried and no compute workload was changed by this audit.

The completed GPTrans-T 100K V4 branch was integrated into `molgap-desktop` by
merge `65f0f53`. Its accepted reference and closed promotion decision remain in
`experiments/pcqm_gptrans_t_100k_v4/`. The OGB-rich EdgeGPS9 100K V4 branch
contained reusable reference infrastructure but no terminal experiment
disposition; merge `becbcf8` preserves that history in `archive` without
activating its tree. Both temporary branch refs were then retired.

The terminal GPTrans-T input-initialization 100K history is preserved by
archive ancestry merge `fc9d0d14`. Its accepted negative decision, canonical
RML records, and artifact provenance are indexed on `molgap-desktop` without
merging the rejected model or launcher implementation.
