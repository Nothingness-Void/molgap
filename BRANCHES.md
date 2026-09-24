# Branch routing

This file owns checkout routing, not model or job status. Read CURRENT_STATE.md
for the desktop state, ROADMAP.md for its queue, and the owning experiment for
dated evidence. Fetch before comparing a remote snapshot.

| Long-lived branch | Role |
|---|---|
| master | Stable delivery; promotion is a separate reviewed action |
| molgap-desktop | Desktop integration, SCNet work, full training and official evaluation |
| molgap-server | Independent server discovery; read its own CURRENT_STATE.md |
| archive | Inactive/rejected histories; never infer live state here |

## Active temporary branches

The 2026-09-24 local branch audit found one retained desktop temporary ref:
`codex/fix/rml-500k-reference` at `b78b978d`. It contains prospective DSAR/DSMR
work and its checkout has uncommitted experiment/evidence files. The committed
tip is also at `origin/codex/fix/rml-500k-reference`; reconcile its evidence
before integrating it into `molgap-desktop`. This routing statement does not
assert a remote job's current state.

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

## V5 workflow boundary

The V5 contract is the stable operating topology for this checkout. The
`molgap-desktop` branch owns full/evaluation/submission work and any explicitly
desktop-owned 500K question; `molgap-server` owns its independent bounded
100K/500K research loop. Desktop shutdown does not transfer jobs to the server,
and no live cross-machine monitor, takeover, or conversation bridge is implied.
When desktop returns, reconcile the authoritative remote state and durable
artifacts before resuming or submitting anything. Exchange evidence through
reviewed Git commits and compact indexes. See the [V5 common contract](docs/operations/MOLGAP_COMMON_DIRECTION_V5_FINAL.md)
and [V5 desktop handoff](docs/operations/DESKTOP_AGENT_HANDOFF_V5_FINAL.md).
