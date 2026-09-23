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

| Branch | Reason retained / evidence entry |
|---|---|
| codex/exp/gptrans-pair-norm-500k | Submitted Kaggle3 PairNorm bridge; terminal artifacts are not locally available |

Inspect these named branches only when working on their question. Their old
root documents do not override the integration branch. Do not wholesale merge
server discovery history to obtain one desktop implementation.

The GPTrans-T/K1 convergence branch reached terminal acceptance at `cac804c`.
Its experiment evidence is integrated into `molgap-desktop`; retain the branch
ref for source provenance until its commits are durably reachable from an owner
or archive branch.
The distance-angle and Xi'an audit branches reached terminal decisions at
`86a4489` and `3fc1d80`. Their evidence is integrated into `molgap-desktop`;
retain the branch refs under the same provenance rule.

## Cleanup evidence

The full-run branch through fb05260 was integrated with explicit resolution of
CURRENT_STATE.md and ROADMAP.md. Superseded scratch-control, submission-backup
and EdgeState304 tips were preserved as archive merge parents (b2f65e0); that
merge intentionally retained the archive tree rather than activating their
code. Earlier retired tips were already reachable from origin/archive.
Working directories and untracked payloads were retained when branch refs were
removed. Detached old worktrees are historical snapshots, not work entrypoints.
Completed historical ESGPS6-304 (03f5253) and GPTrans-T 500K (c86970d) histories
were also preserved in archive. Their positive results retain their original
contracts; archival does not reclassify them as rejected. GPTrans full execution
is integrated separately. Retrieve an old file with `git show <commit>:<path>`.

The old local master tip 36a8156 was already reachable from remote desktop.
The local master ref was aligned to origin/master after verifying that
preservation; no submission-process history was promoted to delivery.

The primary local checkout D:/文档/molgap now uses molgap-desktop. The previous
desktop checkout is a detached snapshot; use the primary checkout for new work.

The completed matched-500K V4, geometry-transfer, and IMS V4 infrastructure
branches were integrated or preserved in `archive` before their local and
remote branch refs were removed. The Xi'an audit and distance-angle branches
have terminal decisions recorded above; their refs remain for provenance.
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
