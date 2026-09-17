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
| codex/exp/gptrans-full-convergence | Active repaired GPTrans-T/K1 convergence and terminal acceptance chain |
| codex/exp/xian-determinism-audit | Bounded Xi'an determinism and same-allocation repeatability follow-up |
| codex/exp/ogb-rich-edgegps-v4-100k | Separate reference/infra work without terminal disposition |
| codex/scnet-distance-angle-triangle-500k | Submitted experiment without a terminal decision at its branch tip |

Inspect these named branches only when working on their question. Their old
root documents do not override the integration branch. Do not wholesale merge
server discovery history to obtain one desktop implementation.

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
remote branch refs were removed. The GPTrans V4 and Xi'an worktrees were
committed and pushed, but remain active until their terminal questions close.
No remote scheduler was queried and no compute workload was changed by this audit.
