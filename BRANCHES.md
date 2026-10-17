# Branch routing

This file owns checkout routing, not model/job status. CURRENT_STATE.md owns state,
ROADMAP.md owns queue order; experiment decisions own dated evidence. Fetch before remote comparison.

| Long-lived branch | Role |
|---|---|
| master | Stable delivery; promotion is a separate reviewed action |
| molgap-desktop | Desktop integration, SCNet work, full training and official evaluation |
| molgap-server | Independent server discovery; read its own CURRENT_STATE.md |
| archive | Inactive/rejected histories; never infer live state here |

## New desktop experiment lifecycle

Accepted K1 execution-speed custody: `codex/exp/k1-fused-layout-speed-100k`;
[decision](experiments/pcqm_k1_fused_layout_speed/terminal_decision.md) owns
diagnostic non-promotion routing. The owner is a Git/evidence identity, not a
permanent checkout or a requirement to duplicate ignored artifacts.

New questions use a separate `codex/exp/<question>` worktree from verified desktop,
never a retained experiment/archive/server tip. Verify upstream/worktree first.
Query desktop routing/RML/evidence; inspect other branches only for their question or explicit integration.

Keep plans/implementation/submission/reconciliation/acceptance/decision/RML,
including seeds/retries, on the experiment branch. A finished job is not closure.
Promptly accept/decide/route evidence; never leave accepted evidence only on temporary
refs or merge live work merely because a kernel was submitted/finished.

After a reviewed terminal decision:

| Decision | Git destination |
|---|---|
| Positive and adopted | Merge the experiment branch into `molgap-desktop`; rebuild and check desktop RML. |
| Accepted diagnostic, no model promotion, with reusable implementation | Merge its reviewed evidence and reusable implementation into `molgap-desktop`; record explicitly that the model recommendation is unchanged. Rebuild and check desktop RML. |
| Negative under its contract | Merge the complete experiment history into `archive`. Bring only the accepted canonical decision/evidence needed for desktop RML discovery into `molgap-desktop`; do not merge rejected implementation. Rebuild and check desktop RML. |
| Pending, incomplete, or terminal without an adoption/archive decision | Keep the experiment branch until its routing decision is recorded. |

Delete a temporary ref only after durable destination reachability. Archive merges
may preserve ancestry without changing the tree or making old work live.
Desktop-to-master promotion remains separate.

## Local checkout and artifact lifetime

Experiment ownership belongs to the recorded Git branch/commit and canonical
contract, acceptance and decision records. It does not require a permanent local
worktree. Retain source and compact canonical/RML records in their reviewed Git
destination. Large artifacts belong in verified durable storage; download only
the owning task's pinned acceptance inputs into temporary local storage.

Do not hydrate an entire inherited RML catalog into each new worktree. A local
availability claim or a repository-wide check failure is not authorization to
copy unrelated weights, graph shards, profiler traces, wheels or raw outputs.
Report the missing locator and owning check instead; do not weaken that check,
claim success or silently rewrite accepted evidence.

Retire a terminal checkout after its source/metadata reach their reviewed Git
destination, required non-Git artifacts have verified retrievable custody, and
no active task uses it. Pending branch routing requires preserving the Git
branch and unresolved work, not an unconditional copy of every local payload.
Before removal, verify exact paths, dirty/untracked work and hardlinks; deleting
a Git branch alone does not clean ignored files or establish artifact custody.
A Git archive merge does not preserve ignored payloads.

Migrate storage through an explicit hash-bound custody record. Preserve frozen
scientific evidence and honestly reconcile local availability claims using the
owning supported mechanism. Until that mechanism/check supports the intended
storage location, report the limitation; do not delete the only retained copy
or leave a local claim pointing to a missing file. These are storage rules,
not new scientific gates, training authority or automatic deletion.

## Experiment custody index

The [canonical custody index](docs/operations/EXPERIMENT_CUSTODY.md) records
question/branch/commit entries without requiring resident physical checkouts.

## Historical Audit Pointer
The [branch history](docs/operations/BRANCH_HISTORY.md) preserves the previous
audit and cleanup records intact, including dated commit/reachability claims.
The [2026-09-30 reconciliation](docs/operations/LOCAL_RECONCILIATION_20260930.md)
owns that inventory. Neither audit is current remote scheduler truth.
Use a clean desktop worktree for integration and preserve unrelated changes.

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
