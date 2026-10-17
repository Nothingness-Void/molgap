# Branch routing

This file owns branch and custody navigation, not model/job status. CURRENT_STATE.md owns state,
ROADMAP.md owns queue order; experiment decisions own dated evidence. Fetch before remote comparison.

| Long-lived branch | Role |
|---|---|
| master | Stable delivery; promotion is a separate reviewed action |
| molgap-desktop | Desktop integration, SCNet work, full training and official evaluation |
| molgap-server | Independent server discovery; read its own CURRENT_STATE.md |
| archive | Inactive/rejected histories; never infer live state here |

## New desktop experiment lifecycle

Accepted local K1 execution-speed custody belongs to
`codex/exp/k1-fused-layout-speed-100k`; its
[decision](experiments/pcqm_k1_fused_layout_speed/terminal_decision.md) owns
diagnostic non-promotion routing. Physical checkout removal does not change
production, remote jobs or that decision.

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
| Pending, incomplete, or terminal without an adoption/archive decision | Keep the owner branch/recovery ref and exact evidence; a verified cold snapshot may replace its physical checkout. |

Delete a temporary ref only after durable destination reachability. Archive merges
may preserve ancestry without changing the tree or making old work live.
Desktop-to-master promotion remains separate.

## Retained Checkout Map

This heading remains for existing links. Use the
[retirement index](docs/operations/LOCAL_RETIREMENT_INDEX.json) to resolve an
owner ID to its immutable original HEAD, archive/recovery ref, repository-relative
canonical path and verified external custody manifest. The executor publishes
actual snapshot refs and manifest hashes after preservation; an unpublished entry
is pending custody, never proof of archival. Physical paths are historical locators.

Checkouts are temporary. Keep the primary checkout and only worktrees needed by
active local work. Preserve exact source/configuration, dirty source/RML bytes,
necessary unique model/replay artifacts and retrieval identities, verify recovery,
then remove a completed checkout promptly. Pending scientific questions can also
use recoverable cold snapshots; their decisions, unknowns and owner branches remain
unchanged. Archival does not adopt positive results or turn pending science negative.
See the [local storage lifecycle](docs/operations/LOCAL_STORAGE_RETIREMENT.md).

Restore only the selected owner/question. Do not hydrate an inherited RML catalog
or recreate old checkouts merely to make historical local-artifact declarations pass.
Remote reconciliation and every training/evaluation release retain their existing
platform, role and scientific authority requirements.

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
