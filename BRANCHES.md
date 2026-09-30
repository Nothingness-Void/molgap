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

## New desktop experiment lifecycle

Start each new desktop-owned research question from the verified current
`molgap-desktop` tip in a dedicated `codex/exp/<question>` branch and separate
worktree. Check the local/remote desktop relationship and working-tree state
before choosing that base. Do not use a retained experiment branch, `archive`,
or `molgap-server` as the base for a new desktop question.
Select the question from desktop routing, RML, and the linked canonical evidence
before inspecting implementation on another branch. A retained branch is read
only when its own question or an explicit integration task requires it.

Do the prospective plan, implementation, submission, remote reconciliation,
training acceptance, scientific decision, and terminal RML update on the
experiment branch. Keep retries and seeds for the same question there. A
finished job is not itself a finished experiment: review acceptance and make
the scientific and Git-routing decisions promptly after terminal evidence is
available. Do not leave an accepted desktop result only on a temporary branch.
Do not merge a live experiment merely because its kernel was submitted or
finished.

After a reviewed terminal decision:

| Decision | Git destination |
|---|---|
| Positive and adopted | Merge the experiment branch into `molgap-desktop`; rebuild and check desktop RML. |
| Accepted diagnostic, no model promotion, with reusable implementation | Merge its reviewed evidence and reusable implementation into `molgap-desktop`; record explicitly that the model recommendation is unchanged. Rebuild and check desktop RML. |
| Negative under its contract | Merge the complete experiment history into `archive`. Bring only the accepted canonical decision/evidence needed for desktop RML discovery into `molgap-desktop`; do not merge rejected implementation. Rebuild and check desktop RML. |
| Pending, incomplete, or terminal without an adoption/archive decision | Keep the experiment branch until its routing decision is recorded. |

Verify the experiment tip is durably reachable from the destination before
deleting its temporary ref. An archive preservation merge may retain ancestry
without changing the archive tree; it does not turn an old branch into a live
work entry point. Promotion from `molgap-desktop` to `master` is separate.

## Retained Checkout Map

These branch-local records are not present in the integration checkout.
Use the mapped owner; do not create substitute experiment directories here.
Paths are local checkout locators, not portable artifact provenance.
Check branch/HEAD/status before reading or integrating; fetch before comparing
remote Git snapshots. Inspect only the question required by the task.

| Question / owner | Retained checkout | Branch-local entry |
|---|---|---|
| Desktop pretraining pair: `codex/exp/pretraining-family-pair` | `C:/Users/17449/.codex/worktrees/hierarchy-exposure/molgap` | [terminal decision](C:/Users/17449/.codex/worktrees/hierarchy-exposure/molgap/experiments/pcqm_pretraining_family_pair/terminal_decision.md); [fusion local review](C:/Users/17449/.codex/worktrees/hierarchy-exposure/molgap/experiments/pcqm_pretraining_family_pair/fusion_comparison.md) |
| Desktop geometry: `codex/exp/geometry-v4-500k-pair` | `C:/Users/17449/.codex/worktrees/geometry-v4-500k/molgap` | [fixed-blend scale decision](C:/Users/17449/.codex/worktrees/geometry-v4-500k/molgap/experiments/pcqm_geometry_v4_500k_pair/fixed_blend_scale_decision.md) |
| Desktop chemical auxiliary: `codex/exp/gptrans-chemical-aux` | `C:/Users/17449/.codex/worktrees/gptrans-init-fidelity/molgap` | [STATUS](C:/Users/17449/.codex/worktrees/gptrans-init-fidelity/molgap/experiments/pcqm_gptrans_chemical_aux/STATUS.md), its linked submission_v4 observation and frozen training protocol |
| Desktop precision: `codex/exp/gptrans-fp16-precision-100k` | `D:/文档/molgap-exp/molgap-gptrans-fp16-precision-100k` | Branch-local acceptance / reference-binding records |
| Server DSAR/DSMR: `codex/fix/rml-500k-reference` | `D:/w/rml-500k-fix` | Old branch-local drafts; job/attempt identity incomplete, not live scheduler evidence |
| Desktop retained scale-fit cache | `D:/w/scale-fit` | [Accepted terminal navigation](experiments/pcqm_scale_fit_retained/STATUS.md); diagnostic implementation/evidence integrated into desktop, ignored predictions retained here; no training release |

The pretraining fusion review is uncommitted other-agent work pending review
and integration; its local presence is not a committed desktop evidence claim.
Geometry retains technical reconciliation and final Git routing; chemical
auxiliary retains its dedicated owner through terminal acceptance and disposition.
DSAR/DSMR remains server-owned despite the historical desktop-ref audit wording.
Do not adopt, monitor or submit successors from its stale local RUNNING text.
Retained refs/checkouts are not authority to reopen a closed question.

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
