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

The [bounded K1 scale attribution](experiments/pcqm_k1_scale_attribution_20261025/README.md)
is an accepted NO_TRAIN diagnostic. Reviewed source/evidence and reusable
compatibility/guard repairs from `codex/exp/k1-scale-attribution-20261025` use
the diagnostic non-promotion route into desktop; the model recommendation
is unchanged. Its [custody manifest](experiments/pcqm_k1_scale_attribution_20261025/artifact_custody.json)
owns the small retained prediction matrix. Original negative tail training
remains a separate owner, `codex/exp/k1-tail-regularization-500k`, until its
independent strict qualification and terminal custody are recorded.

The [retained hop-fusion qualification](experiments/pcqm_k1_hop_fusion_qualification/terminal_decision.md)
is a completed NO_TRAIN saved-prediction diagnostic. Its evidence, thin
reproduction wrapper and fixed predictions integrate into desktop through the
diagnostic non-promotion route; no rejected model/catalog implementation is
included. The source commit691bce0c remains bound by its prospective record.

K1 equal-content hop-shell100K is terminally accepted without adoption.
Full owner history through `499571f9` is reachable from pushed archive `9cca9524`.
Desktop retains [strict qualification and attribution](experiments/pcqm_k1_hop_shell_100k/analysis_zh.md),
including two complete replay entries and the completed local frozen-shell diagnostic.
[Terminal custody](experiments/pcqm_k1_hop_shell_100k/terminal_custody.md) owns
source restoration and artifact locations. No hop-shell model/catalog integration
or experimental shared execution extension is adopted here.

K1 Batch Renorm scale-versus-affine100K is terminally accepted without adoption.
Full history through `c677aa45` is reachable from pushed archive `f92e4694`.
Desktop retains [strict replay and final attribution](experiments/pcqm_k1_batch_renorm_100k/final_attribution.md)
for both arms and the completed local Clean-BN diagnostic. The
[custody record](experiments/pcqm_k1_batch_renorm_100k/terminal_custody.md) owns restoration and artifact locations;
no rejected BatchRenorm implementation or catalog hook is adopted.

K1 fixed-versus-learned MsgNorm100K is terminally accepted without adoption.
Owner history through `4105cd3d` is reachable from pushed archive merge
`548082d8`. Desktop retains the [decision and strict replay evidence](experiments/pcqm_k1_message_norm_100k/qualification_20261024.md).
Its [custody record](experiments/pcqm_k1_message_norm_100k/terminal_custody.md)
owns artifact locations and restoration; no MsgNorm implementation is adopted.

K1 graph-wise message DropPath100K is owned by
`codex/exp/k1-message-droppath-100k`, checkout
`D:/molgap-worktrees/k1-message-droppath-100k`.
[Owning status](https://github.com/Nothingness-Void/molgap/blob/codex/exp/k1-message-droppath-100k/experiments/pcqm_k1_message_droppath_100k/STATUS.md)
retains its Kaggle1 comparison; no active implementation is adopted here.

BondLocal plus pair-update-normalization100K is owned by
`codex/exp/gptrans-bondlocal-pairnorm-100k`, checkout
`D:/molgap-worktrees/gptrans-bondlocal-pairnorm-100k`. Its
[status](https://github.com/Nothingness-Void/molgap/blob/codex/exp/gptrans-bondlocal-pairnorm-100k/experiments/pcqm_gptrans_bondlocal_pairnorm_100k/STATUS.md)
owns Kaggle3 identity, receipts and reconciliation. Keep the unresolved question
on its owner; no active model or implementation adoption is implied.

K1 normalization mechanisms is closed without model adoption. Full owner history
through real automatic verification `48e4c6a5` is reachable from pushed archive
`aab98f69` (including original closure `710b8834`); desktop retains
the [accepted evidence and strict qualification](experiments/pcqm_k1_norm_mechanisms_100k/qualification_20261023.md).
Restore that owner only when its normalization implementation is needed; its
ordinary read-only verification commands preserve the frozen training packages.
The completed physical checkout was retired after verified external custody;
the qualification method gives the exact restore command and custody pointer.

The [single-model RTX5060 lifecycle exercise](experiments/local_k1_ten_epoch_flow/attempt_3/method.md)
closed `INFRASTRUCTURE_ONLY`. Source history through `cd8b4b76` is reachable from
the same archive commit. Desktop retains canonical evidence and shared execution
infrastructure; the channel-gain Addon is not adopted. Its temporary C: checkout
was archived and removed; retained execution artifacts stay under D:.

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

## Closed K1 normalization exposure custody

Full history of `codex/exp/k1-norm-exposure-100k` through `0567e679` is
reachable from pushed archive commit `c6f0ee4b935069c2299f0a24200f4a525c96e9fa`.
Desktop retains accepted canonical evidence; no rejected normalization model
implementation is integrated. [Terminal custody](experiments/pcqm_k1_norm_exposure_100k/terminal_custody.md)
owns source/evidence/artifact navigation and physical retirement status.

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

Structural-bias/normalization100K complete negative history is reachable from
archive commit `ac7857565c4c84d9d70a56f003082a0ef8c4c290`, including owner tip
`13c623721696c941cd1819d8be162a09029d08ca`. Desktop keeps accepted canonical
evidence only; see [terminal custody](experiments/pcqm_structural_bias_norm_100k/terminal_custody.md).
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
