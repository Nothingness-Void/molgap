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

## Terminal custody table

Each custody doc owns source restoration, artifact locations, native proofs and
retirement status. Desktop keeps accepted canonical evidence only; no rejected
model or catalog implementation is integrated unless a row says so. Owner refs
marked "retired via index" resolve through the
[retirement index](docs/operations/LOCAL_RETIREMENT_INDEX.json). "Deleted" means
the temporary ref was removed after the commits reached the destination shown.
"Archive" commits are on pushed `origin/archive`.

| Question and outcome | Owner ref | Archive or destination commit | Custody doc and receipt |
|---|---|---|---|
| GPTrans BondLocal plus PairNorm 500K learning-rate pair: accepted, not adopted; lower rate negative; both arms complete strict replay | No local ref | Owner history through `baedffc211490b1d0e805891c04a5628317908b9` in archive `5565121bc2e63acb5eda6b98e6104600cd161ba7` | [Custody](experiments/pcqm_gptrans_bondlocal_pairnorm_lr_500k/terminal_custody.md) |
| K1 retained initialization/forward 500K: accepted, not adopted; four replay entries complete; both pretrained initializations negative | Deleted 2026-10-27 after 235 canonical records were verified | Owner history through `c04ca076` in archive `d01eac748c602aa97764021092ee3cb9096cbd5a` | [Custody](experiments/pcqm_k1_initialization_forward_500k/terminal_custody.md), [qualification](experiments/pcqm_k1_initialization_forward_500k/terminal_acceptance/qualification.md); receipt `D:/molgap-artifacts/retirement/k1-initialization-forward-500k/custody.json` SHA256 `db2956234bbb4ca7eb2b38f4e164fd3ef14a3848e37db60d449d116e1d1fee76`; required continuation prefixes stay on D: |
| K1 scale attribution: accepted NO_TRAIN; model recommendation unchanged | Deleted 2026-10-25 after 25 payloads were verified | Diagnostic source via desktop merge `0593e09e` | [README](experiments/pcqm_k1_scale_attribution_20261025/README.md), [custody manifest](experiments/pcqm_k1_scale_attribution_20261025/artifact_custody.json) |
| K1 tail regularization 500K: negative; both arms strictly replay-ready | Deleted 2026-10-25 | Owner through `cda7921ac72bff68f3996e85335d3343ef1c9f5c` in archive `373e276292ad9af50297638c178a1b0cb8503433` (short `373e2762`) | [Custody](experiments/pcqm_k1_tail_regularization_500k/terminal_custody.md), [import binding 168 records](experiments/pcqm_k1_scale_attribution_20261025/tail_evidence_import.json) |
| K1 hop-fusion qualification: completed NO_TRAIN saved-prediction diagnostic; evidence and thin wrapper integrated, no rejected implementation | Retired via index | Source commit `691bce0c` bound by its prospective record | [Decision](experiments/pcqm_k1_hop_fusion_qualification/terminal_decision.md) |
| K1 equal-content hop-shell 100K: accepted, not adopted; two complete replay entries; local frozen-shell diagnostic done | Retired via index | Owner through `499571f9` in archive `9cca9524` | [Custody](experiments/pcqm_k1_hop_shell_100k/terminal_custody.md), [attribution](experiments/pcqm_k1_hop_shell_100k/analysis_zh.md) |
| K1 Batch Renorm scale versus affine 100K: accepted, not adopted; both arms strict replay; local Clean-BN diagnostic done | Retired via index | History through `c677aa45` in archive `f92e4694` | [Custody](experiments/pcqm_k1_batch_renorm_100k/terminal_custody.md), [attribution](experiments/pcqm_k1_batch_renorm_100k/final_attribution.md) |
| K1 fixed-versus-learned MsgNorm 100K: accepted, not adopted | Retired via index | Owner through `4105cd3d` in archive merge `548082d8` | [Custody](experiments/pcqm_k1_message_norm_100k/terminal_custody.md), [qualification](experiments/pcqm_k1_message_norm_100k/qualification_20261024.md) |
| K1 graph-wise message DropPath 100K: accepted, not adopted; DropPath negative; both arms replay-ready | Deleted after 100 canonical records were verified | Owner through `d54677b5a0e6620f8ee177d75cf275c13757dbe5` in archive `d770ade4563667453bc94176445b2da51673785d` | [Custody](experiments/pcqm_k1_message_droppath_100k/terminal_custody.md), [qualification](experiments/pcqm_k1_message_droppath_100k/qualification.md); receipt `D:/molgap-artifacts/retirement/k1-message-droppath-100k/custody.json` SHA256 `974000e6ec468c4d96dec7f9bb7d4fea48cc22cf53e949138c87c82c749fd458` |
| BondLocal plus pair-update-normalization 100K: positive 100K nomination, no adoption, no 500K/full release; scale/adoption unresolved; both arms replay-ready | Kept: `codex/exp/gptrans-bondlocal-pairnorm-100k` at `3bad3811f876d344373139fa3f70e9263f42697b`; do not merge its model/catalog code | None; checkout retired after 86 canonical records were verified | [Custody](experiments/pcqm_gptrans_bondlocal_pairnorm_100k/terminal_custody.md), [qualification](experiments/pcqm_gptrans_bondlocal_pairnorm_100k/qualification.md); receipt `D:/molgap-artifacts/retirement/gptrans-bondlocal-pairnorm-100k/custody.json` SHA256 `67c442cbfcfbd0e24f0d5f54f630b45d61691fe8658084c644ec45d445eb1ce0` |
| K1 normalization mechanisms: closed without model adoption | Local ref deleted 2026-10-28 (tip `710b8834` is in `origin/archive`; the remote ref remains); checkout retired after verified external custody | Verification `48e4c6a5` (original closure `710b8834`) in archive `aab98f69` | [Qualification](experiments/pcqm_k1_norm_mechanisms_100k/qualification_20261023.md) gives the restore command; restore only when its implementation is needed |
| Single-model RTX 5060 lifecycle exercise: `INFRASTRUCTURE_ONLY`; channel-gain Addon not adopted | No local ref (remote `origin/codex/exp/local-k1-ten-epoch-flow` remains) | Source through `cd8b4b76` in archive `aab98f69` | [Method](experiments/local_k1_ten_epoch_flow/attempt_3/method.md); retained execution artifacts stay on D: |
| K1 execution-speed custody (accepted local) | Local ref `codex/exp/k1-fused-layout-speed-100k` deleted 2026-10-28; its tip `e54108b5` is reachable from `molgap-desktop` and `origin/archive` | Non-promotion route set by its decision | [Decision](experiments/pcqm_k1_fused_layout_speed/terminal_decision.md) |
| K1 normalization exposure 100K: closed | No local ref | Owner through `0567e679` in archive `c6f0ee4b935069c2299f0a24200f4a525c96e9fa` | [Custody](experiments/pcqm_k1_norm_exposure_100k/terminal_custody.md) |
| Structural-bias/normalization 100K: complete negative history | No local ref | Owner tip `13c623721696c941cd1819d8be162a09029d08ca` in archive `ac7857565c4c84d9d70a56f003082a0ef8c4c290` | [Custody](experiments/pcqm_structural_bias_norm_100k/terminal_custody.md) |

## Deleted local refs

On 2026-10-28, 104 finished local refs were deleted: the 95 `codex/retired/*`
recovery refs, six closed infrastructure refs and three closed experiment refs.
Every tip is reachable from `origin/archive` or `molgap-desktop`, and the retirement
index records each retired tip SHA. The external manifest
`D:/molgap-artifacts/retirement/deleted-local-refs-20261028.json` (SHA256
`0c2ecd9dddfd8de67e562d8784ee5f524b748e5be753b4171e3c3c2e04fea716`) lists every
ref with its tip; restore one with `git branch <name> <sha>`. Thirty-one finished
refs deleted earlier (2026-10-27/28) are listed the same way in
`D:/molgap-artifacts/retirement/deleted-local-refs-20261027-earlier.json` (SHA256
`b6339cda893f4c1a36c64c00efdbd2c2e5d88d65e9766ce727b6d634c065fc80`). Open owner refs stay:
the pending-science owners named in CURRENT_STATE, `codex/exp/gptrans-bondlocal-pairnorm-100k`,
and the two active worktree branches.

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
