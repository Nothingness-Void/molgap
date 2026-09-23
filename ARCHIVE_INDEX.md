# MolGap Archive Index

This branch (`archive`) is the durable history line for rejected or inactive
experiments. It is not a live-state or production branch. `master` owns the
production baseline; `molgap-server` owns server-side discovery and
`molgap-desktop` owns full training and official evaluation.

The archive intentionally contains dependency snapshots and source history
needed to reproduce a closed question. Presence here does not mean that a
method is a candidate. Read the desktop branch's `CURRENT_STATE.md` before
using this index for a historical pointer.

## Archived PCQM questions

| Source branch | Tip | Disposition | Decision and evidence |
|---|---|---|---|
| `codex/pcqm-gap-query-pool-seed42` | `8134db7` | Closed after the seed-42 validation screen | `experiments/pcqm_gap_architecture/results/query_pool_seed42/decision.md`; `platforms/_records/kaggle/training/pcqm_gap100k_query_pool_seed42_v1/` |
| `codex/pcqm-local-operator-acceptance-seed42` | `543e2ca` | Closed after all required local-operator candidates failed the advancement gate | `experiments/pcqm_gap_architecture/results/local_operator_search_seed42/decision.md`; `platforms/_records/kaggle/training/pcqm_gap100k_local_operators_seed42_v1/` |
| `codex/pcqm-recurrent-state-seed42-acceptance` | `285e1dc` | Closed after the recurrent graph-state candidate failed the advancement gate | `experiments/pcqm_gap_architecture/results/recurrent_graph_state_seed42/decision.md`; `platforms/_records/kaggle/training/pcqm_gap100k_recurrent_graph_state_seed42_v1/` |
| `codex/pcqm-sparse-triangle-edge-state` | `2de69cb` | Closed after wedge-cache acceptance and a failed GPU submission; no GPU kernel was created and no retry is authorized | `experiments/pcqm_gap_architecture/results/sparse_triangle_edge_state_seed42/decision.md`; `platforms/_records/kaggle/training/pcqm_gap100k_sparse_triangle_wedge_cache_r2/`; `platforms/_records/kaggle/training/pcqm_gap100k_sparse_triangle_edge_state_seed42/` |
| `pcqm sparse torsion EdgeState` | `005a022` | Complete artifact acceptance, but the scientific advancement gate failed; no new seeds or parameter retries | `experiments/pcqm_gap_architecture/results/sparse_torsion_edge_state_seed42/gpu_decision.md`; `experiments/pcqm_gap_architecture/sparse_torsion_edge_state_seed42_protocol.md` |
| `pcqm sparse atom--bond dual stream` | `a659df3` | Complete artifact acceptance, but the candidate regressed and attribution found redundant capacity; mechanism closed | `experiments/pcqm_gap_architecture/results/sparse_atom_bond_dual_stream_seed42/decision.md`; `experiments/pcqm_gap_architecture/post_dual_stream_failure_attribution.md` |

The ring-hierarchy question is not archived: it remains active on
`molgap-server` until its CPU cache and one seed-42 GPU gate reach a terminal
decision.

## Other closed architecture families

These records are retained as historical evidence. They must not be reopened
merely because their source files remain available:

| Family | Closed records |
|---|---|
| QM9 pure-2D follow-ups | `experiments/top20_architecture_qm9/edge_state_jk_readout_r5_decision.md`, `edge_conditioned_r6_decision.md`, `graph_token_r7_decision.md`, `multihop_edge_state_r8_decision.md`, `sparse_path_attention_r9_decision.md`, `directed_edge_state_r10_decision.md` |
| QM9 lightweight repairs | `experiments/top20_architecture_qm9/pair_gps_2d_r2_decision.md` and the R4 fallback record when present |
| Resource-bounded negative branches | `experiments/resource_bounded_architecture/results/gap_rwse_100k_screen/decision.md`, `gated_structural_100k_seed42/`, and `fusion_failure_audit.json` |
| Historical PCQM operator questions | `experiments/pcqm_gap_architecture/results/query_pool_seed42/decision.md`, `local_operator_search_seed42/decision.md`, and `recurrent_graph_state_seed42/` |

The exact metrics, hashes, logs, and acceptance payloads belong to the decision
paths above. Large checkpoints and transient working outputs remain in their
platform record locations rather than being copied into Git.

## Closed source tips preserved on 2026-09-23

The archive merge retains these nine complete source histories as parents without
activating their old root documents or implementations in the archive tree.
Retrieve a record with `git show <tip>:<path>`; its own decision remains the
authority for metrics and contract. Their temporary branch names are not live
work queues.

| Source branch | Tip | Historical disposition | Owning record |
|---|---|---|---|
| `codex/exp/gptrans-conditional-flow-100k` | `575a30af` | Negative; terminal RML retained | `experiments/pcqm_gptrans_conditional_flow_100k/decision.md` |
| `codex/exp/gptrans-flow-ablation-100k` | `76c8fd3e` | Negative propagation ablation | `experiments/pcqm_gptrans_flow_ablation_100k/decision.md` |
| `codex/exp/k1-induced-pair-100k` | `c587be48` | Negative induced-pair screen | `experiments/pcqm_k1_induced_pair_token_100k/decision.md` |
| `codex/exp/k1-mose-lite-slot-100k` | `bd303f72` | Closed without training | `experiments/pcqm_k1_structural_lite_slot_100k/analysis.md` |
| `codex/exp/k1-recurrent-pair-bridge-100k` | `fc131155` | Negative recurrent bridge | `experiments/pcqm_k1_recurrent_pair_bridge_100k/decision.md` |
| `codex/exp/k1-sparse-pair-spse-100k` | `60a77731` | Negative sparse-pair screen | `experiments/pcqm_k1_sparse_pair_100k/decision.md` |
| `codex/exp/k1-stateless-pair-bridge-100k` | `e56f7c6f` | Negative stateless bridge | `experiments/pcqm_k1_stateless_pair_bridge_100k/decision.md` |
| `codex/exp/pcqm-500k-module-attribution` | `8421d1d8` | Completed diagnostic; proposed screens resolved separately | `experiments/pcqm_500k_module_attribution/decision.md` |
| `codex/fix/ims-convergence-runtime` | `e777ebcb` | Historical execution repair; terminal full-run authority remains desktop-owned | `experiments/pcqm_gptrans_full_convergence/results/resource_repair_20260918.md` |

## Desktop terminal source tips preserved by `234f511`

These source tips are reachable from `archive` through merge `234f511`. Their
canonical decisions remain the authority; the archive merge did not activate
their old code or root documents. Selected evidence is also indexed on
`molgap-desktop`.

| Source tip | Historical disposition | Decision |
|---|---|---|
| `0460ba6` (PairNorm 500K) | `NEGATIVE_UNDER_CONTRACT`; the earlier 100K shortlist is a separate decision | `experiments/pcqm_gptrans_pair_norm_500k/decision.md` |
| `818cf7e` (feature denoising) | `INCONCLUSIVE` under source/finalization binding gaps | `experiments/pcqm_gptrans_feature_denoising_100k/decision.md` |
| `2519f4e` (PairValue) | `INCONCLUSIVE` under the strict replay/reference contract | `experiments/pcqm_k1_pair_value_100k/decision.md` |
| `3fc1d80` (Xi'an determinism) | `NEGATIVE_UNDER_CONTRACT` after cross-process gradient replay | `experiments/pcqm_xian_determinism/decision.md` |

The desktop-owned full convergence, Kunshan distance-angle, and PairNorm
100K/profile source histories also remain reachable through this archive
merge; their subsequent Git integrations into `molgap-desktop` own their live
code and records.

## Desktop migration boundary

The desktop migration was prepared from commit `638c9f9` on 2026-09-17. The
following tracked trees remain here at their original paths and were removed
from `molgap-desktop`:

| Archived tree | Contents |
|---|---|
| `production/history/` | Frozen Phase 1-7 narratives, scripts, plots, and metrics |
| `production/03_train/_retired/` | Replaced 300K production records |
| `production/03_train/gps7_schnet_500k_v3/` | Full V3 route record |
| `production/04_evaluate/model_versions_v1_v2_v3/` | Old model comparison record |
| `production/04_evaluate/overview/` | Superseded presentation reports |
| `production/04_evaluate/pcqm_proxy/` | Historical PCQM proxy metrics |
| `production/04_evaluate/validation_data.xlsx` | Frozen validation workbook |
| `production/05_delta_gw/results/` | Historical Delta-learning outputs |
| `production/06_uq/results*/` | Historical UQ and LoRA-UQ outputs |
| `docs/archive/` | Dated narratives and closed design notes |

Large ignored payloads were moved to the corresponding paths in this local
archive checkout, but are intentionally not committed or pushed. Their
compact records and hashes are the durable source; retrieve large payloads
from the recorded remote locator when needed.

## Desktop exceptions

The desktop branch deliberately retains only what the public compatibility
loader still needs:

- `models/compatibility/routed_v4/` contains five routed-v4 checkpoint assets.
- `models/archive/legacy_checkpoints/` and `models/archive/legacy_metrics/`
  contain local-only legacy runtime assets and compact legacy metrics.
- `data/raw/archive/legacy/` and `data/cache/archive/legacy/` contain local-only
  Phase 3/4/6/7 payloads.

These exceptions are not active production candidates. Their paths and roles
are documented in the desktop branch's `models/README.md` and `NAMING.md`.

## V4 and V5 evidence

The current V5 envelopes remain on the desktop branch because they are active
reference pointers, not archived production trees:

| Evidence | Disposition |
|---|---|
| GPTrans-T 100K V4 | Reusable reference, closed for 100K promotion |
| Matched 500K V4 bridge | Nomination evidence; V5 wrapper waits for a second durable K1 copy |
| Full K1/GPTrans-T fusion | Closed without promotion |
| Geometry-transfer 500K | Incompatible nomination; blocked |
| OGB-rich EdgeState | External submission pending review; not a MolGap promotion |
| GINE 1M | Local PCQM specialist; not a leaderboard result |

Use the desktop `models/REFERENCE_INDEX.md` and each experiment's
`v5_evidence.json` for the authoritative V5 wrapper and current disposition.

## Retention policy

- Retire a temporary branch only after its tip is durably reachable from its
  owner branch or `archive`, and any active worktree changes are preserved.
- Do not start new training from this branch. New work belongs to
  `molgap-server` or `molgap-desktop` and must have a roadmap entry.
- A closed experiment is archived once its decision is final and its compact
  evidence is retrievable. Do not create another permanent branch for a seed,
  kernel version, or retry.
- Live state and production identity remain authoritative on the desktop/server
  lines; this index is historical context only.
