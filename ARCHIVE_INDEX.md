# MolGap Archive Index

This branch (`molgap-archive`) is the durable history line for rejected or
inactive experiments. It is not a live-state or production branch. `master`
owns the production baseline; `molgap-server` owns server-side discovery and
`molgap-desktop` owns full training and official evaluation.

The archive intentionally contains dependency snapshots and the source history
needed to reproduce a closed question. Presence here does not mean that a
method is a candidate. The authoritative live decision remains on
`molgap-server` only for accepted or still-active work; the full decision and
implementation for a closed method live here.

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

These records were already present in the inherited experiment tree and are
kept on this branch as historical evidence. They must not be reopened merely
because their source files remain available:

| Family | Closed records |
|---|---|
| QM9 pure-2D follow-ups | `experiments/top20_architecture_qm9/edge_state_jk_readout_r5_decision.md`, `edge_conditioned_r6_decision.md`, `graph_token_r7_decision.md`, `multihop_edge_state_r8_decision.md`, `sparse_path_attention_r9_decision.md`, `directed_edge_state_r10_decision.md` |
| QM9 lightweight repairs | `experiments/top20_architecture_qm9/pair_gps_2d_r2_decision.md` and the R4 fallback record when present |
| Resource-bounded negative branches | `experiments/resource_bounded_architecture/results/gap_rwse_100k_screen/decision.md`, `gated_structural_100k_seed42/`, and `fusion_failure_audit.json` |
| Historical PCQM operator questions | `experiments/pcqm_gap_architecture/results/query_pool_seed42/decision.md`, `local_operator_search_seed42/decision.md`, and `recurrent_graph_state_seed42/` |

The exact metrics, hashes, logs, and acceptance payloads belong to the
decision paths above. Large checkpoints and transient working outputs remain
in their platform record locations rather than being copied into Git.

The archived commits preserve candidate source, protocols, launch metadata,
terminal logs, metrics, acceptance payloads, and failure evidence where those
items were committed by the source branch. Large checkpoints and transient
working outputs remain in their platform record locations rather than being
copied into Git.

## Retention policy

- Keep the original remote experiment branches until their history is no
  longer needed; this archive does not delete them.
- Do not start new training from this branch. New work belongs to
  `molgap-server` or `molgap-desktop` and must have a roadmap entry.
- A closed PCQM experiment is archived once its decision is final and its
  compact evidence is retrievable. Do not create another permanent branch for
  a seed, kernel version, or retry.
- `CURRENT_STATE.md` and `ROADMAP.md` in the production/server lines remain
  authoritative for live work; this index is the archive entry point.
