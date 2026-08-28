# MolGap PCQM Archive Index

This branch is the durable history line for closed `codex/pcqm-*`
experiments. It is not a live-state or production branch. `master` owns the
production baseline; `molgap-server` and `molgap-desktop` own their respective
active work.

## Archived PCQM questions

| Source branch | Tip | Disposition | Decision and evidence |
|---|---|---|---|
| `codex/pcqm-gap-query-pool-seed42` | `8134db7` | Closed after the seed-42 validation screen | `experiments/pcqm_gap_architecture/results/query_pool_seed42/decision.md`; `platforms/_records/kaggle/training/pcqm_gap100k_query_pool_seed42_v1/` |
| `codex/pcqm-local-operator-acceptance-seed42` | `543e2ca` | Closed after all required local-operator candidates failed the advancement gate | `experiments/pcqm_gap_architecture/results/local_operator_search_seed42/decision.md`; `platforms/_records/kaggle/training/pcqm_gap100k_local_operators_seed42_v1/` |
| `codex/pcqm-recurrent-state-seed42-acceptance` | `285e1dc` | Closed after the recurrent graph-state candidate failed the advancement gate | `experiments/pcqm_gap_architecture/results/recurrent_graph_state_seed42/decision.md`; `platforms/_records/kaggle/training/pcqm_gap100k_recurrent_graph_state_seed42_v1/` |
| `codex/pcqm-sparse-triangle-edge-state` | `2de69cb` | Closed after wedge-cache acceptance and a failed GPU submission; no GPU kernel was created and no retry is authorized | `experiments/pcqm_gap_architecture/results/sparse_triangle_edge_state_seed42/decision.md`; `platforms/_records/kaggle/training/pcqm_gap100k_sparse_triangle_wedge_cache_r2/`; `platforms/_records/kaggle/training/pcqm_gap100k_sparse_triangle_edge_state_seed42/` |

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
