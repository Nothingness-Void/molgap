# Phase 8 Command Map

This file maps the supported command surface. It does not declare which
experiment is winning or what should run next.

## Local Commands

| Area | Commands | Role |
|---|---|---|
| Data | `data/build_candidate_identity_exclusion.py`, `data/build_multi2d_sealed.py` | Build leakage-safe identities and sealed sets |
| Data | `data/build_graphs.py`, `data/merge_2d_graph_caches.py`, `data/shard_graph_caches.py` | Build and manage graph caches |
| Training | `training/train_encoder.py`, `training/train_distilled_gps.py`, `training/run_schnet_dim_ab.py`, `training/run_schnet_arch_screen.py` | Train an encoder, compress a fixed teacher, or run reproducible SchNet width/compute screens |
| Training | `training/train_dual_gps_2d_head.py`, `training/train_fusion_head.py` | Train 2D and fusion heads |
| Evaluation | `evaluation/eval_dual_gps_2d_ab.py`, `evaluation/eval_dual_gps_route.py` | Run fixed dual-GPS comparisons |
| Evaluation | `evaluation/eval_multi2d_experts.py` | Run aligned multi-expert comparisons |
| Validation | `validation/accept_chunked_candidate_pool.py`, `validation/accept_complementary_candidate_rounds.py` | Reconcile durable acquisition chunks |

Use `.venv\Scripts\python.exe <script> --help` before launching a command.

## Remote Adapters

| Path | Role |
|---|---|
| `remote/kaggle/acquisition/` | Durable candidate acquisition |
| `remote/kaggle/evaluation/` | Fixed external evaluation |
| `remote/kaggle/training/` | Bounded GPU training payloads |
| `remote/scnet/` | Environment, storage, and smoke-test adapters |

Remote durability and lifecycle rules are in `remote/README.md`.

## Archive

- `archive/archive-r01-*` through `archive/archive-r08-*`: closed numbered branches.
- `archive/scaleup/`: completed scale-up drivers.
- `archive/remote/`: completed remote payloads.
- `archive/legacy/`: superseded exploratory commands.

Do not import from `archive/`. Promote reusable behavior into `src/molgap/` and
keep any supported script as a thin CLI.
