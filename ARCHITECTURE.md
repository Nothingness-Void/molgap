# Architecture

This file answers one question: **to change behavior X, which code owns it?**
For live model recommendations and job status read [CURRENT_STATE.md](CURRENT_STATE.md);
for tracks read [TRACKS.md](TRACKS.md); for naming read [NAMING.md](NAMING.md);
for workflow order read [LIFECYCLE_MAP.md](docs/operations/LIFECYCLE_MAP.md) and
[EXPERIMENT_QUICKSTART.md](docs/operations/EXPERIMENT_QUICKSTART.md).

## Core Boundaries

- Reusable code lives in `src/molgap/`; `production/*/scripts/`, `experiments/*/`, and `platforms/local/` wrappers stay thin.
- Evidence lives in `experiments/` and `platforms/_records/` and never supplies runtime logic (never treat `experiments/_closed/` scripts as active CLI behavior).
- Model assets live in `models/` (authoritative registry in `src/molgap/constants.py`); archived code lives on `archive` branch.
- Active CLIs resolve paths from `constants.py` (`ACQUIRE_DIR`..`DATABASE_DIR`), never `Path(__file__).parents[n]` (`tests/test_repository_layout.py`).

## Active Core Ownership

### Local CLI, Launchers, Platform Adapters & Dashboards
| Path | Owns |
|---|---|
| `platforms/local/run.py` | Selected-checkout launcher (uses worktree `src/` with primary `.venv`) |
| `platforms/local/accept_run.py` | Local acceptance & `--closure-input` wrapper over `accept-workflow` |
| `platforms/local/infra_budget.py` | Standalone infra size check against `docs/operations/INFRA_BUDGET.json` |
| `platforms/local/agent_infra_bench.py` | Full 12-task agent infra A/B benchmark & `IMPROVED` gate (`--base` vs `--candidate`) |
| `src/molgap/experiment_cli.py`, `src/molgap/experiment_input.py`, `src/molgap/experiment_run.py`, `src/molgap/local_runtime.py` | Unified local CLI ([EXPERIMENT_CLI.md](docs/operations/EXPERIMENT_CLI.md)), human-input freeze/compile, frozen-run lifecycle |
| `src/molgap/experiment_workflow.py`, `src/molgap/kaggle_workflow.py`, `src/molgap/kaggle_submission.py` | Platform-neutral preparation, Kaggle release binding, journaled submission state machine (`PHASE_NEXT`, `SUBMIT_UNKNOWN`) |
| `platforms/kaggle/submit_workflow.py`, `platforms/kaggle/retrieve_family_outputs.py` | Kaggle submission & exact-version selective retrieval (`--all-arms`, `--strict`, `--paused`, role-history retention) |
| `platforms/ims/check_queue.py`, `platforms/ims/dashboard/en.json`, `platforms/kaggle/dashboard.py` | IMS SSH snapshot collector / stale-snapshot HTTP dashboard (`failure_code`, `stale`) & Kaggle read-only quota dashboard |
| `platforms/scnet/check_dcu_environment.py` | SCNet DTK/DCU & PyG/GPS/SchNet smoke checker (`--skip-schnet`, `--allow-cpu` exit & evidence semantics) |

### Model Registry, Active Backbones & Shared Training
| Path | Owns |
|---|---|
| `src/molgap/experiment_catalog.py`, `src/molgap/experiment_execution.py` | Single family/addon declarations, compiler runnable status, worker dispatch |
| `src/molgap/gptrans.py`, `src/molgap/gptrans_bond_local.py`, `src/molgap/pcqm_gptrans_v4.py`, `src/molgap/gptrans_execution_*.py` | GPTrans-T node/pair core, true-bond residual addon, V4 100K reference, accelerated backend |
| `src/molgap/categorical_gps.py`, `src/molgap/pair_gps_2d.py`, `src/molgap/pcqm_gap_architecture.py`, `src/molgap/edge_state_*.py` | K1 categorical GPS, PairGPS, OGB EdgeState models, adapters & training core |
| `src/molgap/family_training_*.py`, `src/molgap/training/`, `src/molgap/data/`, `src/molgap/models/` | Shared K1/GPTrans training loop/state/loaders/policy, packed fixed records, model substrate |
| `src/molgap/training_logging.py`, `src/molgap/model_substrate.py`, `src/molgap/fast_sandbox.py`, `src/molgap/source_dependencies.py` | Frozen training log cadence ([TRAINING_LOGGING.md](docs/operations/TRAINING_LOGGING.md)), addon slots, CPU sandbox, AST source closure |

### Research Memory (RML), Acceptance & V5 Governance
| Path | Owns |
|---|---|
| `src/molgap/research_memory/` | RML compiler & CLI ([RML_INTERFACES.md](docs/operations/RML_INTERFACES.md)): `cli.py`, `schemas.py` (`TRAJECTORY_OUTCOMES`, `FAILURE_MODES`, closure facts), `cards.py`, `saved_analysis.py`, `validate.py` |
| `src/molgap/experiment_terminal_staging.py`, `src/molgap/experiment_run_acceptance.py`, `src/molgap/experiment_attribution.py` | `draft-closure`, `data_open_evidence` / role-audit drafting (`unobserved_earlier_invocation`), strict acceptance & `INCONCLUSIVE`-only runtime-drift closure, attribution |
| `src/molgap/v5_common.py`, `src/molgap/v5_desktop.py`, `src/molgap/comparison_readiness.py`, `src/molgap/screen_policy.py`, `src/molgap/evidence_pointers.py` | V5 evidence envelopes, desktop reconciliation gates, strict comparison / replay-pool eligibility under drift, SHA pointers |
| `src/molgap/common/`, `src/molgap/compat/` | Cross-host identity/IO/path/spec/transport helpers and lazy legacy import aliases |

### Public Inference & Production (Track A)
| Path | Owns |
|---|---|
| `src/molgap/inference.py`, `src/molgap/__init__.py` | Public lazy inference API (`load_repaired_2m_2d`, `load_hybrid`, `load_routed_dual_gps_hybrid`, batch predictors) |
| `src/molgap/constants.py` | Repository paths, stage roots, hyperparameters, authoritative model registry |
| `src/molgap/graphs.py`, `src/molgap/gps.py`, `src/molgap/schnet.py`, `src/molgap/gine.py`, `src/molgap/egnn.py`, `src/molgap/fusion.py`, `src/molgap/multi2d.py` | SMILES-to-2D/3D PyG graphs (ETKDG), core encoders, `FusionHead`, multi-2D ensembles |

> Note: Recommended registry keys live only in `CURRENT_STATE.md`; `artifact_retained: False` entries are provenance-only and cannot be loaded.

## Tree & Asset Map

| Path | Owns | Entry point |
|---|---|---|
| `production/` | Shipping pipeline in data-flow order (`01_acquire`..`07_database`) | `production/README.md` |
| `experiments/` | One directory per open or closed scientific question | `experiments/README.md` |
| `platforms/` | Compute adapters, remote boundaries, dashboards, records | `platforms/README.md` |
| `research_memory/` | RML schemas, lifecycle docs, generated derived indexes (`families.md`) | `research_memory/README.md` |
| `data/raw/`, `data/cache/`, `models/README.md` | Raw source tables, regenerable caches, checkpoint asset map | `models/README.md` |

## Specialized & Legacy Modules (`src/molgap/`)

- **PCQM data & Route B:** `pcqm_fixed_datasets.py`, `pcqm_official_edge_state.py`, `pcqm_expert.py`, `pcqm_route_b*.py`, `pcqm_graph_inputs.py`, `pcqm_k1_gptrans_fusion.py` (kept for canonical RML bindings; Sep-2026 full runners live on `origin/archive`).
- **QM9 & conformer screens:** `qm9_screen.py`, `qm9_conformer.py`, `qm9_payloads.py`, `qm9_fusion.py`, `conformer_ab.py`, `pubchemqc.py`, `pubchemqc_architecture.py`, `geometry_features.py`, `structural_encoding.py`, `etkdg_array.py`, `portable_radius.py`.
- **Fusion, routing, data repair & reporting:** `hierarchical_fusion.py`, `hierarchical_external_eval.py`, `hierarchical_oracle.py`, `conservative_fusion_*.py`, `route_b_fusion.py`, `multi2d_router_fusion.py`, `router.py`, `router_sampling.py`, `oof_planning.py`, `ensemble_evaluation.py`, `data_repair.py`, `distillation.py`, `retention.py`, `gap_specialization.py`, `artifact_acceptance.py`, `residual_attribution.py`, `model_reporting.py`, `inference_benchmark.py`, `experiment_db.py`, `utils.py`, `tensornet.py`.
- **V4 & packaging internals:** `v4_runtime.py`, `v4_bundle.py`, `v4_submission.py`, `v4_cli.py`, `v4_audit.py`, `experiment_package.py`, `experiment_preflight.py`, `experiment_launch.py`, `experiment_prospective.py`, `experiment_terminal.py`, `experiment_family_workflow.py`, `experiment_family_artifacts.py`, `experiment_source_inventory.py`, `kaggle_pair_runtime.py`, `training_reproducibility.py`, `execution_comparison.py`, `repaired_2m_*.py`.
