# Architecture module index

This is the conditional-read module map for [`ARCHITECTURE.md`](../../ARCHITECTURE.md).
It is grouped by responsibility so the root architecture document remains a
stable routing page. Edit the owner named here; keep scripts thin and do not
move reusable behavior into experiment result directories.

## Family models and runtime adapters

| Module | Owns | Edit when |
|---|---|---|
| `k1_edge_memory.py`, `k1_edge_slot_interaction.py`, `k1_edge_kaggle_runtime.py` | Real-bond storage/read separation, slot interactions, and isolated K1 Kaggle execution | Testing normalized edge reads without modifying frozen backbone tensors |
| `gptrans.py` | Compact GPTrans-T node/all-pairs propagation core | Changing node-to-node, node-to-pair, or pair-to-node behavior |
| `gptrans_variants.py`, `gptrans_variant_checks.py` | Parameter-free relation-flow hypotheses and remote-only model checks | Changing GPTrans candidates without modifying the frozen core |
| `gptrans_memory.py`, `gptrans_kaggle_runtime.py` | Persistent pair readback variants and isolated Kaggle staging | Testing relation memory without changing frozen optimizer/data |
| `pcqm_gptrans_v4.py` | V4-certified fixed-100K GPTrans reference runtime | Changing GPTrans baseline data, optimizer, determinism, recovery, or evidence contracts |
| `k1_pair_value.py` | Model-only desktop value-decoupled PairToken addon | Constructing the shared addon without replacing historical `k1_pair_token.py` |
| `pcqm_gap_architecture.py` | Official OGB categorical Structural GPS and persistent EdgeState Gap-only candidates | Screening bounded pure-2D PCQM architectures |
| `pcqm_expert.py` | PCQM GINE graph contracts, packed scale-up, checkpoints, and artifact acceptance | Continuing or validating the benchmark-only PCQM specialist |
| `pcqm_k1_full_runner.py` | Frozen K1 full-role preflight, exact-step training, atomic resume, and self-contained bundle | Changing the audited K1 desktop handoff contract |
| `pcqm_k1_variants.py`, `pcqm_k1_variants_runner.py` | Isolated K1 information-flow variants and fixed PCQM-100K v4 training contract | Changing bounded K1 architecture screens |
| `pcqm_k1_pair_selection_diagnostic.py`, `pcqm_k1_pair_selection_acceptance.py` | Frozen PairToken assignment statistics and independent no-inference artifact acceptance | Auditing relation attention concentration without training |
| `edge_state_model_only_v1.py`, `edge_state_training_core.py` | EdgeState/depth/K1 construction and reusable target, sampler, step, evaluation, and resume primitives | Integrating a compatible experiment-owned trainer |
| `experiment_execution.py: graph_training_adapter`, `shared_model_adapter.py` | Generic pure-2D graph model factory and normalized Gap model adapter | Adding a new graph family through the static factory boundary |
| `graph_screen_training.py`, `pcqm_graph_inputs.py` | Reusable graph-screen trainer and PCQM graph input contract | Extending a supported pure-2D graph screen without copying a trainer |
| `documentation_check.py` | Read-only active document pointers, anchors, and entrypoint budgets | Checking navigation after a module or document change |

## Core model and geometry primitives

| Module | Owns | Edit when |
|---|---|---|
| `constants.py` | Repository paths, hyperparameters, and model registry | Adding or retargeting an explicit registry entry |
| `graphs.py` | SMILES-to-2D/3D PyG graphs and ETKDG construction | Changing graph or conformer representation |
| `gine.py` | `GINEWrapper` local message passing | Changing the reusable GINE encoder |
| `egnn.py` | Lightweight equivariant 3D encoder | Testing a low-compute SchNet alternative |
| `gps.py` | `GPSWrapper` and 2D encoding | Changing the 2D encoder |
| `pair_gps_2d.py` | Persistent-pair GPS candidates and bounded repairs | Changing PairGPS node/pair exchange or path/triplet updates |
| `structural_encoding.py` | Resumable random-walk positional-encoding caches | Changing RWSE construction or cache contracts |
| `schnet.py` | `SchNetWrapper` and 3D encoding | Changing the PyG SchNet encoder |
| `geometry_features.py` | Low-cost local angle and dihedral features for 3D encoders | Changing the geometric feature set |
| `schnetpack.py` | Optional SchNetPack 2.x batching/regression | Changing the alternate DCU-portable 3D path |
| `portable_radius.py` | Vectorized PyTorch batched radius graph | Running SchNet where `torch_cluster` is ABI-incompatible |
| `etkdg_array.py` | Framework-neutral ETKDG shard construction for CPU clusters | Building conformer shards without PyG |
| `ogb_features.py`, `v4_runtime.py` | OGB categorical features and portable source/state/runtime helpers | Changing feature encoding or low-level compatibility |

## QM9, PCQM, and fusion research owners

| Module | Owns | Edit when |
|---|---|---|
| `qm9_screen.py` | Fixed QM9 splits, graph caches, encoder training, and embedding export | Changing the architecture-screen protocol |
| `qm9_data.py` | Lightweight QM9 acquisition and deterministic split primitives | Changing QM9 source or split behavior |
| `qm9_local_hierarchy.py` | Track C OGB/RWSE cache, local hierarchy pretraining, matched Gap controls, and DCU checkpoints | Changing the bounded QM9 local-supervision question |
| `qm9_conformer.py` | Paired-conformer QM9 training and evaluation | Changing conformer-robust training experiments |
| `qm9_payloads.py`, `qm9_fusion.py` | Cached embedding alignment, view combination, gates, residual heads, and routing | Changing architecture-screen payload or QM9 fusion behavior |
| `conformer_ab.py` | Paired ETKDG/MMFF timing and bounded-fusion evaluation | Comparing conformer cost against accuracy |
| `pcqm_route_b.py`, `pcqm_route_b_training.py`, `pcqm_route_b_search.py` | Aligned PCQM caches, shard-streamed Gap continuation, nested search, checkpointing, and embedding export | Preparing or changing the Track B encoder protocol |
| `pcqm_route_b_acceptance.py`, `pcqm_route_b_evaluation.py` | Strict encoder-output acceptance and fixed official-valid replay/evaluation | Changing Track B acceptance or evaluation |
| `pcqm_route_b_fusion.py`, `route_b_fusion.py` | Bounded multi-expert fusion and identity A/B | Changing Track B frozen-embedding fusion |
| `ensemble_evaluation.py` | Identity-aligned equal-seed evaluation | Changing multi-seed accuracy evidence |
| `oof_planning.py` | Immutable scaffold folds and OOF prediction contracts | Changing Router-label preparation |
| `hierarchical_fusion.py`, `hierarchical_external_eval.py` | Frozen 2D plus bounded dual-SchNet correction and same-molecule external evaluation | Changing staged Fusion behavior |
| `conservative_fusion_payload.py`, `conservative_fusion_runner.py` | Aligned frozen-2D/dual-SchNet training, resumable execution, and external gate | Changing the Colab precision path |
| `fusion.py`, `hybrid.py` | `FusionHead` and `EndToEndHybrid` | Changing embedding-level or joint 2D/3D fusion |
| `gap_specialization.py` | Gap-only caches, embedding parts, and specialist head training | Changing frozen-embedding Gap specialization |
| `pubchemqc_architecture.py`, `repaired_2m_3d_colab.py`, `repaired_2m_schnet.py` | PubChemQC and repaired-2M 3D training/shards | Changing accepted 3D production research |
| `multi2d.py`, `multi2d_data.py`, `multi2d_router_fusion.py` | Aligned experts, fixed ensembles, accepted pools, scaffold caches, and prediction routing | Changing pure-2D serving or dataset assembly |
| `data_repair.py` | Durable row ledgers, quality flags, identity reconciliation, and fixed-size repair manifests | Repairing a scaled B3LYP corpus without overwriting raw data |
| `distillation.py`, `hierarchical_oracle.py` | Teacher embeddings/student exports and budgeted expert-switch evidence | Changing compression or routing feasibility analysis |
| `router.py`, `router_sampling.py` | Router losses, descriptors, policies, projectors, diverse selection, and scaffold keys | Changing learned routing or sampling |
| `residual_attribution.py`, `model_reporting.py` | Paired residual diagnosis and evidence-backed comparison tables | Changing model comparison reports |
| `pubchemqc.py`, `utils.py` | PubChemQC acquisition/normalization and shared splits, metrics, SMILES, fingerprints, and IO | Changing source acquisition or cross-cutting utilities |

## Lifecycle, evidence, and platform owners

| Module | Owns | Edit when |
|---|---|---|
| `experiment_spec.py` | Frozen family/addon identity, arm bindings, and scientific contract | Adding a reviewed static family or addon declaration |
| `experiment_execution.py` | Static family/addon dispatch, `TrainingAddon`, `TrainingAdapter`, and generic graph factory registration | Binding an owning model/trainer to the shared lifecycle |
| `experiment_training_worker.py` | One isolated worker per arm/phase and explicit resume/preflight flags | Changing subprocess transport, not training science |
| `experiment_family_workflow.py`, `experiment_family_artifacts.py` | Producer events, frozen-recipe output profiles, inspection, and terminal bridge | Adding a compatible family output profile |
| `experiment_training_hooks.py` | Opt-in family screen events, trainer binding, and retained-prefix guards | Connecting an owning trainer without copying its loop |
| `experiment_prospective.py`, `experiment_workflow.py`, `experiment_workflow_resume.py` | Per-arm RML planning, composed preparation, and recovery against an existing immutable release | Changing lifecycle sequencing or incomplete-workflow staging |
| `experiment_package.py`, `experiment_source_inventory.py` | Explicit source allowlist, package identity, reviewed bootstrap dependencies | Changing packaged source selection |
| `experiment_preflight.py`, `experiment_launch_config.py` | Supported preflight and strict launch schema/binding | Changing release input or launch boundary |
| `experiment_launch.py` | Local launch receipts and immutable byte publication | Changing launch binding or atomic no-overwrite publication |
| `experiment_inspection.py` | One immutable output-inspection snapshot | Changing descriptor/closure input inspection |
| `experiment_resume.py` | Hash-bound checkpoint transport and safe restoration; delegates native state checks to family owners | Changing portable incomplete-run recovery |
| `experiment_allocation.py`, `experiment_retention.py` | Physical allocation observation and compact execution retention | Changing allocation ledgers or execution manifests |
| `experiment_terminal.py` | Terminal descriptor translation and explicit closure delegation | Changing terminal binding or translation |
| `training_reproducibility.py` | FP32 determinism, runtime manifests, RNG structure, finite-state, and atomic artifacts | Changing reusable execution provenance |
| `screen_policy.py`, `comparison_readiness.py`, `v5_common.py` | Screen comparability, V5 comparison classes, identity, and evidence-envelope validation | Changing scientific comparison gates |
| `server_control.py`, `server_acceptance.py` | Server-local A/B handoff and V5 fail-closed comparison evidence | Changing server operational semantics or promotion evidence |
| `research_funnel.py`, `evidence_index.py`, `experiment_db.py` | Hypothesis gating, discoverable evidence export, and normalized cross-experiment inventory | Changing evidence navigation or comparison inventory |
| `cost_ledger.py`, `runtime_profiling.py`, `screen_backtest.py` | Hardware-native cost, stage timing, and conservative same-contract screening calibration | Changing cost/profiling/backtest evidence |
| `research_memory/` | Deterministic V5 trajectory, cost, role, reference, READY, completeness, and terminal helpers | Changing committed RML schemas, compiler, replay, or fail-closed rules |
| `kaggle_workflow.py`, `kaggle_pair_runtime.py` | Kaggle staging, assigned-device all-arm barrier, and worker retention | Changing local platform preparation or runtime orchestration |
| `kaggle_accelerator_push.py`, `kaggle_output_retrieval.py` | Kaggle release/receipt adapter and bounded hash-pinned output transport | Changing platform transport or retrieval; keep submit outside shared CLI |
| `platforms/kaggle/retrieve_family_outputs.py` | Thin family/execution manifest selector over shared retrieval | Selecting a supported family output profile |

## Historical and production-only owners

| Module | Owns | Edit when |
|---|---|---|
| `artifact_acceptance.py` | SchNet, repaired-2M primary, and independent secondary 3D artifact gates | Changing remote-output acceptance |
| `inference.py`, `inference_benchmark.py`, `__init__.py` | Model loading, batch prediction, routing, embeddings, UQ, latency, and lazy public exports | Changing prediction or public import behavior |
| `production/history/` | Frozen phase 1-7 reproducibility evidence | Reproducing linked history only; do not extend |
| `tensornet.py`, `visnet.py`, `late_router.py` | Vendored or closed historical implementations | Reproducing the linked archive decision only |

The exact registry and asset paths remain authoritative in
[`src/molgap/constants.py`](../../src/molgap/constants.py). Family contracts,
addon config fields, and registered execution modes are authoritative in
[`experiment_spec.py`](../../src/molgap/experiment_spec.py) and
[`experiment_execution.py`](../../src/molgap/experiment_execution.py), not in
this index.
