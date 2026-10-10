# Experiment Addon Guide

Use the [modular workflow](EXPERIMENT_WORKFLOW.md) for the registered prepare,
platform handoff and acceptance path. This page is the short reuse map; linked
contracts own detailed schemas and evidence rules.

## Pick the Operation

| Operation | Reuse owner | Boundary / next pointer |
|---|---|---|
| Prepare or accept a registered multi-arm run | `experiment_workflow.py`, local [CLI](EXPERIMENT_CLI.md) | The [workflow](EXPERIMENT_WORKFLOW.md) owns the concise plan and lifecycle. |
| Draft a fixed100K K1 pair | [k1_pair_preparation.py](../../src/molgap/k1_pair_preparation.py): `build_inputs` | Reuses fixed data/target bindings and family recipes; fresh questions need fresh IDs, protocol and explicit release. Drafts do not publish RML or authorize training. |
| Train GPTrans | [pcqm_gptrans_v4.py](../../src/molgap/pcqm_gptrans_v4.py): `run_training`; [pcqm_gptrans_full_runner.py](../../src/molgap/pcqm_gptrans_full_runner.py): `train_full` | Fixed V4 screen and full-role recipes are distinct; use their owning contract. |
| Train EdgeState | [edge_state_training_core.py](../../src/molgap/edge_state_training_core.py): `save_checkpoint` / `restore_checkpoint`; [pcqm_official_edge_state.py](../../src/molgap/pcqm_official_edge_state.py): `train_official_edge_state` | The shared core supplies primitives; the official trainer owns its experiment. |
| Change a candidate within a registered family | Family addon/mode hook plus frozen Spec recipe/config | Reuse the registered model, trainer, resume and output owners. See [profiles](EXPERIMENT_FAMILY_WORKFLOW.md). |
| Add a new family | Spec contract, one training adapter, model/trainer, output profile and source inventory | See `experiment_spec.py`, `experiment_execution.py`, and [workflow registration](EXPERIMENT_WORKFLOW.md). |
| Resume or infer from a checkpoint | Family resume loader/model factory; public [inference.py](../../src/molgap/inference.py) | Start at the accepted checkpoint owner and [asset map](../../models/README.md). |
| Clean retained K1 inference | [k1_frozen_inference.py](../../src/molgap/k1_frozen_inference.py): `load_selected_k1`, `predict_clean` | Bind accepted epoch/parameters, normalization, pure-2D role and checkpoint bytes prospectively; [qualification example](../../experiments/pcqm_k1_consistency_fusion_transfer/protocol.md). No training or role authority is provided by the helper. |
| Qualify K1 execution changes without scientific training | [k1_execution_efficiency.py](../../src/molgap/k1_execution_efficiency.py): `probe_k1_execution_efficiency`; [k1_execution_layout.py](../../src/molgap/k1_execution_layout.py): `cpu_layout_context` | Reuse the family factory/step with frozen train-only batches, ordered RNG/state controls and a bounded prospective diagnostic. [CPU limits](../../experiments/pcqm_k1_execution_efficiency/analysis_zh.md); [local RTX5060 qualification](../../experiments/pcqm_k1_execution_efficiency/gpu_attempt_001/terminal_decision.md). Layout is opt-in, not a production default; local equivalence/speed is not native T4 or full-training qualification. |
| Diagnose K1 TF32 precision separately | [k1_precision_diagnostic.py](../../src/molgap/k1_precision_diagnostic.py): `probe_k1_precision`, `prediction_drift`, `qualify_precision_report` | Reuses strict scratch/inference owners, temporarily restores backend flags and checks every case. [Execution review](../../experiments/pcqm_k1_execution_efficiency/EFFICIENCY_REVIEW.md); [TF32 limits](../../experiments/pcqm_k1_execution_efficiency/tf32_attempt_001/terminal_decision.md). TF32 is a different precision contract; frozen train-output drift is not retrained MAE or Tensor Core dispatch qualification. |
| Retain isolated CUDA kernel attribution | [cupti_activity.py](../../src/molgap/cupti_activity.py): `CuptiActivityCollector` | Disposable exclusive Windows-x64 process, verified API28/Kernel9 ABI, synchronized teardown and fail-closed errors/dropped records; no model execution or speed authority in the collector. [K1 dispatch evidence](../../experiments/pcqm_k1_execution_efficiency/kernel_attempt_001/terminal_decision.md) is instrumented attribution, not native throughput, Tensor Core utilization or retrained quality. |
| Copy scalar-rich checkpoint state | [checkpoint_snapshot.py](../../src/molgap/checkpoint_snapshot.py): `checkpoint_to_cpu` | Optional detached CPU tensor clones with exact immutable-scalar fast path; atomic serialization remains with its owner. [CPU fixture qualification](../../experiments/pcqm_k1_execution_efficiency/checkpoint_io_attempt_001/terminal_decision.md) is not native CUDA-to-CPU resume qualification or end-to-end training speed. |
| Reuse epoch sampler checkpoint metadata | [edge_state_training_core.py](../../src/molgap/edge_state_training_core.py): `epoch_permutation_state` | Opt-in two-key scalar cache uses the original permutation/hash owner; fresh cursor dictionaries, unchanged resume verification. [Host-path check](../../experiments/pcqm_k1_execution_efficiency/HOST_OVERHEAD.md) is CPU infrastructure qualification only; no active sender/default change. |
| Publish large machine-readable trace JSON | [training_reproducibility.py](../../src/molgap/training_reproducibility.py): `atomic_json(..., compact=True)` | Optional standard-encoder formatting, unchanged default/atomic owner. [Synthetic sizing](../../experiments/pcqm_k1_execution_efficiency/JSON_PUBLICATION.md) retains all fields and requires receipts to hash actual new bytes; not native training speed or active-payload qualification. |
| Reuse deterministic K1 loader workers | [k1_loader_reuse.py](../../src/molgap/k1_loader_reuse.py): `K1LoaderReuse`; shared [packed reader](../../src/molgap/packed_graph_dataset.py) | Opt-in original list sampler, explicit loader generator and caller-supplied epoch/resume indices; immutable plain lists or exact packed ConcatDataset only. [Packed CPU qualification](../../experiments/pcqm_k1_execution_efficiency/loader_packed_attempt_001/terminal_decision.md) separates cold first-fetch cost; no native GPU pinning, checkpoint replay or stochastic-worker parity. Registered100K's global-RNG loader is not qualified; active payloads unchanged. |
| Measure joint K1 execution contracts | Existing scratch probe's `optimizer_case_flags`, precision/layout contexts; [cuda_execution_closure.py](../../src/molgap/cuda_execution_closure.py): `close_cuda_execution_diagnostic` | Reuses owner steps and V5/RML closure. [Joint qualification](../../experiments/pcqm_k1_execution_efficiency/combined_attempt_001/terminal_decision.md) binds actual source/cases/windows/roles/cost; optimizer/TF32 variants never grant strict execution-only or quality admission. |
| Analyze saved predictions | [analyze_pair.py](../../experiments/pcqm_gptrans_100k_transfer_control/analyze_pair.py): `paired_metrics` | Align identities and roles first; do not run it as prediction-only analysis because its full CLI also performs checkpoint inference. |
| Describe retained K1 trace pairs or raw/clean endpoint errors | [k1_saved_analysis.py](../../src/molgap/k1_saved_analysis.py): `trace_pair`, `bn_rows` | Arithmetic only; use accepted source/role identities and a prospective scope. The K1 local paths/recipes in its prepare wrapper are examples, not authorization or a general launcher. |
| Bind source and local observations | [local CLI](EXPERIMENT_CLI.md) and shared identity/package owners | `run-diagnostic` does not load checkpoints or run inference. |
| Accept artifacts and qualify a comparison | [family output hooks](EXPERIMENT_FAMILY_WORKFLOW.md), [comparison_readiness.py](../../src/molgap/comparison_readiness.py), [full acceptance](../../src/molgap/pcqm_gptrans_full_acceptance.py): `accept` | Mechanical inspection, scientific comparison, runtime qualification and promotion keep their separate gates under the [V5 contract](MOLGAP_COMMON_DIRECTION_V5_FINAL.md). |
| Close and index evidence | [RML entry](../../research_memory/README.md), [lifecycle](../../research_memory/LIFECYCLE.md), [experiment_terminal.py](../../src/molgap/experiment_terminal.py) | Query canonical evidence first; never hand-edit derived indexes. |
| Submit, reconcile or retrieve on a platform | Platform skills and existing adapters | Use `kaggle-molgap-workloads`, `ims-molgap-workloads`, or `scnet-bw-dcu-molgap`; see [platform routing](../../platforms/README.md). |

## Addon boundary

The family trainer owns model construction, loader-facing training, optimizer,
selection and resume behavior. Once its addon is registered, a same-family
experiment supplies only the addon selection, frozen contract/recipe and
config. It does not add an experiment-specific training loop, packager,
terminal writer or platform submitter. A new family registers its model/trainer
adapter, output profile and explicit package dependencies once.

Unknown families, addon versions, modes and output profiles fail closed. A Spec
that passes structural validation is not thereby executable, GPU-qualified,
scientifically accepted or authorized. Platform submission and remote state
remain with the applicable platform skill and adapter. The main
[workflow](EXPERIMENT_WORKFLOW.md) owns the supported registry and minimal plan.

## Shared Core

The local CLI owns identity, preparation and mechanical acceptance; family
trainers own execution; platform skills and adapters own submission and remote
reconciliation. Reuse pointers do not grant scientific, resource, role or
submission authorization.
