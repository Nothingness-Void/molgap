# Experiment Addon Guide

Use the [modular workflow](EXPERIMENT_WORKFLOW.md) for the registered prepare,
platform handoff and acceptance path. This page is the short reuse map; linked
contracts own detailed schemas and evidence rules.

## Pick the Operation

| Operation | Reuse owner | Boundary / next pointer |
|---|---|---|
| Prepare or accept a registered multi-arm run | `experiment_workflow.py`, local [CLI](EXPERIMENT_CLI.md) | The [workflow](EXPERIMENT_WORKFLOW.md) owns the concise plan and lifecycle. |
| Draft a fixed100K K1 pair | [k1_pair_preparation.py](../../src/molgap/k1_pair_preparation.py): `build_inputs` | Reuses fixed data/target bindings and family recipes; fresh questions need fresh IDs, protocol and explicit release. Drafts do not publish RML or authorize training. |
| Local K1 execution-speed diagnostic | [k1_local_speed.py](../../src/molgap/k1_local_speed.py): `run`; `k1_execution_layout.py`, `k1_loader_reuse.py` | Reuses owner model/step/order and pinned CPU TRAIN inputs. [Accepted10ep scope](../../experiments/pcqm_k1_fused_layout_speed/README.md) is not a registered T4/quality trainer or a new training release; defaults remain unchanged. |
| Train GPTrans | [pcqm_gptrans_v4.py](../../src/molgap/pcqm_gptrans_v4.py): `run_training`; [pcqm_gptrans_full_runner.py](../../src/molgap/pcqm_gptrans_full_runner.py): `train_full` | Fixed V4 screen and full-role recipes are distinct; use their owning contract. |
| Train EdgeState | [edge_state_training_core.py](../../src/molgap/edge_state_training_core.py): `save_checkpoint` / `restore_checkpoint`; [pcqm_official_edge_state.py](../../src/molgap/pcqm_official_edge_state.py): `train_official_edge_state` | The shared core supplies primitives; the official trainer owns its experiment. |
| Change a candidate within a registered family | Family addon/mode hook plus frozen Spec recipe/config | Reuse the registered model, trainer, resume and output owners. See [profiles](EXPERIMENT_FAMILY_WORKFLOW.md). |
| Add a new family | Spec contract, one training adapter, model/trainer, output profile and source inventory | See `experiment_spec.py`, `experiment_execution.py`, and [workflow registration](EXPERIMENT_WORKFLOW.md). |
| Resume or infer from a checkpoint | Family resume loader/model factory; public [inference.py](../../src/molgap/inference.py) | Start at the accepted checkpoint owner and [asset map](../../models/README.md). |
| Clean retained K1 inference | [k1_frozen_inference.py](../../src/molgap/k1_frozen_inference.py): `load_selected_k1`, `predict_clean` | Bind accepted epoch/parameters, normalization, pure-2D role and checkpoint bytes prospectively; [qualification example](../../experiments/pcqm_k1_consistency_fusion_transfer/protocol.md). No training or role authority is provided by the helper. |
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
