# execution_infra (execution speed, operator profile, T4 topology)

## Quick answer
Recent closed (of 5):
TB-family-session-speed-20261014 | NO_TRAIN | shared-family-execution | reviewed-metric-prefix-layout-transport-distance | null | -
TB-family-topology-t4-100k-k1_mean2-s42 | INCONCLUSIVE | family-topology-t4-execution | shared-mask-index-reuse-and-exact | null | -
TB-family-topology-t4-100k-gptrans_local-s42 | INCONCLUSIVE | family-topology-t4-execution | shared-mask-index-reuse-and-exact | null | -
Open (ACTIVE) 0: -
Reopen: Separately authorized profiler-off same-source equivalence and throughput test / Separately authorized repeated profiler-off same-runtime old/new execution control; no automatic ret / Separately authorized representative data/runtime/platform test
src (src/molgap/): execution_comparison.py, experiment_execution.py, family_execution_diagnostic.py, family_execution_profile.py, family_model_ops.py, family_training_core.py, family_training_data.py, family_training_device_inputs.py, family_training_kernel.py, family_training_loaders.py, +14 more
Experiments: family_100k_operator_profile, family_checkpoint_efficiency, family_session_execution_speed, family_topology_t4_100k, local_k1_ten_epoch_flow

## Closed (5)
TB-family-session-speed-20261014 | NO_TRAIN | shared-family-execution | reviewed-metric-prefix-layout-transport-distance | null | -
TB-family-topology-t4-100k-k1_mean2-s42 | INCONCLUSIVE | family-topology-t4-execution | shared-mask-index-reuse-and-exact | null | -
TB-family-topology-t4-100k-gptrans_local-s42 | INCONCLUSIVE | family-topology-t4-execution | shared-mask-index-reuse-and-exact | null | -
TB-family-100k-operator-profile-k1_mean2-s42 | INCONCLUSIVE | family-100k-operator-profile | observation-only-shared-phase-operator-profiling | null | -
TB-family-100k-operator-profile-gptrans_local-s42 | INCONCLUSIVE | family-100k-operator-profile | observation-only-shared-phase-operator-profiling | null | -

## Open (0)

## Levers (first 10)
family | arm | outcome | effect_meV | failure_mode | mechanism
family-100k-operator-profile | gptrans_local-s42 | INCONCLUSIVE | null | - | observation-only-shared-phase-operator-profiling
family-100k-operator-profile | k1_mean2-s42 | INCONCLUSIVE | null | - | observation-only-shared-phase-operator-profiling
family-topology-t4-execution | family-topology-t4-100k-gptrans_local-s42 | INCONCLUSIVE | null | - | shared-mask-index-reuse-and-exact
family-topology-t4-execution | family-topology-t4-100k-k1_mean2-s42 | INCONCLUSIVE | null | - | shared-mask-index-reuse-and-exact
shared-family-execution | family-session-speed-20261014 | NO_TRAIN | null | - | reviewed-metric-prefix-layout-transport-distance

## Reopen conditions
Separately authorized profiler-off same-source equivalence and throughput test
Separately authorized repeated profiler-off same-runtime old/new execution control; no automatic retry
Separately authorized representative data/runtime/platform test

## src
src/molgap/execution_comparison.py
src/molgap/experiment_execution.py
src/molgap/family_execution_diagnostic.py
src/molgap/family_execution_profile.py
src/molgap/family_model_ops.py
src/molgap/family_training_core.py
src/molgap/family_training_data.py
src/molgap/family_training_device_inputs.py
src/molgap/family_training_kernel.py
src/molgap/family_training_loaders.py
src/molgap/family_training_outputs.py
src/molgap/family_training_policy.py
src/molgap/family_training_recipe.py
src/molgap/family_training_runner.py
src/molgap/family_training_session.py
src/molgap/family_training_state.py
src/molgap/gptrans_execution_backend.py
src/molgap/gptrans_execution_hotpath.py
src/molgap/gptrans_execution_tuning.py
src/molgap/k1_execution_aggregation.py
src/molgap/k1_execution_backend.py
src/molgap/k1_execution_layout.py
src/molgap/k1_execution_profile.py
src/molgap/k1_execution_qualification.py

## Experiments
- experiments/family_100k_operator_profile: scripts prepare.py
  - trace experiments/family_100k_operator_profile/rml/gptrans_local/rml_finalized/trace.json
  - trace experiments/family_100k_operator_profile/rml/k1_mean2/rml_finalized/trace.json
- experiments/family_checkpoint_efficiency: scripts close.py, legacy_oracle.py, prepare.py, run.py
- experiments/family_session_execution_speed: scripts close.py, prepare.py, run.py
- experiments/family_topology_t4_100k: scripts prepare.py
  - trace experiments/family_topology_t4_100k/rml/gptrans_local/rml_finalized/trace.json
  - trace experiments/family_topology_t4_100k/rml/k1_mean2/rml_finalized/trace.json
- experiments/local_k1_ten_epoch_flow: scripts -
  - trace experiments/local_k1_ten_epoch_flow/attempt_3/prospective/rml_finalized/trace.json
