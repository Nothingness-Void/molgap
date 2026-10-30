# MolGap experiment entrypoints

These are routes in the **selected checkout**, not frozen APIs for every model
family. For implementation changes, inspect the selected callable and nearest real
caller/test. Routine command use needs only its documented inputs. Do not
import an experiment-local implementation as a universal model adapter.

## New training or an addon

- For registered Kaggle work, use `submit_workflow.py --experiment` and
  `platforms/local/accept_run.py`. The platform skill supplies commands. Read
  `EXPERIMENT_ADDON_GUIDE.md` only when changing a family/addon; reuse the registry.
  Check current support rather than treating model/Spec support as executable
  training. Family trainers own loading, optimizer, selection and resume.
  Unsupported families/platforms use their existing owners, not copied plumbing.
- `src/molgap/pcqm_gptrans_v4.py` supplies the GPTrans V4 family training and
  preflight entrypoints. `src/molgap/edge_state_training_core.py` supplies
  reusable EdgeState target statistics, sampler, step, evaluation, and
  checkpoint primitives. `src/molgap/v4_runtime.py` supplies shared runtime
  compatibility and state-identity helpers. Use only the relevant family.
- Read that module's owning test, such as `tests/test_pcqm_gptrans_v4.py` or
  `tests/test_experiment_preflight.py`, before extending it.

## Retained predictions or frozen-checkpoint work

- For saved predictions, follow the owning acceptance and row-alignment code.
  `src/molgap/comparison_readiness.py` checks V5 comparison readiness; it does
  not generate new predictions.
- For frozen-checkpoint inference, locate the **accepted checkpoint's**
  matching loader and model factory in its owning experiment or public
  `src/molgap/inference.py` route. `src/molgap/conformer_ab.py` has one specific
  accepted frozen-route example. Confirm checkpoint bytes, architecture,
  transform, role, and source-index semantics before an intervention.
- `python -m molgap.experiment_cli run-diagnostic` and
  `src/molgap/experiment_runner.py:run_experiment` only probe metadata or
  construct a random model. They do not load a frozen checkpoint, perform a
  forward pass, or provide inference authority.

## Execution profiling

- Use the owning profile adapter and its frozen equivalence criteria. The
  training-only example is
  `experiments/pcqm_gptrans_step_profile_v5/run_profile.py` with
  `experiments/pcqm_gptrans_step_profile_v5/test_static.py`.
- `src/molgap/inference_benchmark.py` benchmarks specific public inference
  routes; it is not a generic training profiler. Keep profiling cost and
  scientific comparison evidence separate.

## Identity, package, preflight, and launch binding

- Start at `docs/operations/EXPERIMENT_CLI.md`, then read only the relevant
  detailed `EXPERIMENT_*` contract. Use `python -m molgap.experiment_cli`
  commands `validate-spec`, `plan-prospective`, `package`, `preflight`, and
  `launch-receipt` when their declared capability fits. `plan-prospective`
  publishes RML and may partially publish on failure; inspect its result.
- The normal submitter already calls `experiment_preflight.check_release_inputs`.
  Only a direct low-level adapter needs a separate `check-release` call. Bind each arm's recipe
  to its actual LF-normalized archive bytes. Supply the frozen initialization
  artifact for each pinned state SHA, not a locally constructed model as a
  cross-runtime substitute. Declare loader/trainer imports and inspect selected
  trusted training pickle dependencies; a trainer import alone can miss a
  serialized graph class. This check does not execute a model or qualify a GPU.
  The owning recipe and role contract still determine permitted inputs.
- Library owners are `src/molgap/experiment_spec.py`,
  `experiment_prospective.py`, `experiment_package.py`,
  `experiment_preflight.py`, and `experiment_launch.py`. Package source must be
  an explicit tracked allowlist. The preflight mode must support the family;
  loader-only does not certify model execution. A launch receipt is a local
  observation, not a platform submission.
- For a real remote submit, use the applicable Kaggle, IMS, or SCNet adapter
  and workload skill. Verify the returned run and source identity separately.

## Terminal acceptance and RML

- Start with `platforms/local/accept_run.py` for a run-keyed workflow or
  `platforms/local/train.py --accept-only` for a local single-model run. Read
  `research_memory/LIFECYCLE.md` when changing RML implementation. The adapter validates observed artifacts and exports the accepted
  evidence inputs; RML finalization does not rerun predictions.
- `python -m molgap.experiment_cli terminal` translates a validated descriptor
  read-only by default. `--execute` invokes
  `src/molgap/experiment_terminal.py:execute_terminal_descriptor`, which uses
  `src/molgap/research_memory/terminal_wiring.py:close_terminal_multi_arm`.
  Translation is not an acceptance dry run. Inspect
  `tests/test_experiment_terminal.py` for the binding contract.
- `python -m molgap.research_memory` exposes `recover-trace`,
  `terminal-pipeline`, `validate`, `rebuild`, `check --frozen`, and
  `replay-pool`. `research_memory/LIFECYCLE.md` owns their role, cost, trace,
  minimal-retention, and replay limits. Each arm needs its own accepted
  terminal evidence before claiming two replay-ready entries.
