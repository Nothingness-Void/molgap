# Experiment Addon Guide

Use the [quickstart](EXPERIMENT_QUICKSTART.md) for the default workflow. This
page owns extension APIs; the [registered workflow](REGISTERED_EXPERIMENT_WORKFLOW.md)
owns preparation, acceptance and recovery contracts.
The [CLI reference](EXPERIMENT_CLI.md) documents the existing local commands;
submission remains with the owning platform adapter.

Before adding plumbing, use the repository's
[experiment reuse skill](../../.agents/skills/molgap-experiment-reuse/SKILL.md).
It routes to existing components and records only the missing behavior; the
contracts below remain authoritative. It does not add a second execution core.

Execution diagnostics reuse the [shared-session profiler](FAMILY_EXECUTION_PROFILE.md).
Training telemetry reuses [standard logging](TRAINING_LOGGING.md), configured by
`build_family_recipe(..., logging=...)`, not an experiment-specific printer/timer.

## New addon checklist

For a slot addon on the shared K1/3 or GPTrans/2 recipe. Each step was run on a
throwaway K1 `post_aggregation` gain on 2026-11-01.

1. **Module.** Add `src/molgap/<name>.py`. The first docstring line is the
   `catalog --compact` purpose. Define `apply_addon(model, config)`: take
   `model._molgap_substrate` or create `ModelSubstrate(model, family)`, then
   `owner.attach(config["name"], module, SlotBinding(family, SlotName.<SLOT>, layer))`
   and return the model. `substrate_scale_addon.py` is the template. Slots
   (`model_substrate.py`): `PRE_MESSAGE` (field `node`), `POST_AGGREGATION`
   (K1 `aggregated`, GPTrans `node_update`), `PRE_READOUT` (`embedding`, layer `None`).
   Start parameters at zero or identity.
2. **Catalog entry.** Add one `(name, "1"): AddonContract(...)` to `ADDONS` in
   `src/molgap/experiment_catalog.py`: `family`; a new `exclusive_group`
   (addons with the same group cannot compose); `source_module`;
   `config_fields` as `AddonConfigField(key, value=...)` (literal), or `kind=
   "integer"` with bounds, or `"initialization_seed"`; `family_versions=("3",)`;
   `execution_modes=(("3", "composed"),)`; `apply_hook="molgap.<name>:apply_addon"`.
   The hook must end in `:apply_addon` and live in `source_module`.
3. **Compiler.** Nothing else to register. `experiment_execution.py` derives the
   adapter from `execution_modes`, and the source inventory adds `source_module`.
   An addon with a `policy_hook` or `extra_source_files` also lists those here.
4. **Local test, no Kaggle** (CPU, `CUDA_VISIBLE_DEVICES=-1`):
   - `python -m molgap.experiment_cli catalog --compact` shows
     `addon <name>/1 (...) runnable=true | registered for compiler family ...`.
   - One synthetic batch, which checks finite output, the addon's own gradients
     and the state roundtrip; expect `PASSED`:
     ```python
     from molgap.experiment_execution import build_addon_declaration
     from molgap.fast_sandbox import run_family_sandbox
     d = build_addon_declaration(("neural_atom_k1", "3"), "<name>", repo_root=".")
     print(run_family_sandbox("neural_atom_k1", addons=(d,))["status"])
     ```
   - After the Spec exists: `experiment_cli sandbox --spec ... --arm candidate --plan ...`
     ([quickstart](EXPERIMENT_QUICKSTART.md#happy-path)).
5. **Do not touch** trainers (`family_training_runner.py`, `k1_screen_training.py`,
   `pcqm_gptrans_v4.py`), `experiment_execution.py`, `experiment_source_inventory.py`,
   other `ADDONS` entries, anything under `experiments/` you do not own, or
   `research_memory/derived/`. Then follow the quickstart from
   "Supply one human input".

## Addon boundary

The family trainer owns construction, loading, optimizer, selection and resume.
For a frozen nondefault base, a registered `policy_hook` supplies
`FamilyModePolicy(construct=..., mount=...)`: construct returns the base;
`mount(recipe, model)` attaches parameters after strict loading and returns the model.
Use the existing substrate; the loader seals it before optimizer/EMA creation.
Slot-only addons retain `composed`; no separate training or packaging path is needed.

Unknown families, addon versions, modes and output profiles fail closed. A Spec
that passes structural validation is not thereby executable, GPU-qualified,
scientifically accepted or authorized. Platform submission and remote state
remain with the applicable platform skill and adapter. The main
[workflow](EXPERIMENT_WORKFLOW.md) owns the supported registry and minimal plan.

## Shared Core

The [architecture owner map](../../ARCHITECTURE.md) locates reusable modules;
the [Common API](../../src/molgap/common/README.md) exposes declaration,
identity and transport helpers. New preparation packages the fixed tracked
runtime distribution automatically. Selected allowlists and AST discovery
remain compatibility/diagnostic APIs, not per-addon setup.

## Server capability boundary

Query `catalog` and `workflow-info --spec SPEC` for capabilities rather than
maintaining another family/mode table here. Registration, random construction
and synthetic sandbox checks do not establish native or real-data qualification.
Historical K1/EdgeState selected-shard preflight retains its loader-only limits;
see [K1 compatibility](K1_ADAPTER.md), [EdgeState](EDGE_STATE_ADAPTER.md) and
[preflight](EXPERIMENT_PREFLIGHT.md). The EdgeState model-only recipe is not an
executable training contract. Verification scope is retained in
[the server review](shared_experiment_verification.md).

An existing immutable reference remains the default. Optional
`prospective.same_run_replay` is for an explicitly justified new paired run,
not a reason to retrain the baseline for every candidate. Merely writing a
terminal descriptor does not make either arm replay-ready.

## Addon Ownership

1. Implement the model delta under `src/molgap/` and add one `AddonContract`
   in `experiment_catalog.py`. Use `AddonConfigField` for frozen literals or
   bounded integers; bind applicable family versions, source module and
   zero-initialization requirements. Spec validation consumes this contract.
2. Put `execution_modes=((family_version, mode), ...)` in that same contract.
   Generic graph/slot addons bind a reviewed `apply_hook`; policy modes use
   their existing `policy_hook`. `experiment_execution.py` derives
   `TrainingAddon` entries; callers do not maintain a second registration.
3. Add focused configuration, forward/gradient, initialization and state checks.
   Declare only genuine extra dependencies in the catalog; tracked runtime
   Python is included by the fixed distribution. Keep composition order and
   parameter ownership explicit before optimizer construction.

`build_addon_declaration(family, addon, repo_root=..., config=..., version="1")`
validates config and computes the source digest. `config=None` supplies defaults
only when every config field is literal; bounded fields require explicit values.
The scientific input uses ordered `{name, version, config}` objects and calls
this builder itself. Direct callers freeze its result in the Spec.

`build_family_recipe(...)` calls the owning recipe builder. Shared K1/3 and
GPTrans/2 use `configuration=FamilyRecipe(...)`; see the
[shared training API](FAMILY_TRAINING_CORE.md). Historical adapters retain their
fixed mode builders and reject unsupported generic overrides. Reference arms
use `addon=None`; ordered composition requires reviewed compatible hooks.
Do not edit copied recipe JSON to manufacture a new contract.

The family owner supplies loader, sampler, training, selection and resume;
new pure-2D PCQM families reuse the [graph extension API](GRAPH_SCREEN_EXTENSION.md)
where applicable. Objectives or input structures need their existing step/policy
or input adapter. Legacy adapters retain their single-addon/mode restrictions.
Static reviewed hooks are not arbitrary caller imports or runtime plugins.
`FamilyExecution.scientific_input_abi` selects an existing compiler/input ABI;
model-interface and feature metadata live in that contract and feed derived
execution views. A different input ABI needs its owning adapter.
Conditional optimizer parameters belong in the addon's catalog metadata;
the shared Session must not add addon names or model-prefix branches.

Platform submission, reconciliation and retrieval stay with the owning workload
skill/adapter. The local CLI has no training or platform-submit command.
Server planning/closure accepts server-owned trajectories; Desktop keeps its
independent jobs, records, recovery and evaluation authority. Shared code does
not transfer custody or native qualification.

## Desktop arm packing

Pack two approved independent questions only when the allocation, runtime,
memory and durable checkpoints support both. Prefer separate visible GPUs;
one-GPU sequential packing requires a combined plan within the job limit.
Retain separate trajectories, RNG/optimizer/state, traces, role use and terminal
decisions plus each arm's assignment and the full observed allocation cost.
Reuse the accepted reference. Idle capacity authorizes no extra candidate,
seed or protected role; a justified single arm records why pairing was infeasible.

## Normal workflow

Follow the [quickstart](EXPERIMENT_QUICKSTART.md). Historical runs retain their
original frozen package and matching recovery APIs. Server integration scope
is recorded in [the integration record](SHARED_TRAINING_SERVER_INTEGRATION.md).
