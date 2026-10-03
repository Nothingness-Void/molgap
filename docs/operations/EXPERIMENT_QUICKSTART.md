# Experiment infrastructure quickstart

This is the shortest route for a new agent. It selects an existing owner,
freezes one scientific contract, and reuses package, launch, recovery,
acceptance, and RML plumbing. It does not authorize compute or submit a job.

## 1. Route the question

Read [`AGENTS.md`](../../AGENTS.md), [`CURRENT_STATE.md`](../../CURRENT_STATE.md),
the relevant [`ROADMAP.md`](../../ROADMAP.md) section, and the owning experiment
decision. Then query the registry before writing code:

```powershell
$python = '.\.venv\Scripts\python.exe'
& $python -m molgap.experiment_cli workflow-info --spec $spec
```

Choose one route:

| Question | Route |
|---|---|
| A variant of a registered family | Add one typed addon and use the existing family trainer/output/recovery hooks. |
| A new pure-2D graph Gap family | Implement the small graph model owner and register the generic graph adapter described below. |
| A different target, geometry, trainer contract, or platform | Use or add the owning adapter; the shared graph screen is not a fallback. |

`workflow-info` reports declaration versus executable support. A declaration,
metadata probe, or synthetic fixture is not real-shard, GPU, scientific, or
platform qualification.

## 2. Add an existing-family addon

Keep the change to the scientific delta:

1. Add an `AddonContract` in `src/molgap/experiment_spec.py` with typed frozen
   configuration, supported family versions, source module, and any exclusive
   group or single-addon rule.
2. Add one or more reviewed `TrainingAddon(name, version, mode, ...)` entries to
   the owning `TrainingAdapter` in `src/molgap/experiment_execution.py`.
   Existing family adapters keep their mode dispatch and declared stacking
   limits; the generic graph adapter may compose approved addons in Spec
   order and additionally binds each
   `apply_hook="molgap.<module>:apply_addon"`. Put only genuinely new source
   dependencies in `extra_source_files`.
3. Implement the model delta under `src/molgap/`, build the declaration with
   `build_addon_declaration(...)`, build the fixed recipe with
   `build_family_recipe(...)` (generic graph recipes also require explicit
   `recipe_config`), and add focused configuration,
   initialization, source-digest, and hook tests.

The family or shared adapter owns input semantics, sampler, optimizer, selection,
resume, output profile, and native checks for its declared scientific contract.
The generic PCQM graph route reuses `pcqm_graph_inputs.py` and
`graph_screen_training.py`; an addon supplies only its model delta. Do not copy
a trainer, source inventory, launcher, receipt, recovery transport, or
terminal/RML closure. See the
[addon guide](EXPERIMENT_ADDON_GUIDE.md) and the [registered workflow](REGISTERED_EXPERIMENT_WORKFLOW.md#add-an-addon).

## 3. Add a new pure-2D graph Gap family

Use the [graph extension contract](GRAPH_SCREEN_EXTENSION.md) for the small
model/addon interface, registration entries, and complete recipe input fields.

The generic graph extension is intentionally narrow. It is available only after
the static registry and focused tests accept all of these contracts:

- `FamilyContract` uses features `ogb-atom9-bond3-rwse16-v1`, roles
  `("train", "development")`, recipe `graph_gap_screen_v1`, sampler
  `seed42-epoch-global-randperm-v1`, and transform `train-mean-unbiased-std`.
- `graph_training_adapter(family, model_factory=..., addons=..., source_files=...)`
  selects `graph_screen_training.py` and `graph-screen-v1`.
- The family module exposes reviewed `make_model(arm) -> torch.nn.Module`; its
  `forward(batch)` returns one finite normalized Gap value per graph.
- An addon exposes `apply_addon(model, config)` and is registered with a typed
  `AddonContract` plus a `TrainingAddon` hook. The hook is a small model delta,
  not a second trainer.
- `pcqm_graph_inputs.py` is the shared accepted PCQM input contract; source
  hashes and frozen initialization remain bound by the shared Spec and launch
  checks.

The generic path currently covers this pure-2D direct-Gap screen only. It does
not claim support for HOMO/LUMO targets, 3D or privileged geometry, arbitrary
conformer rules, arbitrary callbacks/imports, non-PCQM scientific contracts, or
IMS/SCNet/other platform submission. New-family registration remains subject to
family-agent review and actual focused tests; do not describe declaration
support as a qualified family.

## 4. Prepare, submit, and accept

Freeze the Spec, prospective records, recipe, source package, and platform plan
before any remote action:

```powershell
& $python -m molgap.experiment_cli validate-spec --spec $spec
& $python -m molgap.experiment_cli prepare-workflow --spec $spec --repo-root . --plan $plan --output $freshOutput
```

`prepare-workflow` validates every arm, packages the explicit source allowlist,
binds the recipe/initial state/prospective records, freezes the launch config,
and writes release/timing reports. It is preparation, not submission. Use the
combined path once; it publishes prospective records after the static gates.
For other owning adapters, follow their separate planning entry. Use the
selected platform workload skill and adapter to submit and reconcile the exact
returned run/version. The shared CLI has no submitter.

After retrieval, inspect and close through the owning family workflow:

```powershell
& $python -m molgap.experiment_cli accept-workflow --spec $spec --repo-root . --package $package --expected-package-identity $packageIdentity --receipt $receipt --outputs $outputs --locations $locations --execution-root $executionRoot
```

Without `--execute`, this is mechanical verification only. Scientific
acceptance, role/cost review, replay readiness, promotion, and RML closure keep
their existing owners. Read [EXPERIMENT_CLI.md](EXPERIMENT_CLI.md) for exact
fields and [REGISTERED_EXPERIMENT_WORKFLOW.md](REGISTERED_EXPERIMENT_WORKFLOW.md)
for output and retention boundaries.

## 5. Recover an incomplete run

Reconcile the exact original platform run/version first. Recovery requires every
arm to be incomplete and checkpoint-bearing, the original package and ACTIVE
prospective bytes, and the registered family/native runtime. Build one fresh
bundle per arm, then prepare a fresh recovery directory:

```powershell
& $python -m molgap.experiment_cli build-resume --spec $spec --repo-root . --prepared $originalPrepared --package $package --expected-package-identity $packageIdentity --receipt $receipt --arm $armId --source-output $retainedArm --output $freshArmBundle
& $python -m molgap.experiment_cli prepare-resume --spec $spec --repo-root . --prepared $originalPrepared --package $package --expected-package-identity $packageIdentity --receipt $receipt --resume-plan $resumePlan --output $freshRecovery
```

Completed arms, missing checkpoints, mixed completed/incomplete sets, changed
prospective bytes, unsupported packaged runtimes, and extra pickled input
staging are rejected or remain with their owning adapter. Recovery reuses the
original scientific contract and does not create a new prospective action.

## 6. Verify the local infrastructure

Use [infrastructure verification](INFRASTRUCTURE_VERIFICATION.md) for the
repeatable local lifecycle, documentation, and RML checks and their scope.
Keep experiment evidence in its decision rather than expanding this entry.
