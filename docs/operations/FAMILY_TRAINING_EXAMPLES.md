# Accelerated family examples

These are construction/preparation examples, not execution authority. Supply the
accepted manifest, frozen initialization, train-only normalization, role and
budget bindings from the owning protocol. No real graphs are needed for the sandbox.

## K1: config and a small addon

```python
from molgap.experiment_execution import build_addon_declaration, build_family_recipe
from molgap.family_training_data import DatasetSelection
from molgap.family_training_recipe import FamilyRecipe, ScheduleSpec
from molgap.fast_sandbox import run_family_sandbox
from molgap.k1_execution_backend import K1ExecutionConfig

family = ("neural_atom_k1", "3")
addon = build_addon_declaration(family, "k1_node_scale", repo_root=repo)
assert run_family_sandbox(family[0], addons=(addon,),
    k1_execution=K1ExecutionConfig())["status"] == "PASSED"
recipe = FamilyRecipe(family[0], "composed", 42, initial_state_sha256,
    DatasetSelection(manifest_path, manifest_sha256, False), train_rows, 0,
    optimizer_steps, learning_rate, weight_decay,
    ScheduleSpec("cosine_step", optimizer_steps), "final-live",
    "python-epoch-shuffle", checkpoint_every, target_mean_eV, target_std_eV,
    addons=(addon,), k1_execution=K1ExecutionConfig(), torch_version="2.7.1")
contract = build_family_recipe(family, addon="k1_node_scale",
    configuration=recipe, train_role_identity=train_role_identity)
```

`SlotScale` is the small example in
[`substrate_scale_addon.py`](../../src/molgap/substrate_scale_addon.py).
For a new idea, register its `AddonContract` and `TrainingAddon.apply_hook`;
the hook attaches a module to `ModelSubstrate` and returns the same model.
No trainer, downloader or platform code is needed. Node, aggregate and readout
examples can be composed in Spec order; frozen replacement modes stay exclusive.

## GPTrans-T: reuse the same preparation route

K1 local pre-norm replacement modes are `k1_prenorm_bn` / `prenorm_bn` and
`k1_prenorm_ada` / `prenorm_ada`. They use the existing statically registered
`FamilyModePolicy.transform_model` hook, not ordered additive slots. The shared
loader strictly installs the common transported full backbone state before the
declared deterministic normalization transformation and optimizer creation. Both declare a
secondary EMA development observation while preserving Live selection; these
extra observations retain explicit trace semantics and do not select a second
model. Their19 local/final norms are an architecture declaration, not adoption.

```python
family = ("gptrans_t", "2")
addon = build_addon_declaration(family, "gptrans_readout_scale", repo_root=repo)
assert run_family_sandbox(family[0], addons=(addon,))["status"] == "PASSED"
recipe = FamilyRecipe(family[0], "composed", 42, initial_state_sha256,
    DatasetSelection(manifest_path, manifest_sha256, False), train_rows, 0,
    optimizer_steps, learning_rate, weight_decay,
    ScheduleSpec("warmup_cosine_epoch", total_passes, warmup_units=warmup_passes),
    "final-ema", "torch-pass-randperm", checkpoint_every,
    target_mean_eV, target_std_eV, ema_decay=.999,
    addons=(addon,), torch_version="2.7.1")
contract = build_family_recipe(family, addon="gptrans_readout_scale",
    configuration=recipe, train_role_identity=train_role_identity)
```

All names above are explicit protocol inputs, not new recommended settings.
The baseline GPTrans-T has no tracked BatchNorm; Clean-BN is rejected for that
composition. For a compatible BN model, see the explicit state policy in the
[core](FAMILY_TRAINING_CORE.md), not an automatic calibration default.

## Sandbox to package to platform

Freeze the contract and Spec, bind `arm.addons` to the same declarations and use
[`prepare_family_training`](FAMILY_TRAINING_CORE.md#public-entry) or the existing
`prepare-family-training` CLI. It writes the neutral source/input package and
release report. Source closure includes the selected addon and registered policy.
The [workflow](EXPERIMENT_WORKFLOW.md) and [Kaggle submission adapter](KAGGLE_WORKFLOW_SUBMISSION.md)
own Kaggle metadata, upload and actual receipts.

IMS reuses [`v4_submit`](../../platforms/ims/v4_submit/README.md). Its stage
entrypoint can be `platforms/ims/run_family_training.py --device 0` followed by
the existing shared worker arguments. Publish neutral sidecars to an authorized
immutable staging path through the IMS skill first; the legacy Git-archive
uploader does not upload arbitrary prepared input bundles. Freeze Torch 2.7.1
explicitly for its declared runtime. Scheduler visibility, actual input transport,
PBS execution, receipt mapping and native qualification require a separate IMS
release; this example submits nothing.
