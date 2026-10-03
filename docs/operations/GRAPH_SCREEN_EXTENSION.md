# Shared graph screen extension

This contract covers new pure-2D PCQM direct-Gap families using the accepted
OGB atom9/bond3/RWSE16 cache. Start with the [quickstart](EXPERIMENT_QUICKSTART.md).
Input acceptance is owned by [fixed datasets](../../platforms/README.md);
scientific comparison remains owned by [screening policy](../../experiments/SCREENING_POLICY.md).

## Small model interface

Put the model under `src/molgap/`. Its reviewed factory is
`make_model(arm) -> torch.nn.Module`; `forward(batch)` returns float32 normalized
Gap with shape `(batch.num_graphs,)` or `(batch.num_graphs, 1)`. Parameters and
floating buffers use FP32. The batch supplies categorical `x`, `edge_index`,
`edge_attr`, `random_walk_pe`, graph membership `batch`, direct-Gap `y`, and
source-row identity. The loader rejects privileged geometry.

The factory receives a detached copy of the frozen arm. Model architecture and
constants live in this source; source identity is pinned by `arm.base.sha256`.
The shared owner loads the separately frozen initialization, runs optimization,
selects development weights, persists predictions/checkpoints/traces, and
validates recovery. Do not add a second trainer to the model module.

Register two static entries:

```python
# experiment_spec.py
("my_graph", "1"): FamilyContract(
    "my_graph", "1", "molgap.my_graph", "graph_gap_screen_v1",
    "ogb-atom9-bond3-rwse16-v1", ("train", "development"),
    "seed42-epoch-global-randperm-v1", "train-mean-unbiased-std"),

# experiment_execution.py
("my_graph", "1"): graph_training_adapter(
    ("my_graph", "1"), model_factory="molgap.my_graph:make_model",
    source_files=("src/molgap/my_graph_helpers.py",)),
```

Omit `source_files` when there are no additional imports. It is an explicit
allowlist of reviewed dependencies. The factory module must match the declared
family source. The shared helper selects the trainer, input loader, output
profile, and required common source automatically.

## Small addon interface

An addon exposes `apply_addon(model, config)`. It may modify the model in place
and return `None`, or return a replacement `torch.nn.Module`. It receives a
detached typed config and supplies only the architecture delta.

Add its `AddonContract` in `experiment_spec.py`, then add
`TrainingAddon("my_delta", "1", "my_delta",
apply_hook="molgap.my_delta:apply_addon")` to the family adapter. Use
`build_addon_declaration(...)` to validate config and bind the actual source SHA.
Config fields, supported family versions, and exclusive groups retain Spec
ownership. Ordered, compatible addons are applied in the arm's Spec order;
mutually exclusive or single-addon contracts remain enforced.

`build_family_recipe(..., addon=("first", "second"), ...)` freezes the generic
`composed` mode for multiple reviewed hooks. Legacy family owners keep their
existing mode and stacking rules.

## Freeze one recipe

Use `load_graph_inputs(...)` from `pcqm_graph_inputs.py` with the accepted
manifest's SHA. The loader resolves nested dataset mounts, loads only topology
shards, concatenates each role in manifest order, and verifies source rows,
targets, features, and bytes. It does not rebuild a platform cache or read
geometry files. Each accepted manifest must match a reviewed fixed-data
fingerprint, including the unchanged full manifest when geometry inventory is
also listed.

`build_family_recipe(family, addon=..., source_idx_sha256=...,
target_sha256=..., recipe_config=...)` requires this complete config:

| Field | Source |
|---|---|
| `epochs`, `learning_rate`, `weight_decay` | Owning approved scientific recipe |
| `target_mean_eV`, `target_std_eV` | Accepted train target mean and unbiased standard deviation |
| `train_rows`, `development_rows` | Loaded accepted role sizes |
| `train_source_idx_sha256`, `train_target_sha256` | Loaded train role digests |
| `manifest_sha256` | Exact retained manifest bytes |
| `row_order_fingerprint` | `sampler_order_sha256(train_rows, 1)` |

The separate `source_idx_sha256` and `target_sha256` arguments come from the
accepted development role. These are frozen input identities, not scores.
The builder supplies optimizer, schedule, physical batch, seed, FP32, metric
semantics, and exposure requirements. The recipe validates the same shared
profile before execution. A changed scientific recipe needs the owning
comparison contract; it does not inherit a baseline's comparability.

Build the initial model through `shared_model_adapter.build_model(...)` under
`configure_fp32_determinism(...)`, freeze its CPU tensor state, and bind
`model_state_sha256(...)` in the Spec. The [release gate](EXPERIMENT_CLI.md)
checks the staged initialization and source package. Keep source, recipe,
initialization, prospective records, and accepted cache immutable.

## Reuse the lifecycle

Use the same `prepare-workflow`, owning platform adapter, reconciled receipt,
bounded retrieval, `accept-workflow`, and incomplete-workflow recovery described
in [the registered workflow](REGISTERED_EXPERIMENT_WORKFLOW.md). The selected
worker calls shared preflight/training/recovery hooks; the model module does not
implement those hooks. Kaggle execution requires its isolated declared T4
allocation. CPU synthetic fixtures establish local mechanical behavior.

The owner freezes online pre-update training MAE and post-epoch development
MAE. It binds the selected model and predictions to each acknowledged
checkpoint, and validates their context, fixed rows, metrics, and trace before
resume. Recovery also checks the frozen AdamW settings/moments, cosine
schedule, and runtime-bound RNG device count. Conditional parameters may have
fewer AdamW updates than the global counter; their steps must remain within
acknowledged exposure. Completed prefixes are rejected.

Focused regression owners are `tests/test_shared_family_registration.py`,
`tests/test_graph_screen_training.py`, and `tests/test_shared_graph_lifecycle.py`.
They cover a genuinely new family, tiny model addons, immutable packaging,
training, recovery, and output/terminal integration. Actual remote runtime
calibration and scientific acceptance retain their existing owners.
