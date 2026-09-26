# EdgeState architecture family

`edge_state_gps/1` is the prospective, **model-only** family for the basic OGB
EdgeState Structural GPS. It does not replace the frozen `neural_atom_k1/1`
family or reinterpret any accepted K1 trajectory. Use the old family when
replaying a frozen K1 experiment; use the new family for a new architecture
question whose reference is the basic EdgeState model.

| Arm | Addon | Constructed model |
|---|---|---|
| Basic EdgeState GPS9 | none | Existing `qm9_local_hierarchy.make_encoder()` |
| Depth-only EdgeState | `edge_state_depth/1`, `{"num_layers": N}` | Same OGB EdgeState class, `N` blocks |
| Neural-Atom K1 | `neural_atom_k1/1`, `{}` | Existing `qm9_neural_atom.make_encoder("neural_atom_k1")` |

Depth is an integer in `[1, 16]`, excluding 9: omit the addon for the 9-layer
base. The two addons are mutually exclusive. Width, heads, dropout, target,
pooling, OGB atom9/bond3 inputs, and RWSE16 remain fixed to the inherited
EdgeState base. The `training.overrides` object remains empty. Depth and K1
selection live in the canonical arm's addon list, so changing either changes
the Spec and arm identities. A 9-layer model is constructed through the exact
existing base factory, not a second copy of that model definition. For other
depths, the same underlying OGB EdgeState class uses the base hyperparameters
with only `num_layers` changed.

`edge_state_metadata()` is a lightweight declaration lookup;
`build_edge_state_model()` constructs with caller-controlled RNG and does not
load checkpoint state. `experiment_runner` supports `adapter_probe` and
`construct` for this family. The shared real-shard preflight returns
`UNSUPPORTED_FAMILY_PREFLIGHT`, not a pass. The Spec's
`edge_state_model_only_v1` recipe name marks this construction boundary, **not
an executable optimizer/data/selection contract**. Its sampler, transform,
roles, and placeholder example hashes are declarations requiring independent
verification by the experiment's frozen training addon.

## Shared training primitives

`molgap.edge_state_training_core` provides the reusable model-facing part of a
new EdgeState experiment, including the base, depth-only, and K1 variants:

- `EdgeStateTrainingBinding.from_spec()` binds the canonical Spec/arm to
  caller-verified source, graph, training-contract, runtime, train-row count,
  physical batch size, and target-statistic identities. Digest syntax alone
  does not authenticate those artifacts.
- `construct_bound_model()` checks the seeded initial tensor-state SHA against
  the Spec. Seed and configure the runtime before calling it.
- `training_target_statistics()` computes train-only mean and unbiased sample
  standard deviation from an explicitly complete target vector; the experiment
  supplies its approved row count and minimum standard deviation.
- `EpochPermutationBatchSampler` reproduces the declared `seed+epoch` global
  permutation and drops the incomplete batch. Its state uses the **processed**
  batch cursor, not DataLoader's potentially prefetched position. Checkpoint
  save/restore verifies the permutation hash, cursor, full-batch step count,
  and sample presentations against the bound train rows and batch size.
- `validate_ogb_gap_batch()`, `normalized_gap_step()`, and
  `evaluate_development()` validate OGB atom9/bond3/RWSE16 tensors, perform one
  FP32 normalized Gap L1 optimizer step, and produce source-aligned eV
  development predictions. The caller supplies the approved role's exact
  `source_idx` set; this API does not grant role access.
- `save_checkpoint()` and `restore_checkpoint()` bind model, optimizer,
  scheduler, RNG, sampler state, and an exact cursor to the
  training binding. The returned file SHA must be retained in trusted evidence
  and supplied on restore. Only load trusted checkpoint bytes: PyTorch pickle
  deserialization is not a security boundary.

The experiment addon still owns the authenticated graph loader, role/split and
source-row checks, approval of train-only statistics, optimizer and
scheduler construction, stopping/selection policy, training loop, measured
cost, trace, and V5/RML evidence. A mid-epoch checkpoint records sampler state
but cannot prove that the addon restored the same source-row-to-dataset-index
mapping or worker/prefetch behavior. The shared
real-shard CLI preflight remains unsupported for this family; a validated
model-facing batch is not a real-data acceptance certificate. Kaggle/IMS/SCNet
submission and remote reconciliation belong to their platform skills, not this
module. No new scientific recipe or platform submission command is implied.

The canonical structural example is
[`examples/edge_state_v1.json`](examples/edge_state_v1.json). It has a basic
9-layer reference and a 6-layer candidate. Its zero digests and local platform
are placeholders: `validate-spec` or `run-diagnostic construct` cannot grant
data access or training permission. For a real submission, the experiment
addon must freeze the exact data roles, training budget, optimizer/scheduler,
initial model hash, checkpoint/resume identity, platform package, and runtime
qualification, then use the shared prospective, package, receipt, terminal and
RML gates described in [EXPERIMENT_ADDON_GUIDE.md](EXPERIMENT_ADDON_GUIDE.md).
Do not route the new Spec through the frozen K1 or old official EdgeState
trainer as if either implemented this model-only recipe.
