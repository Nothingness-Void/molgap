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
