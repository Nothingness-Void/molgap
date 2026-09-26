# K1 construction adapter and server compatibility

`molgap.k1_adapter` resolves `neural_atom_k1/1` declarations without granting
data access, training, checkpoint loading or replay authority. `k1_metadata()`
is a declaration lookup; `build_k1_model()` calls the selected model factory
with caller-controlled RNG. A supplied `initial_state_path` is rejected.

K1 retains the OGB EdgeState GPS9 local backbone and replaces global attention
with Neural-Atom exchanges. The legacy Spec family binds the `pcqm_k1_full`
recipe identity; it is not an executable matched EdgeState-versus-K1 trainer.
For a new basic EdgeState/depth/K1 construction question, use the separate
[EdgeState family](EDGE_STATE_ADAPTER.md) and freeze the actual training
contract in its owning experiment.

## Server implementation mapping

| Declaration | Factory mode | Server compatibility |
|---|---|---|
| `neural_atom_k1/1`, no addon | `qm9_neural_atom.make_encoder("neural_atom_k1")` | Existing baseline factory |
| `neural_atom_k1/1` + `k1_pair_value/1` | `k1_pair_token.make_encoder("neural_atom_k1_pair_token_value_decoupled")` | Declaration only; unsupported by the retained server factory |

The shared adapter imported from desktop selects the value-decoupled mode,
but server's `src/molgap/k1_pair_token.py` accepts only
`neural_atom_k1_pair_token` and `neural_atom_k1_pair_token_node_return`.
Its mode guard rejects the requested addon before model construction.
The example `examples/k1_v1.json` can pass schema validation and metadata probes
without being constructible in this checkout.

Do not map the addon silently to either existing mode or replace the retained
server implementation: those files own different frozen historical mechanisms.
Enabling value-decoupled execution needs an explicit reviewed implementation
and adapter/source-identity integration, with focused compatibility tests.
This documentation update does not authorize that migration or new training.

## Evidence and execution

Declared base, addon and initialization hashes must be verified against actual
source and state by the owning experiment. Spec/arm identity, empty addon config,
seed and source bindings remain unchanged. A constructed model alone cannot
authenticate the loader, training recipe, role history or checkpoint.

The legacy adapter does not implement resume. The shared
`edge_state_training_core.py` has separate binding/checkpoint primitives for
the EdgeState model-only family; their presence does not extend this legacy
family's trainer. K1 real-shard preflight remains unsupported by the shared
core. Use the [addon guide](EXPERIMENT_ADDON_GUIDE.md) for ownership and
[verification record](shared_experiment_verification.md) for tested scope.
