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
| `neural_atom_k1/1` + `k1_pair_value/1` | `k1_pair_value.make_encoder("neural_atom_k1_pair_token_value_decoupled")` | Desktop model implementation, copied under a non-conflicting server module name |

The addon constructs K1 first, then inserts one 32-channel ordered-pair token
after layer 6. Pair selection uses channel normalization and masked softmax.
Its value projection starts at identity; its return projection starts at zero.
The implementation contains no optimizer, loader, training recipe or resume.

On 2026-09-27 the complete model file was copied unchanged from
`a47b945fc50a904a3d97949fbd8bdb478a10d73a:src/molgap/k1_pair_token.py` to
`src/molgap/k1_pair_value.py`. Only the adapter's module routing and its copied
test import were adjusted. The donor Git blob is
`6c4801fdf60a933ddec04d06292585d10a6d3bf8`. Desktop had extracted that model
from its `codex/exp/k1-pair-value-100k` source at
`6859c2ffde1b906c3087edf2e1d29bab59b15873`.

Server's `k1_pair_token.py` remains unchanged: it owns the coupled and
node-return historical modes, parameter tables and mechanism checks. Do not
alias those modes to the new implementation. Fresh packages must include the
new module and authenticate actual source bytes; frozen historical packages
and Spec declarations are not silently rewritten. The structural example's
all-zero hashes remain placeholders, not executable authorization.

## Evidence and execution

Declared base, addon and initialization hashes must be verified against actual
source and state by the owning experiment. Spec/arm identity, empty addon config,
seed and source bindings remain unchanged. A constructed model alone cannot
authenticate the loader, training recipe, role history or checkpoint.

The legacy adapter does not implement resume. The shared
`edge_state_training_core.py` has separate binding/checkpoint primitives for
the EdgeState model-only family; their presence does not extend this legacy
family's trainer. K1 preflight supports only a selected train topology shard
from its accepted full manifest; model smoke is unsupported. See
[preflight](EXPERIMENT_PREFLIGHT.md) for the precise family/role boundary.
Use the [addon guide](EXPERIMENT_ADDON_GUIDE.md) for ownership and
[verification record](shared_experiment_verification.md) for tested scope.
