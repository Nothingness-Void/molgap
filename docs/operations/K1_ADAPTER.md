# K1 model adapter

K1 is a Neural-Atom variant of the basic OGB EdgeState Structural GPS9 model.
The basic model is already constructed by
`molgap.qm9_local_hierarchy.make_encoder()`; K1 replaces its global GPS
attention with local blocks plus Neural-Atom mixers. The family contract
exposes `architecture_base` and adapter metadata exposes
`architecture_base_factory` for this lineage. Neither is a new training or
comparison authority. In the frozen K1 Spec, `reference` still means the K1
reference for the pair-token addon, not the basic EdgeState model. A direct
EdgeState-vs-K1 training comparison needs its own matched, reviewed recipe;
`pcqm_k1_full` hardcodes K1 and must not be relabeled as that recipe.

`molgap.k1_adapter.k1_metadata(spec, arm_id)` resolves declarations without model
imports. `build_k1_model(spec, arm_id)` constructs a model using caller-controlled
RNG. Both accept exactly `ExperimentSpec`, revalidate its canonical JSON and
reconstructed identity, and resolve the exact arm ID. They reject subclasses,
provider objects, snapshots passed as arms, unknown keyword arguments, and
instance-level method injection. The schema remains `molgap-experiment-spec-v1`.

| Family | Addon | Config | Factory and mode |
|---|---|---|---|
| `neural_atom_k1/1` | none | n/a | `molgap.qm9_neural_atom.make_encoder("neural_atom_k1")` |
| `neural_atom_k1/1` | `k1_pair_value/1` | `{}` | `molgap.k1_pair_token.make_encoder("neural_atom_k1_pair_token_value_decoupled")` |

The addon is K1-only, requires its source SHA-256 declaration, and cannot be
stacked or repeated. GPTrans addons and their existing semantics are unchanged.
Baseline arms use empty addons and `addon_semantics: baseline`; candidate arms
use exactly the addon above and `addon_semantics: ordered`. Scientific role is
retained from the spec; the adapter grants no data-role access.

The model wrapper constructs the existing K1 base first and inserts one
32-channel all-ordered-pairs relation token after the sixth local block and its
Neural-Atom mixer. Selection uses per-pair normalization and masked softmax.
The value projection starts at identity, preserving the coupled pair-token
function; the return projection starts at zero, preserving the base output.
Parameter initialization order and state keys follow the source value mode.
No training recipe, loader, dataset identity, or output location is embedded.

## Migration provenance

Extracted from
`D:/文档/molgap-exp/k1-pair-value-100k/src/molgap/k1_pair_token.py`,
source branch `codex/exp/k1-pair-value-100k`, inspected HEAD
`6859c2ffde1b906c3087edf2e1d29bab59b15873`.
The tracked source file had no working-tree modifications at inspection.
These historical path/branch strings are provenance only, never runtime imports
or experiment identities. The extraction retains only the value-decoupled model
path, removes the coupled/node-return mode switches and experiment parameter
tables, and expresses initialization/mechanism checks in synthetic tests.

The source checkout and old entrypoints remain untouched as regression
references. `pcqm_k1_variants.py` and `pcqm_k1_variants_runner.py` were not
migrated. No optimizer, EMA, scheduler, checkpoint, resume, loader, evidence,
protected-role record, platform packaging, or runner semantics were migrated.

## Execution boundary

The existing baseline checkpoint/resume owner is
`molgap.pcqm_k1_full_runner`. Unified adapter checkpoint/resume is **not
implemented**, and that runner is not extended to train or resume this addon.
`initial_state_path=None` is accepted; every non-None path is rejected without
loading. A `frozen_state` declaration remains prospective and does not cause a
state load. A future runner must bind initialization bytes and enforce all
declared recipe/data/source identities before execution.

Metadata includes family/version, mode/addon, arm ID/digest/scientific role,
spec digest, factory/source module, and checkpoint ownership/limitations.
Construction proves neither runtime evidence nor replay eligibility. There is
no replay-ready declaration, dual-arm runner, terminal descriptor, or RML wiring.

Models require the existing Torch/PyG/OGB environment and OGB atom9/bond3 inputs
with RWSE16. Tests use synthetic unequal-size graphs and placeholder digests;
they require no real data or remote credentials. Callers set the RNG seed;
the adapter never seeds or changes global registries.

## Tests for the reviewing agent

Tests were authored but not executed in this implementation task. From the
worktree, using its configured project environment, run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_experiment_spec.py tests/test_k1_adapter.py tests/test_gptrans_adapter.py
```

This worktree had no `.venv\Scripts\python.exe` at handoff. The reviewer must
provide the project virtual environment before running that command; do not
substitute system Python.

Coverage includes baseline state/RNG/parameter/forward equivalence, real addon
dispatch, identity/zero initialization, coupled-value equivalence, pair masking
and mass invariants, source-wrapper integration, strict rejection boundaries,
canonical round trips, and unchanged GPTrans regression tests. Runtime and
numerical behavior remain unverified until the reviewer runs these tests.
