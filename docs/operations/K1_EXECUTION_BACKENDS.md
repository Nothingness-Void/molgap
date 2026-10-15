# K1 execution providers

The [profiler analysis](../../experiments/k1_infrastructure_t4_100k/results/profiler_deep_analysis.md)
identified repeated deterministic aggregation preparation. The execution providers
below reuse the existing model, training session, native gate and platform adapter.
They introduce no new model parameters, addon, optimizer or training loop.

## Select an explicit provider

```python
from dataclasses import replace
from molgap.k1_execution_backend import K1ExecutionConfig

# recipe is the admitted, frozen-science FamilyRecipe under preparation.
recipe = replace(recipe, k1_execution=K1ExecutionConfig())
```

Freeze this through the existing `build_family_recipe` / family preparation flow.
The typed options become part of recipe and source identity. `k1_execution=None`
preserves the old serialized recipe identity and native execution path. Never
retrofit options into an already published package or continuation checkpoint.

| Option | Provider |
|---|---|
| `aggregation="cached_scatter"` | Prepare destination indices once per batch; cache flattened destination/channel offsets across all nine local blocks. Native sorted `index_put_(accumulate=True)` remains the reduction owner. |
| `pooling="cached_scatter"` | Reuse graph membership offsets and CPU-derived exact graph counts; retain sum/divide mean semantics. |
| `projection="edge"` | Preserve original endpoint-gather then projection order. |
| `evaluation_memory_fill=False` | Apply the training memory-fill scope to evaluation; deterministic algorithms remain enabled. |

These are the explicit compatibility-oriented defaults. `aggregation` and
`pooling` also accept `native` or the experimental `segmented` provider;
`projection="node"` projects before gathering repeated endpoints. Neither
experimental option is assumed bitwise-compatible or faster. All options must
pass the same native equivalence gate; there is no tolerance switch or silent
fallback. Whole-model compilation, AMP/TF32 and altered batch/data order are
outside these providers.

## State and lifetime

`k1_execution_aggregation.py` owns the reduction primitive and gradient mapping.
`k1_execution_backend.py` owns typed configuration and instance-local binding;
`family_training_kernel.py` owns its training/evaluation scope. Dense layouts and
local-block/mixer forward identities remain unchanged. All bindings restore on
exceptions and reject nesting. Graphs, edge/dropout order, model state keys and
parameter identities are preserved. Segmented sorting touches only the temporary
aggregation view and preserves duplicates/isolated nodes.

Plans live for one batch, including both mean2 forwards. The cached provider
allocates O(E*C) integer offsets for each used channel width; they are retained
for backward and released with that graph. Segment sorting metadata is lazy and
unused by the cached provider. There is no cache accumulated across training steps.

Shared clipping reuses the fixed model parameter tuple. Checkpoint finite checks
aggregate results per device; one checkpoint-local validation scope reuses only
unchanged, already verified tensor identities/versions. Integer/bool tensors are
inherently finite. Floating/complex state, optimizer moments and buffers remain
checked; failed checks retain diagnostic paths. No cross-step validation cache,
atomic-save weakening or checkpoint format change is introduced.

## Native admission

For configured K1 recipes, the existing all-arm preflight first verifies repeat,
profile/plain when requested, and serialized resume. A same-init/data/RNG native
shadow then compares the complete two-step model/optimizer/schedule/metric/RNG
state exactly, excluding only the intentionally different cursor recipe identity.
Evaluation on the same training fixture is compared at retained two-step weights.
No development or protected role is added by this scratch proof.

The certificate binds the shadow checkpoint, candidate repeat checkpoint,
evaluation outputs and source IDs. Formal workers recheck their SHA, schema,
recipe and exact comparisons before training. A CPU unit test does not qualify
CUDA or establish speed. Native timing must compare admitted configurations with
capture off and identical exposure, then include evaluation and persistence.
`k1_execution_qualification.py` owns this proof; the shared runner delegates to it.

The [2026-10-15 verification](../../experiments/k1_infrastructure_t4_100k/results/implementation_verification_2026-10-15.json)
records synthetic and saved-artifact checks on Torch2.7.1+cpu. No additional GPU speed result was measured.
The earlier100K result remains its own immutable decision.
