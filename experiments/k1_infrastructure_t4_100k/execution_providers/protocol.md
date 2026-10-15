# Cached K1 execution protocol — 2026-10-15

The user explicitly authorized one new original-versus-latest K1 dual100K
comparison with profiler after the implementation repair. This authorization
releases one private Kaggle3 T4x2 job, seed42, one isolated worker per device.
It does not release another architecture, extra seed or automatic retry.

## Intervention

`original` uses the retained K1/2 original epoch-loop interface. `optimized`
uses K1/3 and explicitly freezes `K1ExecutionConfig` as recorded in
[authority](authority.json): cached aggregation, cached mean pooling, original
edge projection order, and disabled evaluation memory fill. Shared checkpoint
finite-validation batching and fixed clipping-parameter reuse are included.
The [implementation guide](../../../docs/operations/K1_EXECUTION_BACKENDS.md)
owns code behavior and limitations. Segment reduction, node projection,
compilation, mixed precision and TF32 are excluded from this round.

The [earlier protocol](../protocol.md#matched-scientific-fields) owns unchanged
scientific fields: accepted fixed100K inputs, identical raw initialization,
train0..99999/internal development100000..149999, physical batch128,
40passes/31240updates/3998720presentations, target transform, optimizer,
schedule, clipping and selection. Neither arm reads protected roles.
The original arm is rerun only to measure matched execution in this allocation;
this does not create another scientific baseline or change a model contract.

## Admission, measurement and durability

Both owners must pass source-bound native repeat, profiler/plain and serialized
resume gates before either formal worker starts. The configured candidate also
must pass the exact native-backend two-update state and retained-weight
training-fixture evaluation proof. No approximate acceptance or silent backend
fallback is allowed. A qualification failure terminates this job before formal
training; preserve its evidence and reconcile before any retry.

Both formal recipes skip120 updates and capture16 actual training updates;
shapes, stacks and memory capture remain off. Retain both Chrome traces and
profile reports. Reuse the [measurement and acceptance rules](../protocol.md#measurements-and-acceptance):
parent-clock train-worker wall is primary; capture-off pass spans retain passive
observers. Include setup, native preflight, eval, save and full allocation cost.
Recompute aligned50K endpoint MAEs in float64 and compare predictions/state.
The candidate configuration has not yet established a native speed gain.

Each arm checkpoints atomically at every complete pass and preserves independent
retrievable chunks. The shared all-arm barrier, deadline and output inspection
remain mandatory. The ceiling is7wall hours/14allocated T4 hours including setup,
with60s stop reserve. At the ceiling retain acknowledged checkpoints and stop.
No automatic continuation, protected evaluation, architecture promotion or
Replay eligibility follows from this execution diagnostic.
