# Bounded common-cycle screen preparation

Recorded2026-10-14, Asia/Tokyo. The user authorized planning, testing and one
two-arm experiment. This decision records preparation evidence, not a trained
model, completed GPU submission or a positive architecture result.

## Design

The [protocol](protocol.md) selected one local-bond parent and one zero-start
common-chordless-cycle pair-embedding candidate, seed42/fixed100K/physicalBS128/
FP32/EMA.999/60passes. A common-cycle embedding adds96stored parameters, of
which64are active. It preserves global/nonbonded communication and introduces
no teacher, pretrained weights, geometry inputs, alternate database or loss.

An adjacency-only same-capacity control was considered, then dropped before
any submission. Shared execution uses foreachAdamW and bound-foreachEMA; its
result cannot be called an exact match to the old native-loop reference merely
because both use AdamW. The two-arm design reserves one unchanged parent under
the new shared contract. This is not baseline retraining per later candidate.

## Tested capability and its limits

The [planning checks](planning_checks.json) record separate synthetic test
invocations. No real molecular model forward, checkpoint inference or graph
cache was run locally. A frozen-archive synthetic WedgeData roundtrip exercised
the actual packaged serialization dependency in an isolated interpreter.
Neither construction tests nor synthetic gradients qualify a real T4 runtime.
The pre-existing user edit in `tests/test_research_memory.py` was preserved.

The exact accepted100K and500K manifest metadata was downloaded without graph
payloads; SHA256matched their existing acceptance records. These legacy fixed
files use `geometry_shards`; they are not interchangeable with the IMS
topology-only manifest layout. Coordinates and labels may be deserialized
incidentally with these accepted bytes, but cycle derivation reads only atom
count and edges. The encoder transfers no coordinate tensor.

## CPU attempts

- [v1](cpu/failure_v1.md): manifest-layout error before the first graph.
- [recovery v1](cpu/failure_recovery_v1.md): missing existing serialization
  owner while opening the first graph. Both are infrastructure-only, not route
  negatives, with exact receipts, raw logs and native CPU cost retained.
- [recovery v2 receipt](cpu/recovery_v2_submission.json): one CPU-only source-
  bound attempt after an isolated serialization test. A receipt or RUNNING
  observation does not establish complete sidecar acceptance.
  Its later [CPU acceptance](cpu/decision_accepted.md) verified all selected
  rows/bytes and closed the cache question without any training claim.

## GPU release gaps identified before training

1. CPU sidecar derivation passed saved-byte/source-row/role/schema acceptance.
   Reuse that exact manifest and payload identity; do not rebuild platform data.
2. The unchanged parent has no accepted shared-contract reference yet.
   `prepare_family_training` accepts shared interfaces, but the scientific
   `server_acceptance.validate_server_scientific_prelaunch` gate still requires
   an accepted reference bundle; generic `check_acceptance_plan` supports only
   existing-reference screen expectations, not a shared-session same-run
   baseline. Do not fabricate a future accepted baseline or compare oldMAE.
3. A real source/recipe/runtime qualified repeat/resume certificate, native
   memory and whole-budget checks are required. Ten-pass independently
   retrievable checkpoint chunks and the8wall-hour ceiling need explicit
   tested shared-runner support, not only an overwritable last checkpoint.
4. The complete fixed500Kdev50K frozen audit must have its own packaged action,
   matching shared candidate/reference loaders and unchanged100Ktransform.
   The old10Kcohort entry cannot substitute for it. No500Ktraining is authorized.
5. The shared preparation call supplies `execution_source_commit` to
   `plan_prospective`, whose checked-in signature does not accept it. Repair and
   test source/commit binding before invoking that release path.

These gaps were recorded rather than hidden by using a historical native
trainer or relaxing acceptance. The user was asked whether to complete
same-run reference/candidate release support without a third arm, or establish
and accept the new shared reference first. No GPU job, new seed or scientific
success was asserted by this record.
