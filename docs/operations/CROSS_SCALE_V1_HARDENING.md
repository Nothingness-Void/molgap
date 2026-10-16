# Cross-scale V1 hardening

Implementation started on 2026-10-16 JST at Server `f0676fbf`, following the
read-only review retained on the user's Desktop. Remote Server was observed at
`63481324`; neither historical acceptance nor V4/V5 conclusions are rewritten.
The [acceptance](CROSS_SCALE_V1_HARDENING_ACCEPTANCE.md) owns final results.

## Reuse versus gap

| Capability | Reused owner | Narrow change |
|---|---|---|
| Common Spec/hash/atomic publication | `common.spec`, `common.wire`, `common.io` | No parser, identity system or submitter replacement |
| Updates, schedule, sampler, resume, logging | Family Core/Session/Recipe/Loaders/Logger | Unchanged; real Session recovery fixtures remain the authority |
| Observed curves | `cross_scale_analysis.align_curves` | Independent LR/exposure history; per-point reasons; exact input types; checkpoint provenance |
| Cohort acceptance | `cross_scale_inputs.validate_nested_cohorts` | Immutable in-process receipt bound to train/dev manifests, source/order/target/topology/authority/history |
| Terminal outputs | Existing `InspectionSnapshot` / `validate_snapshot` | Recheck immutable output bytes and completed exposure, seed-bound recipe, selected metrics; no status-only completion |
| Reports | Existing analysis owner and immutable IO | Four question kinds, five grades, JSON/Markdown; null missing evidence |
| Backtest | `screen_backtest`, RML validation/grouping/replay/policy registry | Read-only projection; legacy heuristic remains descriptive, early stop NOT_RELEASED |
| Random data roles | Fixed PCQM assets / `PackedGraphDataset` / tensor digests | Indexed role selection and filtered repacking, not graph reconstruction or a second training loader |

## Counterexamples

F1: LR history retains rows without MAE. A step-2 disagreement makes later
points context-only, but preserves a genuinely matched step 1. Each affected
point carries the offending coordinate and reason.

F2: actual optimizer updates and presentations are checked independently of
planned endpoints and complete passes. Batch, accumulation, drop/tail policy,
non-monotonic recovery and double-counted exposure cannot silently match. A
retained tail without observed verification is context, not inferred work.

F3: `nested_cohorts_verified="False"` raises; a bare `True` grants nothing.
`verify_comparison_cohorts` must validate real member arrays before producing
the receipt consumed by the comparator. Remount/reload requires revalidation;
there is no portable boolean certificate or unconditional exemption.

F4: terminal/scientific booleans are exact types. RUNNING, COMPLETE,
STOP_FOR_COST, FAILED and INCONCLUSIVE are explicit execution observations,
not scientific outcomes. Favorable interpretation additionally needs current
owning inspection snapshots for both outputs of every seed, complete trace,
checkpoint/exposure, matching frozen seed and measured endpoint effect.
Missing endpoints remain inconclusive, not architectural negatives.

## Report semantics

`MATCHED_STEP_SCALE` estimates a declared training-member diversity change;
unique member counts may differ, but actual presentations and controlled
conditions must match. `SAME_SCALE_MODEL` compares an architecture change.
`TUNED_RECIPE_SYSTEM` requires symmetric frozen tuning identity/budget and is
not architecture-only causality. `CONTEXT_OR_PORTABILITY` stays context.

Grades are `STRICT_WITH_DECLARED_SCALE_INTERVENTION`, `STRICT_SAME_SCALE`,
`SYSTEM_COMPARISON`, `CONTEXT_ONLY`, `INCOMPARABLE`. These are analysis/report
grades, not new V5 release classes. Missing source/checkpoint/native cost can
downgrade a whole report independently of valid earlier curve points.

100K score never sets a 500K quality veto. All outputs have automatic training
action false. Uncalibrated STQS/early stopping is not connected to control.
Two seeds are exploratory; row bootstrap is not training-seed variance.

## Historical regressions

The graph lifecycle fixture generated declarations in the compatibility shim
without importing `MappingProxyType`; its generated source now imports it.
The GPTrans legacy package inherited the shared-session inventory, including
unselected K1 trainer/addons. Inventory selection removes unselected concrete
mechanisms/trainers but preserves shared update/session/profiling primitives,
then restores declared family/addon roots. Static dependency discovery is not
weakened; an extra regression requires the legacy GPTrans shared dependencies.
The original exclusion assertions are retained. The CLI command-set test now
explicitly includes the already-existing local `check-workflow`; its forbidden
network/dynamic dispatch assertions remain unchanged.

Prior base reproductions are in the Desktop review. This is fixture/inventory
maintenance, not a scientific-contract or training-algorithm change.

## Boundary

Keep `cross_scale_checkpoint.py` as the existing thin CPU adapter. No new model,
optimizer, sampler, loop, checkpoint system, official-role evaluation or GPU
training is introduced. Real dataset acceptance is separately recorded by the
[Unified V2 contract](PCQM_UNIFIED_ROLES_V2.md).
