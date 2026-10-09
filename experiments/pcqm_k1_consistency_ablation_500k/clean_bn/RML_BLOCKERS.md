# Frozen-reference qualification gap

2026-10-09 inspection. Both original prospective trajectories under kaggle1_v1/
freeze state_at_start.reference_ids=[] and decision_state.active_reference_ids=[].
Both Spec arms declare scientific_role=candidate; neither prospective record
declares a reference comparison binding. The protocol's coefficient0 comparison
is still meaningful numerically, but is not an implemented strict V5 reference
bundle, observed role/trace binding and comparison-readiness acceptance.

The shared terminal pipeline must retain the existing training traces.
terminal_wiring.build_default_trace_manifest derives reference_id="" from these
frozen inputs. schemas.validate_trace_manifest requires a nonempty reference ID;
finalize also rejects a manifest reference absent from frozen reference_ids.
Thus simply choosing an existing evidence ID does not repair the defect.

Do not edit original prospective bytes or invent a historical reference. Do not
drop training traces, bypass terminal_wiring, relabel training as a NO_TRAIN run,
or weaken the validator to manufacture terminal/replay acceptance. Continued
physical Kaggle jobs also preclude same-run replay qualification under the
existing continuation rules. Row bootstrap is not training-seed uncertainty.

Acceptance retains canonical observed-only60-epoch traces through the shared
recovery helper, with exact optimizer steps/presentations/LR/development metrics.
Unobserved checkpoint IDs, EMA metrics and cumulative/device time stay null.
The train field is the optimization objective scaled by training std, not a
clean training inference MAE. Draft manifests and actual schema rejection are
kept beside diagnostic analysis, not published as accepted replay manifests.

Scientific closure, source/artifact custody and honest qualification exclusion
can be recorded now. Original traced RML finalization is blocked, not complete.
Resolving missing-reference storage would require an explicit reviewed RML
representation repair; it must never retrospectively create strict comparison.
This gap does not authorize retraining or a successor.
