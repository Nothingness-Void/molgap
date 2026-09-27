# PCQM geometry component attribution diagnostic

## Question and evidence at planning

The accepted, matched 500K EdgeState pair found a 0.0035180449 eV internal
development gain when distance and angle features were added together. Which
rows account for that gain, and is the retained reference qualified for strict
V5 replay comparison? The owning accepted evidence is
`experiments/pcqm_distance_angle_500k/v5_evidence.json`; its frozen protocol,
local acceptance, artifact reconciliation, and role records remain authoritative.

Alternative explanations include a diffuse small improvement, a gain
concentrated in hard molecules, a geometry-specific benefit, additional model
capacity, and seed or optimization effects. Existing combined-versus-2D
predictions cannot isolate distance from angle or distinguish the latter three.

## Frozen local action

Read only the two SHA-bound `direct_gap_development.pt` files accepted by the
owning experiment. Require exactly 50,000 ordered source indices 500000..549999,
identical finite targets, finite predictions, and exact expected hashes.
Recompute MAE, per-row absolute-error delta, win/tie/loss rates, and signed
residual correlation. Summarize fixed target-gap bins [0,2), [2,4), [4,6),
[6,8), [8,infinity) eV, baseline-absolute-error quartiles, and prediction-
disagreement quartiles. Report row counts and candidate-minus-baseline MAE in
each stratum. These are exploratory descriptions of an already selected
development role, not independent tests or causal feature attribution.

Separately inspect both retained canonical traces for observed optimizer step,
sample presentations, checkpoint identity, and timing fields. Report their
presence or absence without filling them from epoch numbers. Use the original
V5 envelope and reference-reuse index to state only the existing comparison
qualification. No model checkpoint, graph cache, remote scheduler, official
validation, test-dev, or test-challenge role is accessed.

## Decision boundary and cost

The local action is justified by the already accepted paired improvement of at
least 0.001 eV under the original point gate. Its purpose is to decide whether
a separately frozen component experiment could change a model or budget
decision. This diagnostic cannot promote an arm or make the old reference
strictly replay-ready. A later distance-only/angle-only run requires a new
prospective training contract, executable cache and platform preflight,
strictly defined comparator, role/cost budget, and two independently complete
replay records. Do not retrain the accepted 2D or combined arm just to close a
record gap. If these requirements cannot be met, record `NO_TRAIN` and the
specific blockers.

Expected hardware cost for this action is local CPU only. CPU and wall time
will be reported as measured when available; accelerator device and queue time
are not applicable. No new protected-role use is authorized.
