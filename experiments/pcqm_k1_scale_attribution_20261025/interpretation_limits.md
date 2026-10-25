# Post-closure interpretation review — 2026-10-25

This review corrects the scope of language in the frozen
[attempt002 decision](attempt_002/terminal_decision.md) and
[attribution](attempt_002/attribution.md). Their numerical observations, hashes,
NO_TRAIN disposition and original bytes remain unchanged.

“Buffer-only explanation fails” means the gap persists under the predeclared
16,384-member, dropout-disabled, 128-batch TRAIN recalibration. This procedure
is not an oracle over every possible BN estimator or calibration population.
It establishes that this specific remedy does not recover the retained tail;
it cannot rule out every buffer-based contribution.

“Remaining learned function difference” refers to the remaining difference
between independently calibrated predictors. Weights, activations and their
estimated BN statistics are coupled. The result does not uniquely assign the
remaining regression to weights, optimizer reset, single forward, learning-rate
restart or a training-time BN mechanism.

The selected higher-WD point retains a small0.034752meV advantage after Clean-BN;
its interval crosses zero. “No retained advantage” should be read as no material
or interval-supported advantage, not an exactly zero point difference.

The train/development divergence is measured on fixed cohorts and is sufficient
to reject more of the same tail recipe as an evidence-backed remedy for inadequate
fitting. It is not a universal K1 capacity limit or proof that every longer
schedule is harmful. Calibration and TRAIN evaluation share members; development
has influenced historical selection; row bootstrap excludes seed variability.

The scale synthesis also corrects the cross-cohort PairToken interval to an
independent-row bootstrap and a signed gain change of -2.627 meV. Its original
authority is `f426350ad636d32ea05c059664cea35fa44f1618`,
`experiments/pcqm_k1_cross_scale_frozen/decision.md`, lines19–20. The cohorts
are disjoint; only contrasts on the same500K rows can use paired resampling.

No model promotion, new gate or scientific-record rewrite follows this review.
