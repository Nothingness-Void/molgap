# Stabilized SwiGLU attribution — 2026-11-06 JST

Scope correction: decision_runtime_inconclusive.md owns the terminal outcome.
The runtime-scope mismatch blocks strict causal comparison. All curve,difference,
stratum and blend descriptions below are NOT_STRICT endpoint diagnostics.

The useful result is complete stable training without a terminal accuracy gain.
All40epochs produced finite development metrics; aligned retained predictions
are finite. This configuration no longer reproduces the observed interrupted
run's evaluation collapse. Source,initialization,data,roles and recipe are bound
by the standard retained artifacts. This does not prove numerical stability on
every possible input,or isolate normalization from the retrained model weights.

The shared saved-artifact analyzer reports early improvement followed by loss of
relative advantage: pass10 -12.606meV,pass20 -7.861meV,pass25 -1.003meV,
pass40 +0.746meV; mean last-five delta+0.662meV. Both complete models select40.
The final interval[-0.326,+1.783]meV covers zero,while clearly failing the required
3meV improvement gate. The supported descriptive failure mode is
early_gain_fades,not an identified universal defect in SwiGLU.

The candidate's final online training MAE is0.0660921 normalized-Gap
(shared analyzer converts it to0.0842734eV). This is an in-pass average under
optimization and dropout,not a fixed-weight/fixed-cohort fit measurement.
The reference has no matched online-training eV observation in this analyzer.
Underfitting,overfitting,insufficient exposure and representation harm therefore
remain insufficient_evidence; do not infer them from this train/dev difference.

Post-hoc slices show compensation: below4eV delta-3.175meV,but6–8eV
+3.854meV; these are development target-derived groups,not deployment routing
rules or held-out specialization. A best constant shift reduces only12.3percent
of the small overall delta,so a global offset is not the dominant explanation.
Saved-prediction50:50 averaging with the old reference yields0.133720579eV;
this is a same-development post-hoc diagnostic with two-model inference cost,
not a promotion or evidence that this package beats accepted Muon0.132520668eV.

Cheapest discriminator already obtained: matched full-horizon saved-prediction
and curve comparison. No material endpoint benefit justifies another training
run here. Keep the stable implementation recoverable with its accepted evidence;
do not stack it onto running Muon work or silently retry with another recipe.

Reproduction inputs are the existing diagnostic
`D:/molgap-artifacts/runs/pcqm-k1-swiglu-prenorm-100k-20261106-s42/v1/reports/diagnostic_21c83bb61ea6d1ddde0f695bdf0894726bfb1552ce78b0939069f48c7f2519e1.json`
and exact-version artifacts under
`D:/molgap-artifacts/retrieved/pcqm-k1-swiglu-prenorm-100k-20261106-s42/v1/version1/`.
No checkpoint inference or extra model execution ran during this attribution.
