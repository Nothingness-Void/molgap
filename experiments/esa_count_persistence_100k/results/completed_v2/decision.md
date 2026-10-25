# Count conditioning retained a smaller late-horizon benefit — 2026-10-25

Exact Kaggle2 kernel140722385/version1 completed both60-pass arms. Four
preflight/train workers returned0. Independent saved-output checks verified
46,860 optimizer updates/5,998,080 sample presentations per arm, identical
50,000 Common-Dev source rows and targets, frozen initial states, FP32/TF32off,
BS128/seed42 and matching optimizer/schedule/selection. The only scientific
intervention was the existing512-parameter count readout. The preceding
infrastructure failure remained separate and was not an architecture negative.

## Frozen scientific decision

| Arm | Selected Common-Dev MAE, eV | Selected pass | Parameters |
|---|---:|---:|---:|
| ESA reference |0.20361478926420212|47|5,015,233|
| ESA count readout |0.2004349772810936|33|5,015,745|

The candidate reduced the saved-prediction endpoint MAE by0.0031798119831085206eV
(about1.56%). It passed this experiment's prospectively frozen **point-gain**
gate of0.003eV. The scientific terminal decision was therefore
`POSITIVE_UNDER_CONTRACT`; the reference was enrolled as an accepted comparator
with `INCONCLUSIVE`, not declared a superior architecture. No threshold was
changed after observing predictions.

The2,000 paired row-bootstrap draws gave gain95% interval
[0.00182156799,0.00451885868]eV. This was uncertainty over retained fixed rows,
not training seeds or model selection. It did not establish a lower gain bound
of0.003eV. Common-Dev had already been reused for discovery and was not an
independent confirmation role. The contract's positive label consequently did
not release another seed,500K, full training or a leaderboard claim.

## Trajectory-level interpretation

The candidate led57/60 aligned evaluation points and all final10 points.
However, its same-step gain declined from0.0109673eV atpass10 and0.0107135 at
pass20 to0.0023285 atpass60; mean final10 gain was0.0023417eV. Final-live
scores were0.2045999/0.2022714eV, distinct from the selected checkpoints above.
Thus the mechanism retained a late benefit but did **not** stop advantage fade.
This60-pass cosine contract was not a continuation of the historical20-pass
cosine schedule, so that earlier scalar gain was not a same-trajectory point.

Both train online errors fell while development scores regressed after their
selected passes. Atpass60 online normalized-Gap MAE was0.010988/0.011169 for
reference/candidate (approximately0.014011/0.014241eV after the frozen scale).
The larger candidate did not obtain a better final online training fit. These
observations argue against a simple insufficient-exposure or parameter-count
explanation and favor investigating generalization/information retention.
Online training and fixed-weight development measurements are not identical
evaluations; their numerical gap is descriptive, not a causal decomposition.

All four prespecified descriptive target ranges improved: <4eV by0.005324,
4–6 by0.003296,6–8 by0.002181 and>=8 by0.006398eV. The formerly regressing
low-Gap target bin did not regress in this run. Only50.726% of individual rows
improved. This was not proof of a size-specific chemical mechanism because
saved predictions lacked independently aligned atom-count/structure metadata.
Target bins cannot become inference-time routing features.

Separate same-role median-bias removal left gain0.003194eV, so a global offset
did not account for the difference. A fixed equal blend was0.1911589eV and the
post-hoc best global weight was near0.53; these were exploratory reused-role
calculations, not accepted new ensemble/router candidates. No additional
training or inference was performed for these arithmetic diagnostics.

## Efficiency, qualification and retention

The complete retry allocation measured6,580.792093s (about1h50m) across two
T4s:13,161.584187 device seconds. Adding the failed Python allocation gave
6,801.672197s (about1h53m), without double-counting overlapping worker/root
ledgers. The candidate's observed process wall was1.30% higher than reference;
this included setup/evaluation, not a paired kernel-throughput benchmark.
Queue, pre-Python provisioning and full billing remained unmeasured.

Hashes, native/output acceptance and actual reference-first canonical closure
belong to [summary](summary.json), [attribution](attribution.json) and
[RML status](rml_status.json). Prospective capability or scheduler completion
alone was not called Replay-Ready. The existing immutable baseline is reusable
only after its own accepted bundle and actual replay membership are verified.
The user's unrelated derived views were preserved; a source-built replay
observation and retained generated snapshot were used at closure.

## Disposition

Retain count readout as a within-ESA positive mechanism, with no automatic scale
or successor. The family remained unqualified for a direct absolute-score rank
against old K1/GPTrans contracts. The evidence did not justify extending training
past60 passes or reopening PPGT conditioning. A later authorized question would
need to target the remaining information/generalization limitation rather than
stacking this result with unrelated marginal modules. Historical weak or
infrastructure-only records were not rewritten.
