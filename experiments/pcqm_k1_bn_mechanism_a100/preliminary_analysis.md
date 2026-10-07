# Preliminary interpretation of user-reported Colab output

Historical pre-retrieval interpretation, superseded by the locally verified
[terminal decision](terminal_decision.md) and [paired analysis](analysis.json).

Authority: user_reported_results.json, not yet locally hash-verified artifacts.
Terminal scientific/RML acceptance and paired intervals remain pending retrieval.

| Frozen BN state | Reported MAE(eV) | Original minus case(meV) |
|---|---:|---:|
| Original |0.104904070497|0|
| Dropout off, one pass |0.103658944368|1.245126|
| Dropout off, two passes |0.103658944368|1.245126|
| Dropout on, one pass |0.104506775737|0.397295|
| Dropout on, two passes |0.104568310082|0.335760|

Clean calibration numerically reproduces the prior accepted1.245meV recovery
on the same consumed50K role. With fixed learned parameters, enabling dropout
only during buffer estimation loses0.847831/0.909366meV relative to clean
calibration, supporting a buffer-estimation/inference distribution mismatch in
this frozen-state intervention. This does not show that training dropout harms
learned parameters or justify removing its regularization.

Off1/off2 aggregate MAE are identical at printed precision. On2 is worse than
on1 by0.061534meV. Under reset+cumulative minibatch averaging, duplicated clean
passes are an expected near-null control; this does not reproduce historical
EMA updates on evolving weights, so it cannot exonerate two-forward training
BN chronology. Equal aggregate MAE does not prove byte-identical predictions.
All four mechanism contrasts have absolute point gaps below the predeclared
1meV nomination threshold; no significant-difference or threshold-passing
claim is made without local paired artifact analysis.

Each24-30s case includes calibration and full development inference. It is not
training per-step/epoch speed. Native T4 full-epoch cost and single-vs-double
forward training MAE remain unresolved; do not replace their evidence with this
buffer-only inference diagnostic or with previous bounded A100 scratch timing.

Evidence-ranked next action after acceptance: use the already planned raw and
clean-BN outputs in the submitted consistency500K pair, separating learned
parameter benefit from output-state effects. Single-forward training quality
would need its own matched contract; no successor or full run is launched.
