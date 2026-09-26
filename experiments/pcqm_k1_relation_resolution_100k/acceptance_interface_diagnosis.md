# Acceptance interface diagnosis

On September 27, 2026 the RRWP v1 worker completed all forty epochs. Luna's
first saved-artifact acceptance stopped at `RRWP equation not verified`.
The frozen training source and hash-bound `arm_record.json` emitted two checks:
`rrwp_batched_graphs_match_independent_expected=true` and
`rrwp_isolate_self_transition=true`. The first compares all eight batched walk
powers against the independent small-graph oracle; the second verifies isolated
atoms retain unit self-transition probabilities.

The acceptance consumer erroneously requested the nonexistent field
`rrwp_powers_match_independent_expected`. The correction requires both actual
producer fields explicitly, along with all original receiver, graph-isolation,
permutation, gradient/resume and memory checks. Missing/false/non-boolean
success remains rejected. Original remote source, weights, predictions,
checkpoints, logs and scientific thresholds were not changed or regenerated.

The corrected no-inference acceptance passed for RRWP. This was a local
acceptance-interface defect, not a remote training failure or permission to
rerun training. Regression tests compare producer/consumer field names and
reject each missing/false field and the invented legacy alias. No local model
execution occurred.
