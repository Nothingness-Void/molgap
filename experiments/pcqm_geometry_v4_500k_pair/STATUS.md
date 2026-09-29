# Operational status

The two desktop-owned 500K routes were submitted on 2026-09-28. Kaggle1
generated physical kernel
`nothingnessvoid/molgap-geometry-v4-500k-gptrans-k1-s42`; the
[launch observation](launch/attempt_001.json) binds its frozen identity. On
2026-09-29 the authoritative Kaggle CLI reported `COMPLETE`, and the retrieved
`kernel_status.json` reported `COST_STOPPED` for each arm after five epochs.

Both arm preflights, runtime certificates, partial metadata hashes, exact
19,530-step/2,499,840-presentation traces, and frozen cost-stop calculations
were checked locally. The [decision](decision.md) is `STOP_FOR_COST` for both
arms. Independent RML terminal finalizations are in
`gptrans_distance_only/rml_finalized/` and
`k1_distance_angle/rml_finalized/`. Both partial traces are excluded from the
replay pool. Neither arm has 60-epoch V4 acceptance or a strict paired result.
The remote-listed model, prediction, and checkpoint bytes were not retrieved
or locally validated. No successor or resume was submitted.

This is a dated observation, not live scheduler state.
