# Desktop Handoff Acceptance (2026-09-06)

The desktop verified the server's three-seed confirmation and accepted
GraphState9 as the full-data candidate. This replaced the earlier two-arm
scratch proposal before either arm trained. The existing full numerical audit
remained valid; see `../pcqm_geometry_scratch_control/audit_acceptance.json`.

The server's exact metrics and architecture decision are preserved in
`server_evidence/decision.md`. The frozen source is identified in
`server_evidence/desktop_handoff.md`. The A100 gate and full-data experiment
have their own contract in `protocol.md`; no full-scale accuracy claim was
available at handoff acceptance.

## Initial Timing Gate (2026-09-07)

Job 1448892 completed 64 FP32 train batches (12,288 graphs) in 12.932596 s,
with the expected 3,665,809 parameters and 98.684% GPU memory reserve. Its
12-epoch projection including the frozen margin was 51,204.080 s (14.223 h),
so the gate rejected the run before full training. This was a budget rejection,
not a model-quality result. A representative multi-shard timing calibration
was prepared under the unchanged scientific configuration; its method is
specified in `protocol.md`.

## Representative Gate and Time Override (2026-09-07)

Eight shuffled train strata covered 196,608 measured molecules after warm-up.
The estimate included measured per-shard IO and atomic-checkpoint cost and
reached 45,353.313 s (12.598 h) with 20% reserve. Minimum GPU memory reserve
was 98.793%; the model count was 3,665,809 and every measured FP32 step was
finite. The old 12-hour time gate still rejected this result.

The user explicitly waived that time ceiling while preserving the model and
12 training epochs. The approved override retained strict numerical, memory,
cache and source checks and allowed a 24-hour cumulative budget across at
most two serialized 14-hour scheduler segments. Exact evidence is retained
in `preflight_representative.json` and `runtime_override.json`.

The 14-hour scheduler request 1450739 was held as `(long)` beyond maintenance
and cancelled while still queued. Four four-hour segments were then submitted
with serialized afterok dependencies and 3.5-hour atomic pause boundaries.
These are segments of one training run, not four models; a complete manifest
makes later segments exit without further optimization. Exact job IDs are
retained in `launch.json`.
