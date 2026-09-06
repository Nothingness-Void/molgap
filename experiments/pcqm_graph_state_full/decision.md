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
