# Kunshan persistent-vector screen status

- Stage: unchanged paired screen running as Kunshan job `121002721`.
- Models: frozen GraphState9 control followed by persistent VectorState16.
- Platform: SCNet Kunshan, one Hygon DCU, sequential candidate execution.
- Protocol: ../../kunshan_vector_state_protocol.md.
- Plan/budget: ../../kunshan_discovery_plan.md.
- Preflight job `120998294` ended after 4:01. Exact parameter counts and every
  model-path check passed except the symmetry harness, which mixed a CPU
  synthetic coordinate tensor with a device transform before batch transfer.
  This is an infrastructure-only failure and is eligible for one repaired
  preflight retry. No training or sealed evaluation role was read.
- Repaired preflight job `120998699` completed in 3:05 with every declared
  check accepted. Formal job `120999044` was submitted from source commit
  `a359c282e7d6a341d4716f31a37f052ea94412fa`; it was pending for priority at
  the first post-submit check. See `launch.json`.
- Formal job `120999044` failed before epoch 0 after 2:34 because Bash expanded
  an empty `resume_args` array while `set -u` was active. It produced no
  checkpoint, metrics, progress, MAE, throughput or memory evidence. The
  accepted preflight remains valid; the wrapper repair does not change the
  architecture, data, seed or training contract. See `failure_120999044.json`.
- The wrapper was repaired by replacing the unset-array expansion with explicit
  fresh/resume branches. Source `45d2e28db49adb24ca38d141f0362a9bc4e912a6`
  was pushed and remotely syntax-checked. Retry job `121002721` started on
  `e06r4n13`; its run manifest has passed the original failure point and fixes
  the accepted cache, source, two candidates and sealed-role flags. See
  `retry_launch.json`.
- No scientific result or seed-43/44 authorization exists.
- Terminal handoff coordinator: 01a025a1-3b87-7781-8a91-f183193f7865.
- Persistent Luna Max monitor task: 01a04479-ca44-7d31-95c4-6be485f256cc.
