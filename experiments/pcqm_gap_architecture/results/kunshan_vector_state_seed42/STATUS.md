# Kunshan persistent-vector screen status

- Stage: repairing the bounded remote preflight; no long training submitted.
- Models: frozen GraphState9 control followed by persistent VectorState16.
- Platform: SCNet Kunshan, one Hygon DCU, sequential candidate execution.
- Protocol: ../../kunshan_vector_state_protocol.md.
- Plan/budget: ../../kunshan_discovery_plan.md.
- Preflight job `120998294` ended after 4:01. Exact parameter counts and every
  model-path check passed except the symmetry harness, which mixed a CPU
  synthetic coordinate tensor with a device transform before batch transfer.
  This is an infrastructure-only failure and is eligible for one repaired
  preflight retry. No training or sealed evaluation role was read.
- No scientific result or seed-43/44 authorization exists.
- Terminal handoff coordinator: 01a025a1-3b87-7781-8a91-f183193f7865.
- Persistent Luna Max monitor task: 01a04479-ca44-7d31-95c4-6be485f256cc.
