# Kunshan persistent-vector screen status

- Stage: closed after accepted seed-42 paired comparison.
- Job: `121002721`, `COMPLETED`, exit 0, node `e06r4n13`.
- Models: frozen GraphState9 control followed by persistent VectorState16.
- Platform: SCNet Kunshan, one Hygon DCU, sequential candidate execution.
- Protocol: ../../kunshan_vector_state_protocol.md.
- Plan/budget: ../../kunshan_discovery_plan.md.
- The original exported validation IDs were invalid because PyG increments
  batched attributes containing `index`. Cache-backed identity repair changed
  only row IDs and artifact hashes; predictions, targets, and MAEs were
  unchanged. The repaired artifacts passed no-inference acceptance.
- VectorState improved MAE by less than the plan's `0.001 eV` promotion gate
  and reduced throughput. It is retained as a weak observation and is not
  promoted to more seeds or full-data training. See `decision.md`.
- Earlier infrastructure-only preflight and wrapper failures are preserved in
  `failure_120999044.json`, `launch.json`, and `retry_launch.json`.
- The terminal monitor handoff completed and its heartbeat is no longer active.
