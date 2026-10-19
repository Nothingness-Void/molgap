# Local preparation rejection

V3 was never uploaded or submitted. Its frozen declarations and source
commit099130ac were preserved after the shared source allowlist rejected the
historical [V2 inspection](../retry_v2/results/inspection_2026-10-18.md) path during local package
preparation. The rejection occurred in1.0445525s before an output package or
initial-state validation was created; no GPU or scientific exposure followed.

Raw evidence: `E:/MolGap/artifacts/modern2d-native-preflight/retry_v3/prepare_package_cpu.log.json`
and its sibling `prepare_package_cpu.stdout`. The [CPU device/registry repair](results/repair_2026-10-18.json)
remained scoped local evidence, not a claim that V3 packaging or native execution
was accepted. A separate V4 declaration froze the publishable history binding;
its exact platform state belongs to `CURRENT_STATE.md`, not this rejected run.
