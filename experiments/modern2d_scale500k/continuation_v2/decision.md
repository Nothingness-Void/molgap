# Native checkpoint continuation — 2026-10-26

Kernel140916841/version1 admitted both native preflights and performed formal
training. ESA reached its tenth physical pass; automatic chunk export attempted
to put pending scalar Tensor metrics into a JSON cursor summary and raised a
serialization error. The coordinator then stopped the other worker. This was an
infrastructure interruption, not a cost stop or a model-quality rejection.

The [retained partial evidence](../results/interrupted_r3_v1/summary.json) bound
ESA's native39060-step checkpoint and PPGT's native4047-step checkpoint. Both
passed family-owned state validation. Restore retained optimizer, scheduler,
sampler, RNG, selected predictions and pending metric windows. The JSON summary
converted only finite scalar tensors to exact Python values; the native checkpoint
bytes and tensor dtypes remained unchanged. The original frozen reader independently
accepted both bundles. Historical transport summaries derived their top-level
step count from the last observed metric; authoritative progress came from the
complete native checkpoint and subsequent actual-resume proof, not that summary.

Reuse `build_workflow_resume` and `prepare_resumed_workflow`. The owning Kaggle
publication/push adapters appended source-dataset version2 and kernel version2
after confirming version1 was terminal ERROR. Private source version2 underwent
complete byte roundtrip. Original Spec, source commit7b99147, recipes, prospective
actions, data and scientific endpoints stayed frozen; no new baseline, seed,
architecture or fresh training was released. Both checkpoints preceded the
contract endpoints. The remaining physical passes did not reach another ten-pass
chunk boundary before the fixed final endpoint.

The retained shared allocation ledger measured6877.277398151s. Its identical
copies across bundles were deduplicated by the existing runtime, not added twice.
Continuation retained the original32372s aggregate allocation ceiling, including
the previous invocation, with the existing600s output reserve. Actual resumed
native proofs, completed outputs and scientific comparison remained acceptance
requirements; a scheduler status alone established none of them.
