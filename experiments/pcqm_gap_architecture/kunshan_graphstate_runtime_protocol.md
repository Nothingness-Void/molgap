# Kunshan GraphState runtime gate

## Question

Can the frozen PCQM Gap GraphState9 encoder execute on one Kunshan Hygon DCU
with the accepted official-train-derived 100K/10K cache, while retaining the
12-hour full-data budget evidence needed by the desktop handoff?

This is a runtime and throughput gate, not a new architecture comparison. A
successful run does not authorize full-data training or official-role access.

## Frozen identity

- Candidate: `ogb_distance_angle_triangle_edge_state_graph_state9`
- Model source commit: `9068ddb82e6bdf16b841570abbff023b90c07f07`
- Parameters: 3,665,809
- Inputs: OGB categorical atom/bond features, RWSE16, persistent real-bond
  EdgeState64, sparse non-backtracking wedge state16, ETKDGv3/MMFF94s bond
  distances and wedge-angle cosines, and GraphState64 exchanges after blocks
  3, 6, and 9
- Geometry cache aggregate: `3e4206fd239942ab79f9c4978cd4334f6025ba14e8f1cd1df78c20060a0d1d22`
- Roles: 100,000 train graphs and 10,000 internal validation graphs only

The cache manifest, every shard hash, geometry failure ledger, and both role
counts must be checked before the DCU is initialized. Official validation,
test-dev, and every other sealed role are forbidden.

## Runtime contract

- Seed: 42
- Precision: FP32; do not enable TF32 or BF16
- Batch size: 48; one visible DCU
- Optimizer: AdamW, learning rate `1.6e-4`, weight decay `1e-6`
- Target: direct scalar Gap, L1/MAE objective
- Short gate: 3 complete train epochs plus internal validation
- DataLoader: one process, zero workers, no pinned-memory assumption
- Initialization: fresh random initialization; no checkpoint, pretraining,
  residual target, fusion, or warm start

The runner writes an atomic progress JSON after each epoch, an atomic last
checkpoint, a best model, finite metrics, and a terminal manifest. No test
predictions are produced. A scheduler success without these artifacts is not
accepted.

## Gate

The gate passes mechanically when the cache and source identities match, the
parameter count is exact, all three epochs and validation metrics are finite,
the DCU is used, and the output manifest is complete. Record throughput and
peak memory for the later budget calculation; do not infer scientific
superiority from this short run.

## Follow-up

Only after the gate is accepted should the coordinator decide whether a longer
single-candidate timing run is justified. This protocol never authorizes
official validation/test-dev access, a second candidate, or seeds 43/44.
