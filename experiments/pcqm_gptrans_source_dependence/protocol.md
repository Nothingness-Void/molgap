# Frozen source-dependence assay — 2026-10-11 JST

The user authorized submission following the
[methods/ablation review](../pcqm_gptrans_triplet_portability/research_followup.md).
This is a Track C NO_TRAIN / CONTEXT_ONLY diagnostic, not a new trained
architecture, a training Replay pair, a fresh holdout or a compute release for
source dropout. No fitting, gradient, optimization or checkpoint reselection.

## Frozen inputs and panels

Reuse the independently accepted local-bond parent selected at epoch41 and
triplet aggregation selected at epoch38. Hashes, retained-state identities,
target transform and frozen original architecture sources live in `contract.json`.
Consume accepted cross-platform PCQM100K/500K graph assets, solely their internal
development roles. Do not read training shards, geometry, official validation,
test-dev or challenge. Internal graph labels and retained labels are used only
to verify reproduction; targets/residuals never select panel membership.

Both models observe the same512 rows per role. Original selection is sorted
RandomState42 choice512 from50000 input indices. Later selection is sorted
RandomState42 choice512 within the previously fixed10000 later indices. Exact
membership is frozen in hash-bound `panels.json`. These are reused development
panels, not the full canonical50K later audit and not representative extrapolation
proof. Physical inference batch128, seed42, FP32/noTF32, deterministic eval.

## Read-only observations

- All12 GPA layers: per-graph/head source entropy, maximum mass, virtual-source
  mass and exp(entropy)/valid-source-count. Record separately real-query averages
  and the virtual query. Repeat for the32-channel pair-to-node normalized source
  flow, not just the8-head node flow.
- Real-source-conditional statistics remove virtual source mass and renormalize
  real sources. Zero real mass is explicitly recorded and excluded from those
  conditional means. Never diagnose collapse from virtual attention alone.
- Triplet layers3/6/9/12: inward/outward pre-gate softmax statistics and total
  softmax mass, post-sigmoid gated mass and gated virtual-source mass. Observe
  actual returned pair RMS, input pair RMS and their ratio on valid pairs,
  including virtual pairs. Query/source padding is excluded everywhere.
- Record atom count, directed real-bond count, mean/max degree for every row.
  Preserve row/head/layer observations, not only a global mean.

Hooks shadow deterministic existing projections and read actual branch output;
they never replace arguments, masks, return values, weights or buffers. Each batch
also runs unobserved inference and requires bitwise identical normalized predictions
and unchanged CPU/CUDA RNG. Each model-role pass requires unchanged state digest.
Panel predictions must reproduce accepted original/later predictions with the
existing max-absolute0.001eV / MAE-difference0.0001eV mechanical tolerances.

## Durability, cost and interpretation

One worker isolates one T4 before CUDA import. Sequentially inspect both accepted
models; a second idle allocated T4 does not authorize another model and its cost
is included. The entire setup/environment/inference/output allocation is capped
at1800 seconds and at most1 allocated T4-hour. No automatic retry or successor.
Write each128-row observation chunk and progress atomically, with manifest SHA,
runtime, role events, noninterference and native cost. Retained frozen input
states make recovery independent of transient memory. Fail closed on identity,
reproduction, finite values, layer coverage, inventory or cost mismatch.

Independent local acceptance only reads retained tensors/JSON; no local model
construction/inference. Controller interpretation is required before terminal
RML closure. Compare descriptive distributions by role, model and input size;
do not choose thresholds after viewing results or equate concentration with
overfitting. The assay can prioritize or weaken a source-dependence explanation;
it cannot prove net value of retraining or release500K/full/extra seeds.
