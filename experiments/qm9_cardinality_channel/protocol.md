# QM9 cardinality-preserving channel protocol

**Frozen: 2026-09-10. Track C, seed 42 only.**

## Question

Can a query-gated, unnormalized sum over each atom's exact three-hop 2D support
improve the frozen EdgeState GPS9 representation beyond both the fresh model
and a parameter-matched support-size control?

## Immutable data and model contract

- Reuse the independently accepted GAPE-lite cache: 30,000 QM9 train graphs,
  3,000 internal-validation graphs, OGB 9-channel atom categories, OGB
  3-channel bond categories, RWSE16, split fingerprint
  `62f1cdefdaec6877`, and aggregate SHA-256
  `80a2d256ccbbde8de5862e7ad4fd61ea7c1255a19217e590033a94cabe1c6340`.
- No held-out/test graph, PCQM role, geometry, conformer, external row,
  checkpoint, pretrained weight, teacher distillation, HOMO/LUMO target,
  residual, or prediction fusion is available.
- Every arm retains the same 192-wide, nine-layer, four-head EdgeState GPS,
  persistent 64-channel real-bond state, direct scalar Gap head, and mean
  pooling. This does not copy or compare the desktop 304-wide experiment.

## Paired arms

1. `baseline`: unchanged EdgeState GPS9.
2. `size_control`: after layers 3/6/9, a shared zero-return channel multiplies
   the target atom's projected state by `log(1 + |S_3(i)|)` and a query gate.
3. `cpa`: the same modules, initialization and placements instead sum projected
   source states without normalization over
   `S_3(i) = {j : shortest_path(i,j) <= 3}`, including `i`, before applying
   the same query gate.

The K<=3 support is derived deterministically from the existing directed bond
graph inside the forward pass. It is permutation invariant and imports no new
feature or preprocessing cache. The shared return projection is zero-
initialized, so both controls must exactly reproduce the baseline function at
initialization while retaining a finite nonzero return gradient.

## Training and resource contract

- One private Kaggle2 `NvidiaTeslaT4` task with two isolated workers.
- GPU0 trains `baseline`; GPU1 trains `size_control` and then `cpa`.
- Seed 42, FP32, physical batch 128, no accumulation, AdamW `4e-4`, weight
  decay `1e-5`, gradient clip 1, cosine to `1e-6`, and exactly 40 direct-Gap
  epochs for every arm.
- All arms share row order, labeled exposure, optimizer, schedule and source
  identity. Each epoch writes atomic checkpoints, traces, best models and
  validation payloads.
- Each candidate must stay below 5.1M inference parameters, retain at least
  15% T4 memory reserve, and run at no more than 1.35x baseline epoch time.

## Decision rule

CPA is nominated only when independent no-model acceptance confirms all
contracts and it simultaneously:

- improves over the fresh baseline by at least `0.003 eV`;
- improves over the size control by at least `0.001 eV`;
- satisfies the parameter, memory and runtime gates.

A pass only permits planning one paired PCQM-100K transfer. It does not
authorize another seed, a desktop/full run, official PCQM validation/test-dev,
SCNet, IMS, or a 304-wide variant. Failure closes this exact K=3 shared-channel
mechanism without hop, layer, width, seed, optimizer or schedule retries.
