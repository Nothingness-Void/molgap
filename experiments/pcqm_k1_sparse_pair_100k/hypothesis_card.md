# K1 sparse-pair hypothesis card

## State

- Frozen baseline: K1-v4, `3,658,817` parameters.
- Fixed role: official-train-derived 100K training rows and next 50K
  development rows; official validation and all test roles remain sealed.
- Baseline development Gap MAE: `0.1413736343 eV`.
- Existing all-pair PairToken passed 100K by only `0.0000436 eV` above the
  material gate, then transferred only `0.0005561 eV` at 500K and was closed.

## Evidence

The PairToken causal audit found that learned pair selection, cross-node pairs,
and per-pair normalization were active. K1 attribution found coherent weakness
on cyclic/high-RWSE topology, while scalar changes to the layer-6 exchange all
failed. Published GRIT, SPSE, and sparse higher-order transformer results
support topology-aware relative pair information without requiring a dense
persistent all-pair state.

## Hypothesis

K1 needs a node-specific relation return rather than another molecule-level
token. At layer 6, normalized directed pairs within three topological hops use
bounded non-backtracking path counts and pairwise RWSE, select senders per
receiver, and return a different update to each receiving atom.

Alternative explanation: the 100K PairToken result was optimization noise or
regularization that cannot transfer, so even a better structured pair path may
remain below the material gate.

## Cheapest falsifier

1. Run identical BS128 FP32 profiles on Kaggle T4 and Kunshan DCU.
2. Train once on the faster accepted platform under the frozen 40-epoch K1
   contract.
3. Require at least `0.003 eV` scalar gain and a favorable paired bootstrap
   interval against the frozen K1 prediction payload.

Failure closes this mechanism. It does not release another seed, 500K, full
training, or protected-role evaluation.
