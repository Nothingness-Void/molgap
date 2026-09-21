# Frozen protocol: K1 Graphormer-SPD PairToken

## Question

Can explicit chemical graph distance improve which ordered atom pairs the
accepted layer-6 PairToken selects, without adding a path-value stream,
persistent path state, or dense atom-to-atom attention?

## Evidence basis

- Original PairToken is the only K1 relation mechanism to clear the frozen
  PCQM-100K material gate. Its causal audit showed learned pair selection and
  cross-node pairs are both necessary.
- Chemistry-conditioned selection remained favorable versus K1 but did not
  improve original PairToken. A more direct relational prior is still untested.
- Uniform exact 2/3-hop GraphState mixing was weakly positive, while learned
  relative path values regressed. Those results reject path-message/value
  aggregation, not a scalar distance prior on an already successful selector.
- Full-depth sparse triplet state fit training more strongly, helped hard K1
  rows, and harmed easy rows. The next intervention must therefore be smaller
  and selection-only.

## Single mechanism

Keep the accepted PairToken value, normalization, token FFN, and return path.
Add one learned scalar to each ordered-pair selection logit according to the
shortest-path bucket `0, 1, 2, 3, 4, 5, >=6`:

```text
pair logit(i,j) = accepted PairToken logit(i,j) + bias[SPD(i,j)]
```

The seven biases are zero initialized. The complete candidate is exactly K1
at initialization and retains the original PairToken selection at release.
Shortest paths are derived deterministically from the existing real-bond graph
inside the layer-6 PairToken computation. PairToken is already all-pair, so the
new mechanism adds no asymptotic pair tensor and only seven parameters.

This is not Graphormer's full attention stack, uniform path averaging, learned
path values, ETKDG geometry, a teacher, target residual, or prediction fusion.

## Frozen screen and decision

Use the fixed V4 PCQM-100K/50K roles, direct Gap, seed 42, strict FP32/no TF32,
physical BS128, 40 epochs, 31,240 steps, and the unchanged AdamW/cosine
contract. Official validation, test-dev, and challenge stay sealed.

The candidate must improve K1 by the prospectively frozen policy gate and beat
the retained original PairToken endpoint with a favorable paired interval.
Otherwise this exact SPD-selection mechanism closes. No seed, scale bridge,
protected-role read, or third round is automatic.
