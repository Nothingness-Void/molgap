# Frozen protocol: K1 multiplicative PairValue

## Question

Does the accepted K1 PairToken selector benefit when selection and relation
value use separate representations, with the value changed from an additive
pair feature to an ordered low-rank multiplicative interaction?

## Evidence basis

- Original PairToken is the only K1 relation mechanism to clear its frozen
  PCQM-100K material gate. Frozen-checkpoint interventions showed learned
  selection, cross-node pairs and per-pair normalization are all active.
- Chemistry-conditioned selection was worse than original PairToken; selector
  refinement is closed.
- Node-conditioned return was favorable but sub-threshold; return-allocation
  refinement is closed.
- Functional-group exchange was favorable but sub-threshold. No chemistry
  sidecar or external feature is used here.
- Across retained aligned predictions, PairToken, chemistry-conditioned,
  functional-group and node-return corrections have only moderate pairwise
  correlation (`0.49–0.56`), so relation content is not exhausted, but stacking
  those closed mechanisms is not authorized.

## Single mechanism

At layer 6, preserve PairToken's additive learned-query selector:

```text
selector(i,j) = LN(SiLU(Ws hi + Wt hj))
assignment    = softmax(q · selector)
```

Use independent source and target projections for the selected value:

```text
value(i,j) = LN(SiLU(Vs hi) * SiLU(Vt hj))
token      = sum assignment(i,j) * value(i,j)
```

The ordered Hadamard product is a low-rank feature conjunction. Selector and
value parameters are disjoint. K1, EdgeState, RWSE16, K1 exchanges, target,
head, optimizer and exposure remain unchanged. The final return projection is
zero initialized, so the candidate is exactly K1 at initialization.

This is not dense atom-to-atom attention, geometry, target residual, prediction
fusion, a teacher, a chemistry sidecar, another selector, or another return
gate.

## Frozen screen

- Accepted cross-platform PCQM-100K V4 cache and row order.
- Direct Gap, seed 42, deterministic FP32/no TF32.
- One T4, physical BS128, drop-last, 40 epochs, 31,240 optimizer steps and
  3,998,720 sample presentations.
- AdamW `4e-4`, weight decay `1e-5`, clipping `1.0`, cosine to `1e-6`.
- Immutable K1-v4 reference; original PairToken is a retained mechanism
  control and is not retrained.
- Official validation, test-dev and challenge roles remain sealed.

## Decision rule

The candidate must improve K1-v4 by at least the prospectively frozen
`0.003 eV` gate with a favorable paired interval. It must also improve the
original PairToken point estimate with a favorable paired interval to be
called a value-side mechanism winner. Otherwise this exact mechanism closes;
there is no automatic seed, scale bridge, protected-role access or successor.

