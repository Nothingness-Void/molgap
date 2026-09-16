# Round-2 layer-6 strength audit

## Question

Does K1's layer-6 global exchange have the correct direction but excessive or
insufficient residual strength for topology-extreme molecules?

## Frozen contract

- Reuse the accepted K1-v4 seed-42 checkpoint and fixed 50K development role.
- Perform no training and mutate no checkpoint weight.
- Reproduce the frozen K1-v4 payload before accepting counterfactuals.
- Multiply only the layer-6 global update by
  `0.50, 0.75, 1.00, 1.25, 1.50` before residual addition.
- Keep layers 3 and 9 at coefficient `1.00`.
- Keep local EdgeState, data order, FP32/no-TF32, physical inference batch 128,
  pooling, head, target transform, and role access unchanged.
- Read no official validation, test-dev, or test-challenge role.

## Interpretation gate

A direction is coherent only when at least one non-unit coefficient improves
both overall and topology-extreme-union MAE, its topology-extreme paired 95%
interval is positive, and middle-control regression is no worse than
`0.0005 eV`. A smooth neighboring response is supporting evidence.

Round 2 cannot promote a model. If it identifies no coherent direction, Round
3 is not spent. If it does, Round 3 may train exactly one minimal nested repair
derived from the measured layer-6 magnitude mismatch under the unchanged
PCQM-100K v4 contract. The trained repair still requires at least `0.003 eV`
gain over K1-v4 and receives no automatic seed, shadow, scale, or official-role
authorization.
