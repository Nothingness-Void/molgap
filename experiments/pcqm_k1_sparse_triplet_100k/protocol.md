# Frozen protocol: K1 sparse topological triplet state

## Question

Does persistent state on complete directed non-backtracking bond triplets improve
K1 when it is integrated into the existing local EdgeState stream without
adding coordinates, dense atom attention, prediction fusion, or a new target?

## Evidence basis and round priority

- K1 remains the immutable reusable PCQM-100K reference.
- PairToken content refinements did not beat K1 materially; another atom-pair
  token is therefore lower information gain.
- The earlier pure-2D sparse triangle/wedge EdgeState screen improved its paired
  comparator in all three seeds, with a positive mean direction. Its mechanism
  has not been transplanted into K1.
- Distance/angle Triangle GraphState won a 100K screen but failed at full scale.
  That closes that geometry-heavy GraphState combination, not this topology-only
  K1 intervention.
- Earlier shortest-path candidates were weak or negative. Graphormer-style SPD
  conditioning is retained for a later round only if terminal evidence from this
  round supports additional topological conditioning.

This makes sparse triplet state the first of at most three user-authorized,
evidence-gated rounds. Authorization is a ceiling, not permission to launch
three jobs without terminal reanalysis.

## Single mechanism

For every complete directed non-backtracking path `i -> j -> k`, maintain one
16-channel state. At each of K1's nine layers it reads:

```text
EdgeState(i,j) + center node(j) + EdgeState(j,k)
                         |
               persistent triplet state
                    /             \
          real-bond EdgeState   center node
```

The triplet-to-edge and triplet-to-node projections are zero initialized, so
the candidate is exactly K1 at initialization. Wedges are deterministically
derived from the already accepted real-bond `edge_index`; preflight verifies
adjacency, non-backtracking semantics, uniqueness, and complete set equality.

This is not ETKDG, a bond angle, a coordinate, an all-pairs path matrix, dense
attention, target residual, teacher signal, or prediction ensemble.

## Frozen screen

- Accepted cross-platform PCQM-100K V4 cache and row order.
- Direct Gap, seed 42, deterministic FP32/no TF32.
- One T4, physical BS128, drop-last, 40 epochs, 31,240 optimizer steps and
  3,998,720 sample presentations.
- AdamW `4e-4`, weight decay `1e-5`, clipping `1.0`, cosine to `1e-6`.
- Immutable K1-v4 reference; it is not retrained.
- Official validation, test-dev and challenge roles remain sealed.

## Decision rule

The candidate must improve K1-v4 by the prospectively frozen V5 policy gate,
currently `0.003 eV`, with a favorable paired interval. A smaller positive
endpoint is retained as mechanistic evidence but does not authorize scaling or
extra seeds. A negative endpoint closes this exact K1 sparse-triplet transplant.
Any second round requires a new analysis of the terminal RML corpus; no
automatic successor is authorized by this protocol.
