# Frozen PairToken selection diagnostic

## Question

On the accepted PCQM-100K V4 development role, is the trained layer-6
PairToken assignment concentrated enough to justify a dynamic sparse-pair
architecture question? This is a frozen-checkpoint diagnostic, not a new model
screen or an independent validation estimate.

## Frozen inputs and boundaries

- PairToken checkpoint SHA-256:
  `ac8b576a3c7e44d50012a885e13e4c5904ad17c3574319796b13518efe98d1cc`.
- PairToken prediction payload SHA-256:
  `b67be3f9ba0c67ca0eb1cf7dffa00ad535e9f04f5f8371aadad1e6a2711e0331`.
- Immutable K1-v4 development payload SHA-256:
  `966ed31ba25e024aa82d8032e7ab6e2797402e5845a01888c8f3cfd4c084da91`.
- Fixed graph manifest SHA-256:
  `1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d`.
- Evaluate all 50,000 ordered development graphs, source rows `[100000,150000)`;
  no training, optimizer step, model mutation, official validation, shadow or
  test-role access. Reproduce the frozen PairToken predictions before reporting
  attention statistics. No new graph cache or conformer is constructed.

## Measurements

For every molecule, record the mass on the largest `ceil(0.20 * n^2)` ordered
pair scores, effective support `exp(entropy)/n^2`, diagonal mass, true-bonded
mass, and graph-distance-two mass. Reduce these by atom-count and conjugation
strata, and by paired K1-to-PairToken development gain. Recoverable 5,000-row
numeric chunks remain in ignored remote/local record storage; the Git record
contains only aggregate statistics and content hashes, not new predictions.

## Prospective interpretation gate

Dynamic top-pair sparsification becomes a *candidate question*, not an
authorized training job, only if median top-20%-pair mass is at least 0.80,
at least 70% of molecules individually reach 0.80, and the medians both for
graphs with more than 12 atoms and for the hardest K1-error decile reach 0.80.
Otherwise close the sparse-pair idea. The
diagnostic cannot promote PairToken, consume an independent role, or silently
release a new GPU training run. Any later architecture trial needs a distinct
V5 prelaunch, immutable reference binding, complete feature identity, cost
plan, and prospective replay-ready trajectory.
