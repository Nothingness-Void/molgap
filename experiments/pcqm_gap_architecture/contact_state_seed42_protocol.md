# Non-covalent ContactState seed-42 protocol

## Question

Does a separately normalized through-space relation provide information that
is absent from the accepted covalent EdgeState, wedge geometry, and shared
GraphState backbone?

This protocol first authorizes only a CPU cache and statistics gate. It does
not authorize GPU training until that immutable cache passes acceptance. It
does not authorize cutoff search, neighbor-cap search, extra seeds, full-data
training, official validation/test-dev access, or molecular-research-server
work.

## Frozen relation

- Parent input: the accepted PCQM 100K/10K ETKDGv3+MMFF94s geometry cache,
  aggregate SHA-256
  `3e4206fd239942ab79f9c4978cd4334f6025ba14e8f1cd1df78c20060a0d1d22`.
- Only graphs marked `geometry_valid=1` may receive contact relations. Invalid
  geometry graphs remain present with empty contact tensors.
- An undirected atom pair is a contact exactly when its Euclidean distance is
  at most `5.0 Å` and its covalent shortest-path distance is greater than
  three or disconnected.
- Real bonds, bonded angles, and 1--4/torsion pairs are therefore excluded.
- Accepted pairs are sorted lexicographically and stored in both directions,
  with one finite distance per directed edge. There is no learned or
  label-dependent pair selection and no neighbor cap.

The cutoff and three-hop exclusion are fixed before statistics are observed.
The cache records role-separated pair counts, atom-type-pair counts, atoms and
graphs covered, cross-component contacts, invalid/empty graphs, shard hashes,
and aggregate lineage. No target value is used for contact construction or
acceptance.

## CPU advancement gate

The cache advances only if all 100,000 train and 10,000 internal-validation
graphs are preserved, there are zero conversion failures, all distances are
finite and in `(0, 5.0]`, no contact overlaps a covalent pair within three
hops, both roles contain contacts, and the directed-edge total is at most
10,000,000. Every shard and the parent cache identity must pass no-model
acceptance. Official PCQM validation and test-dev remain unread.

## Conditional GPU question

Only after CPU acceptance may a later frozen protocol compare two fresh
seed-42 models on T4x2:

1. the accepted distance-angle Triangle EdgeState GraphState9 baseline;
2. the identical model plus a narrow separately normalized ContactState,
   exchanged with atoms at fixed blocks.

That later protocol must freeze width, exchange blocks, exact parameter count,
initialization equivalence, optimizer, runtime budget, and acceptance before
submission. A seed-42 gain does not automatically authorize seeds 43/44.
