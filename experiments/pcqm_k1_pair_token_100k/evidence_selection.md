# Evidence selection

## What the audits established

K1's advantage is not explained by a dispensable exchange or a wrong global
residual strength. Deleting any exchange was destructive, and multiplying the
layer-6 update by `0.50`, `0.75`, `1.25`, or `1.50` worsened every audited
stratum. The three sparse rank-one exchanges are a co-adapted backbone and must
remain unchanged.

Increasing the capacity of the atom slot has repeatedly failed: four slots,
four selection heads, a graph-conditioned query, a molecule gate, and a
real-bond relation slot all missed the material gate. Paths, triangles,
directed-bond recurrence, extra local adapters, and direct pair-memory readback
are also closed. Another slot-count, gate, depth, width, or residual-scale
variant would therefore have low information value.

## Remaining positive evidence

GPTrans Pair PreNorm improved its own frozen all-pair relation-flow reference
by `0.0031249 eV`. Separately, K1 is substantially stronger than the GPTrans
100K model. The useful mechanism may therefore be the normalized relation
channel rather than GPTrans's dense node-to-node attention or full backbone.

This motivates one deliberately narrow question: can K1 retain its successful
one-atom-slot communication and add one normalized all-pair **relation token**
without restoring dense atom-to-atom attention?

## Why this is not a reopened failure

- K1-R pools only persistent states on real bonds at all three exchange layers.
- Sparse-path and triangle candidates return many local/path messages to nodes.
- GPTrans uses a full pair tensor throughout twelve attention blocks.
- This candidate forms all ordered pairs from the current layer-6 node states,
  normalizes each pair across channels, pools them into exactly one token, and
  broadcasts one zero-initialized residual once.

The experiment changes one information path, remains pure 2D, adds about 23K
parameters, and preserves the exact K1 function at initialization. A negative
result closes this relation-bottleneck hypothesis; it does not authorize a
different pair width, insertion layer, or pooling head.

