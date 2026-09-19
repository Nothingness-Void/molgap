# Protocol: node-adaptive PairToken return

## Question

PairToken's learned, pre-normalized ordered-pair selection is useful, but its
single relation token is returned identically to every atom. Does conditioning
that return on each current layer-6 node state preserve relation evidence while
avoiding an over-broad molecule-wide correction?

## Single intervention

The K1-v4 backbone, layer-6 insertion, ordered-pair construction, per-pair
LayerNorm, learned pair query, one relation token and zero-initialized return
projection are unchanged from PairToken. Only the return allocation changes:

```text
relation token + current node state
  -> 32-channel node query
  -> SiLU + per-node LayerNorm
  -> zero-initialized projection
  -> node-specific residual
```

The zero projection makes the candidate exactly K1 at initialization. It adds
6,240 parameters beyond PairToken and does not add geometry, atom-to-atom
attention, a graph-level gate, a second encoder, a teacher or target residual.

## Frozen screen

- Fixed PCQM roles: train `[0,100000)`, development `[100000,150000)`.
- Direct Gap; seed 42; deterministic FP32/no TF32.
- Physical BS128, no accumulation, `drop_last`.
- AdamW `4e-4`, weight decay `1e-5`, clip `1.0`.
- Cosine schedule, 40 epochs, 31,240 optimizer steps and 3,998,720 sample
  presentations.
- Official validation, test-dev and challenge roles remain unread.

The recovered K1-v4 bundle is the release comparator. Terminal analysis also
uses the retained aligned PairToken payload to determine whether the changed
return improves the positive parent mechanism. A shortlist requires at least
`0.003 eV` gain versus K1-v4 and a favorable paired interval. Improvement over
PairToken is mechanistic attribution, not a replacement for the K1 material
gate. No seed, 500K bridge or protected role follows automatically.

