# Evidence selection

Decision date: 2026-09-20

## RML evidence used

The validated RML index contains 12 trajectories: one
`POSITIVE_UNDER_CONTRACT`, one `POSITIVE_BELOW_GATE`, four
`NEGATIVE_UNDER_CONTRACT`, one `STOP_FOR_COST`, and five
`INFRASTRUCTURE_ONLY`. The useful causal directions are narrow:

- K1 PairToken was the only 100K mechanism winner, but its 500K bridge retained
  only a sub-threshold gain. Learned cross-node pair selection and pair
  normalization were active; target-scale benefit was not established.
- Selective MoSE residuals improved difficult K1 rows and damaged easy rows.
  Direct replacement, hidden normalization, and molecule-context gating did
  not produce a promotable mechanism.
- GPTrans profiles localized runtime to learned node/pair forward and backward
  work. Shortest-path caching, loader changes, larger physical batches, finite
  checks, optimizer and EMA mechanics did not yield a validated material fix.
- Historical local atom/bond/functional-group reconstruction produced a
  directional `0.00234205 eV` equal-exposure gain on EdgeState, but its exact
  batch-48 allocation remained below its frozen gate and is not a reusable V5
  winner.

## Closed interpretations

This portfolio does not reopen slot count, selector heads, dynamic queries,
global-strength gates, relation slots, edge-memory normalization, GPS++ local
adapters, shortest-path/ring/geometry branches, generic MoSE gate variants,
PairToken batch scaling, GPTrans path caching, or Pair PreNorm continuation.
Width, depth, seed, learning-rate and schedule variants do not constitute new
routes.

## Remaining information-flow questions

### Route A — chemistry-defined functional-group tokens

K1 has strong atom/bond memory but no explicit sparse atom-to-functional-group
hierarchy. The candidate will add deterministic chemistry-defined group tokens
and sparse atom/group exchange while leaving K1 local EdgeState and its three
global exchanges unchanged. This tests a new representation level, not another
global slot or graph-level histogram.

Alternative explanations are that OGB atom/bond features already determine the
same information, or that old fragment/ring failures generalize to all group
tokens. One seed-42 PCQM-100K architecture comparison is the cheapest
falsifier.

### Route B — node-adaptive PairToken return

The accepted PairToken compresses learned ordered-pair evidence into one
relation token and broadcasts one identical residual to every node. Its 100K
gain and weak 500K transfer are consistent with useful relation evidence but an
over-broad return path. The candidate keeps pair construction, learned
selection and pre-normalization unchanged and changes only the return: each
node conditions the relation-token residual on its current state.

Alternative explanations are that the 100K gain was run variation or that the
500K loss reflects pair selection rather than return allocation. One seed-42
PCQM-100K architecture comparison followed by subgroup attribution is the
cheapest falsifier. No 500K bridge follows unless it clears the prospectively
calibrated gate.

## Why pretraining is not the first released route

Local-hierarchy reconstruction remains scientifically interesting, but it
changes objective and schedule identities and therefore cannot be represented
as a strict architecture-only comparison against the immutable K1 reference.
Desktop also owns separately configured pretrained/full-scale work. A server
pretraining question remains reserve-only until Routes A/B finish and a
separate matched training-objective contract is explicitly frozen.
