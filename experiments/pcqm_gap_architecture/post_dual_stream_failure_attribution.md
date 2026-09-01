# Post-dual-stream failure attribution

This note answers one question only: why did the separately normalized sparse
atom--bond dual stream fail, and what must a successor change to avoid a
duplicate experiment? The terminal metric and artifact disposition remain in
[`results/sparse_atom_bond_dual_stream_seed42/decision.md`](results/sparse_atom_bond_dual_stream_seed42/decision.md).

## What the evidence rules out

The run used the accepted 100K/10K roles, seed 42, distance-plus-angle
comparator, training schedule, and sealed-role policy. It completed all 40
epochs. Preflight proved exact shared-backbone initialization, zero residual
injection, finite loss and gradients, and the declared parameter count. The
failure is therefore not attributable to a changed split, early termination,
an inactive optimizer, or accidental nonzero initialization.

Checkpoint inspection performed no model inference. All 88 tensors belonging
to the four bond-attention blocks and shared atom--bond exchange were nonzero
at the candidate's best checkpoint. The four zero-initialized attention output
weights reached RMS values from `0.02697` to `0.02973`; their FFN output weights
reached `0.02195` to `0.02421`. The atom-to-bond and bond-to-atom value weights
reached `0.02783` and `0.03032`. The new path learned; it was not blocked by its
identity initialization.

## Curve-level diagnosis

Across the paired 40-epoch traces, the candidate had lower training MAE in 37
epochs but lower validation MAE in only 15 epochs. At the respective best
validation checkpoints:

| quantity | frozen comparator | dual stream |
|---|---:|---:|
| best epoch | 35 | 37 |
| training MAE (eV) | 0.09983799 | 0.09827208 |
| validation MAE (eV) | 0.13539268 | 0.13646799 |
| validation minus training (eV) | 0.03555469 | 0.03819592 |

The candidate fit the training role more closely while widening the best-point
generalization gap by `0.00264122 eV`. This pattern is evidence of redundant
capacity and weaker generalization, not insufficient capacity or training
horizon.

The frozen comparator already updates persistent real-bond states from atoms
and persistent wedge states in every layer. The candidate's four extra sparse
attention blocks normalized messages over the same non-backtracking wedge
adjacency. They changed the operator but supplied no new molecular relation,
region membership, geometry source, or invariant. More flexible processing of
already represented information explains the lower training error and higher
validation error.

The measured whole-run throughput penalty is real for the accepted run, but it
must not be overinterpreted as a stable kernel cost: the final ten paired
epochs were much closer than the run-wide averages. Accuracy and the wider
generalization gap are sufficient to close the mechanism without relying on a
precise speed attribution.

## Closed retries

The evidence does not support changing learning rate, dropout, number of heads,
exchange rank, width, depth, training length, or seed. Those variants preserve
the same information set and would tune a scientifically failed mechanism.
The sparse atom--bond dual stream remains closed.

## Successor information gate

The next candidate must add a relation that is absent from the atom/bond/wedge
graph, not another processor over those relations. The selected bounded
question is a deterministic smallest-ring hierarchy:

- one node per symmetrized smallest ring;
- explicit atom--ring membership edges;
- explicit fused, spiro, and directly connected ring relations;
- ring composition and conjugation descriptors derived without labels;
- one shared narrow persistent ring update at four exchange points;
- zero-initialized ring-to-atom return path and atom-stream-only pooling.

This is not a retry of the archived structural-adaptor G0 audit. That audit
tested scalar ring/conjugation prevalence in errors from a different frozen
predictor and stopped before model training. It did not create ring nodes,
membership edges, ring--ring relations, or a from-scratch hierarchy on the
official-PCQM 100K roles.

It is also not a graph-token or late-fusion retry. Molecules without rings have
no hierarchy nodes, ring states are local to real structural subgraphs, and
the prediction head receives only the updated atom stream. The corresponding
cache and training contract require a separate protocol before remote work.

## Evidence identities

The machine-readable calculation is
[`results/sparse_atom_bond_dual_stream_seed42/failure_attribution.json`](results/sparse_atom_bond_dual_stream_seed42/failure_attribution.json).
It pins the accepted comparator trace, candidate trace, and candidate checkpoint
hashes. Raw checkpoints and row-level payloads remain in ignored platform
record storage.
