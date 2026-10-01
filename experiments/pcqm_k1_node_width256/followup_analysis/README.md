# Width256 follow-up interpretation

This is a post-hoc interpretation of the accepted width-only screen. It does
not reopen the terminal decision, establish a new gate, fit a predictor or
authorize training. The immutable [decision](../terminal_decision.md) and
[accepted endpoint/trace analysis](../kaggle3_reconciliation_v1/scientific_metrics.json)
remain authoritative. No new molecular cache, checkpoint inference or protected
role was accessed. The descriptive tables reuse the already accepted saved
development predictions; their bins were not prospectively specified.

`describe_saved_results.py` reproduces the tables and `curves.png` using the
original saved-prediction reader. The observed best-epoch selection remains
unchanged. This reader performs no model forward, checkpoint inference or fitting.

## 1. What the added capacity did

The candidate adds 2,376,384 parameters (+64.95%). It changes node embeddings,
local message/MLP channels, node-to-edge input projections, slot key/value input
projections, return projections and the prediction head. It retains edge64,
slot64, one active slot, nine layers, mixer layers3/6/9, RWSE16 and mean pooling.
Width256 therefore adds learned representational capacity, not new chemical
inputs or a larger global slot set.

The final online training MAE is 0.094491 eV versus 0.098231 eV. Its cohort,
dropout and pre-update semantics differ from development evaluation. This is
consistent with stronger fit to training observations, but does not independently
establish overfitting. No fixed-cohort train/evaluation diagnostic was performed.

## 2. The advantage was not stable across exposure

Reference-minus-candidate development gains at matched exposure:

| Epoch | Gain (meV), positive favors256 |
|---|---:|
| 10 | +5.113 |
| 20 | -2.606 |
| 30 | +0.394 |
| 35 | -0.774 |
| 40 | -1.504 |

The frozen selection is the best development checkpoint across all40epochs;
both selected epoch40. Choosing a favorable intermediate comparison would
change that contract. Both minima at the endpoint leave convergence unresolved;
they do not establish that extra training would help the candidate.

## 3. Errors were redistributed

The [descriptive statistics](descriptive_stats.json) verify exact row/target
alignment and prediction hashes inherited from accepted artifacts. Of50,000
rows,24,842 improve, with average absolute-error reduction66.813meV;25,158
worsen, with average increase68.963meV. The weighted effects are-33.195 and
+34.699meV, yielding net+1.504meV error. Mean absolute prediction change is
81.036meV. A small aggregate difference can accompany substantial individual
prediction changes; this does not identify a learnable specialist or router.

| True Gap range | Rows | Candidate minus reference MAE (meV) |
|---|---:|---:|
| <2eV | 25 | +15.393 |
| [2,4)eV | 3,700 | +9.349 |
| [4,6)eV | 34,837 | +1.270 |
| [6,8)eV | 10,451 | -1.372 |
| >=8eV | 987 | +10.443 |

These label bins are descriptive. The tiny<2eV bin is especially uncertain.
Gap bins are not chemical structure classes, and neither mechanism nor a
deployment rule follows from the table. No multiple-comparison inference,
slice selection, calibration or routing was performed.

## 4. A structural fact that width alone leaves unchanged

In `qm9_neural_atom.py`, K1 computes one learned attention weight per atom,
normalized over atoms. In evaluation mode, for one graph:

```
a_i = softmax_over_atoms(q dot W_key LN(h_i) / sqrt(64))
z = slot_seed + sum_i a_i W_value LN(h_i)
u = W_return F(z)
h'_i = h_i + a_i u
```

Thus the instantaneous returned node-by-channel increment is `a u^T`, rank at
most1 in evaluation mode. This does not bound the whole model's rank, its
Jacobian or its total expressiveness: local messages, residual node state and
later nonlinear layers remain. Training return dropout also invalidates a
literal rank-one assertion for each sampled training update.

Four slot self-attention heads do not create four graph slots. With one active
slot there is only one sequence position to attend to. The value projections
and per-head transformations still operate, and atom assignment weights need
not be uniform.

At layer9 the mixer is followed immediately by mean pooling. Because
`sum_i a_i=1`, its exact contribution to the pooled embedding is `u/N`, where
N is the number of atoms. The content u still depends on the molecule. This
is not proof of harmful dilution or of a molecular-size effect. Earlier mixers
at3/6 feed later nonlinear/local computation, so the final-layer expression
does not describe their entire influence.

Widening nodes to256 keeps this single-slot communication structure and
64-dimensional slot/edge state widths. Residual paths preserve node information;
it would be incorrect to claim that the entire network compresses everything
to64dimensions or that edge64 alone sets all message expressiveness.

## 5. What deserves discrimination before another capacity run

The interpretation favors examining how global information is collected,
returned and read out before committing more parameters to nodes. The final
`u/N` term is a specific structural candidate for investigation, not an accepted
failure cause. The cheapest future discriminator would measure the retained
final-slot contribution and its relationship to graph size, with a frozen
diagnostic contract and the correct consumed-role boundaries. This report runs
no such diagnostic and does not propose an unqualified module or successor.

Existing RML was queried before this interpretation. The adjacent K1 PairToken
value-decoupling route already closed after failing its point gates and retaining
strict replay gaps; see [its decision](../../pcqm_k1_pair_value_100k/decision.md).
Consequently a generic recommendation to repeat more pair/value machinery is
not justified here. No K4/SSMA rerun is proposed.

Single-seed variability, the NumPy/package differences, uncontrolled cross-job
timing, and strict RML reference-ID qualification gaps remain as recorded in
the terminal decision. The matched endpoint failure supports retaining192 for
this screen; it does not establish a general information-capacity ceiling.
