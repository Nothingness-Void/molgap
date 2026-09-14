# Release decision — 2026-09-14

## Terminal attribution — 2026-09-14

Both arms completed 60 epochs and passed mechanical acceptance with identical
V4 scientific fingerprints. `memory_value` scored 0.1578704618 eV, worse than
the frozen reference by 0.0012432575. `memory_message` scored 0.1582211042,
worse by 0.0015938999. Neither passed the material-gain gate. Direct memory
readback was closed for this bounded contract; no stacking, seed, scale or
optimization rescue was released. Exact hashes, runtime certificates and
saved-payload recomputation: [acceptance](results/acceptance.json).

The final live training MAEs were 0.0986753 and 0.0990157, versus the reference's
0.1037558. Lower training loss did not translate into better selected EMA dev
loss. Duplicated historical content or altered optimization are plausible, but
no measured activation diagnostic identifies either as the cause. Furthermore,
training loss includes training-mode dropout and uses live weights, while dev
uses evaluation-mode EMA weights: subtracting those scores does not isolate a
generalization gap. All best epochs were 59; finite horizon/EMA effects remain.
The two small adverse deltas are below the 0.003 run-variation floor and justify
failure to promote, not a claim of universal mechanism harm.

The experiment took 11,687 seconds (3.25 wall hours, approximately 6.49 T4
device-hours). Both new rounds together used about 12.63 T4 device-hours based
on allocated-device wall time, not an account billing/quota measurement.

## Two-round closure

The four variants all retained 5,246,817 parameters and the frozen reference
training contract. Only Pair PreNorm passed the declared mechanism gate. Its
0.0031248707 gain exceeded the 0.003 floor by just 0.0001248707 eV. It was retained
as a seed42 mechanism shortlist, not a confirmed full-scale replacement or a
claim of K1 superiority. The other three variants were closed without reruns.
[Two-round summary](results/two_round_summary.json) links each accepted record.

The user-authorized two rounds were exhausted. Terminal artifacts were retained,
the controller handoff was consumed, and no third scientific submission or
official-role access was released. Any further work requires user direction.

## Pre-submission rationale

The accepted [relation-flow round](../pcqm_gptrans_relation_flow/decision.md)
shortlisted pair pre-normalization and rejected logit centering. This justified
one distinct information-flow question, not stacking a marginal winner with a
failed arm. [Release facts](results/release.json) bound the remaining authority.

Code inspection found that the GPTrans pair residual stores `P + delta`, while
the direct pair-to-node message pools only `delta`. History still influences
attention logits; it is not absent from the model. The unanswered question is
whether a direct value/readback path from accumulated relations helps beyond
that indirect path. This is neither a new slot nor a repeated shortest-path,
geometry, directed-bond or K1-relation-slot mechanism.

Two isolated variants were selected to distinguish payload and routing:
`memory_value` uses original delta-derived weights on accumulated pair values;
`memory_message` uses accumulated pairs for both weights and values. Both use
the untouched reference's raw pair/logit pathways, not PreNorm or centering.
They retain all seed-42 tensors and parameters. Full V4 optimization is frozen.

If both regress, direct persistent-pair readback closes; no automatic scaling,
gate, normalization, seed or third-round rescue is released. A material gain
nominates a mechanism only. EMA-selected last-epoch behavior and repeated-dev
selection limit causal and generalization claims. The PreNorm result remains
independently shortlisted regardless of this result.
