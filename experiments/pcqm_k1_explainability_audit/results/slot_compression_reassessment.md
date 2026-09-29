# Slot-compression hypothesis reassessment

Date: 2026-09-30. Saved-evidence review only; no training, inference, remote
job, or protected-role access was performed.

## Evidence identity

The review reused accepted jobs `122268096` and `122270044`, documented in
[Round 1](stage2_round1_decision.md) and [Round 2](stage2_round2_decision.md).
The locally retained Round-1 `causal_audit.json` SHA256 was verified as
`d087c7b0cf776fdb4227edec6faed7df35274070581cb228c593d6199b038532`, matching
its completion manifest. These were frozen-checkpoint interventions on the
50,000-row development role, not new architecture-training comparisons.

## What was actually compressed

`NeuralAtomMixer.compute_update` in `src/molgap/qm9_neural_atom.py` pooled
normalized node values into slots and broadcast their transformed content
using the assignment weights. Its forward returned `hidden + update`.

For one active slot at evaluation time, within one molecule:

```text
Delta H = a u^T       rank(Delta H) <= 1
H_out   = H_in + Delta H
```

The bias-free return projection and disabled evaluation dropout make this
factorization exact for that exchange. It describes the communication update,
not the rank of the retained node representation. Subsequent local blocks can
transform it. Calling K1 a rank-one replacement of all node information would
therefore be incorrect.

## Findings and limits

- Round 1 already measured before/after node dispersion, assignment entropy,
  effective atom count, assignment concentration, and update magnitude at all
  three exchanges. Repeating this measurement would not fill an evidence gap.
- At layer 6, mean dispersion increased to 1.58592 times its pre-exchange value
  in the small/sparse/low-conjugation group, versus 1.09045 in the middle
  control. This does not support a blanket node-collapse explanation.
  Dispersion is not information content: an amplified irrelevant direction can
  increase it while useful distinctions remain poorly represented.
- Removing any exchange substantially degraded the frozen model. This shows
  reliance by co-adapted weights, not that an independently retrained model
  without that exchange must be inferior.
- Round 2 found all four tested layer-6 scale changes harmful. This rejected
  those post-hoc scalar repairs; it did not prove that every learned,
  input-dependent communication mechanism must fail.
- The saved diagnostic summaries contain stratum aggregates, not the
  per-molecule intermediate node/edge states needed to establish which
  task-relevant distinctions were lost. They cannot establish a causal
  explanation of 100K-to-500K transfer decay.

## Decision

No duplicate slot-dispersion run, scalar-rescaling run, or new architecture
screen was released by this review. The broad compression diagnosis was
narrowed to an untested question: whether the *content and addressing* of the
rank-one global update omits useful relation distinctions despite preserving
the local-state bypass.

The most informative distinct follow-up would be a frozen-checkpoint,
row-aligned representation diagnostic: measure node/edge/slot representation
effective rank and readout sensitivity around each exchange, and relate them
to already accepted residual strata. Use one immutable row panel and the same
measurements for existing 100K/500K checkpoints where contracts permit.
Rank alone must not be treated as quality, and correlations must not be called
causal effects. Without matched checkpoints, this would remain a within-model
diagnostic rather than an explanation of scale transfer.

Such a follow-up needs a separately frozen diagnostic protocol and retrievable
row-level outputs before compute release. This review was not a new
replay-ready trajectory and did not alter historical scientific outcomes.
