# K1/PairToken combination transfer decision — 2026-09-23

## Decision

The three frozen saved-prediction rounds leave fixed equal averaging as the
only exploratory combination worth preserving.  The 500K matched60
development endpoint improved over the stronger single K1 model by
`0.0025376673 eV`; all five source-index folds improved.  Fitting a global
weight on 100K and conditioning it on prediction disagreement added no value.
No Molecular Router training or architecture promotion follows from this
analysis.

## Three rounds

| Rule | 100K fit / validation | 500K MAE (eV) | Gain vs stronger single K1 (eV) | Frozen exploratory gate |
|---|---|---:|---:|---|
| R1: equal `0.5/0.5` | fixed in protocol | `0.1010675275` | `0.0025376673` | pass |
| R2: one global weight | K1 weight `0.425`; five-fold OOF `0.1340035632` | `0.1011842855` | `0.0024209093` | fail: `0.0001167580` worse than R1 |
| R3: five disagreement bins | no bin exceeded `0.001 eV` fitting advantage, so all kept `0.425` | `0.1011842855` | `0.0024209093` | fail: no gain over R2 |

The individual matched60 500K endpoints were K1 `0.1036051948 eV` and
PairToken `0.1043037290 eV`.  For R1, the paired 99% **row-bootstrap** gain
interval versus K1 was `[0.0021609101, 0.0029191031] eV`; the five fold gains
were `[0.0022023, 0.0024667, 0.0027063, 0.0027397, 0.0025733] eV`.
The same fixed equal rule on the older 100K role reached `0.1340852096 eV`
versus K1 `0.1413736414 eV` and PairToken `0.1383300447 eV`.

The rule learned from 100K favored PairToken (`K1 weight=0.425`), but the
500K single-model order reversed.  The disagreement rule collapsed to one
global weight in every bin.  These outcomes favor broad residual averaging;
they do not support a chemically specialized molecular switch.

## Scope and cost

Both development roles were previously used for checkpoint or architecture
selection.  The paired row interval excludes training-run variance and
checkpoint-selection uncertainty.  The 100K and 500K endpoint models were
trained on different membership/exposure even though each same-scale pair is
row-aligned.  The result is a transfer diagnostic, not a strict causal or
official leaderboard claim.

All three rounds used saved predictions.  The accepted local audit took
`5.2207364` wall seconds, trained no model and ran no encoder inference.
Deployment would require two encoders; measured inference cost and an unused
evaluation role are absent.  The fixed equal hypothesis can be considered in
a separately frozen confirmation, while learned routing from these signals
remains closed.

Exact input hashes, metrics, 99% intervals, fold gains, learned weights,
role use and timing are in [three_round_transfer.json](results/three_round_transfer.json).
Machine-evidence pointer: `experiments/pcqm_specialist_combination_transfer/results/three_round_transfer.json`.
The mechanical [acceptance](results/acceptance.json) and
[protocol](protocol.md) bind this result to its inputs and frozen rules.
