# Frozen relation-dependency diagnostic — September 27, 2026

## Decision

The three accepted checkpoints depended on their learned relation computations,
but this diagnostic established no new portable architecture winner. Halving,
disabling, or selectively ablating the predeclared updates worsened both internal
development endpoints. No coefficient, router, checkpoint, extra seed, scale-up,
or training successor was selected or released.

This was a separately planned **NO_TRAIN** inference intervention, not a
from-scratch causal architecture comparison or a training-prefix replay entry.
The preceding [training/portability decision](../audit/decision.md) was unchanged.

## Acceptance and identity

Actual private Kaggle2 kernel
`kaseichou/molgap-k1-frozen-relation-diagnostic-s42`, **136078071/v1**, completed.
The [submission receipt](submission_receipt_v1.json) binds that title-derived
slug to the frozen embedded logical identity without `frozen`; these were one
physical run. All three workers executed zero optimizer steps with unchanged
checkpoint weights, FP32, physical BS128 and TF32 disabled. Full-mode predictions
reproduced their accepted payloads exactly on both roles (maximum difference
**0 eV**), before interventions.

[Acceptance](results/acceptance.json), [execution](results/execution.json), and
the three worker terminals bind the frozen source, checkpoints, all atomic
prediction chunks, runtime and role observations. Analysis only read saved
tensors locally; it did not construct or run a model. Official validation,
test-dev and test-challenge were untouched. The two roles each contained 50,000
rows: fixed100K internal dev `[100000,150000)` and fixed500K internal dev
`[500000,550000)`. The latter was already consumed by the preceding audit; this
was neither a new sealed confirmation nor training on 500K molecules.

Observed wall time was **730.402750175 seconds**; two Tesla T4 allocations cost
**1,460.80550035 allocated-device seconds (0.405779306 hours)**, including setup
and idle allocation, below the 5,400-second cap. This describes occupancy, not
Kaggle billing. Worker process CPU times were retained; complete allocation CPU
and queue times were unavailable, not zero.

## What the interventions established

All values below are eV, recomputed in float64 from retained FP32 predictions.
Each cell lists original100K internal dev / fixed500K internal dev. Positive
delta means worse. Complete conditions, descriptors, diagnostics and paired
intervals are in [analysis](results/analysis.json).

| Model | Full minus frozen K1 | Targeted ablation minus full | Half-update minus full | Whole-branch off minus full |
|---|---:|---:|---:|---:|
| Receiver-pair | -0.001866 / +0.002190 | +0.040561 / +0.029976 | +0.056913 / +0.046941 | +0.190980 / +0.183851 |
| Triplet-aggregate | -0.001516 / +0.001942 | +0.010829 / +0.009338 | +0.047961 / +0.047681 | +0.163367 / +0.178761 |
| RRWP-pair | -0.002449 / +0.003576 | +0.004146 / +0.005083 | +0.074514 / +0.064284 | +0.235967 / +0.216744 |

- Receiver `common_return` preserved each graph's mean update but removed
  receiver differences. Its regression supported useful receiver-specific
  computation within that checkpoint.
- `triplet_off` removed the learned aggregate projection including bias,
  preserving pair normalization and receiver processing. `rrwp_off` removed
  the complete walk-relative projection including bias. Both harmed their
  trained networks; these paths were active, not unused implementation branches.
- Every non-full condition also worsened each predeclared low/middle/high
  conjugation group on both roles, with descriptive paired 95% intervals above
  zero. Neither simple attenuation nor the targeted deletion repaired the
  observed low-conjugation weakness.

The off networks were **not K1**: the remaining weights had co-adapted to their
added branches. A large knockout loss did not measure an equally large training
gain or identify a deployable model. The intervals used 2,000 paired row
bootstrap replicates; they were not seed uncertainty, multiplicity-adjusted
confirmation, or proof of a universal mechanism.

## Net benefit and the most coherent surviving signal

The retained matched [parent comparisons](../audit/results/portability_analysis.json)
were essential to interpret knockout results:

- Triplet versus Receiver was +0.000350 on original dev and -0.000248 on
  fixed500K dev; both paired intervals crossed zero. Its incremental net
  benefit was unresolved despite dependence on its triplet branch.
- RRWP versus Receiver was -0.000584 on original dev (interval crossed zero),
  then +0.001386 on fixed500K dev (interval entirely unfavorable). Added walk
  information did not yield a portable increment in this matched screen.
- Receiver versus K1 had the clearest localized signal: high-conjugation
  groups improved by 0.002639 and 0.005306 on the respective roles, with both
  paired intervals favorable. However, low-conjugation fixed500K rows regressed
  by 0.010517, and the overall model regressed. The group boundaries came from
  previously examined data, so this was an explanatory hypothesis, not a
  specialist/router qualification or a promotion.

Receiver update RMS was 0.4606 on original dev and 0.4563 on fixed500K dev;
update/hidden RMS was 0.5228 and 0.5426. These means did not show a simple
overall update-amplitude explosion. They did not exclude rare-tail effects,
representation changes or compensating hidden-state behavior.

## Attribution and stop

The evidence ruled against two narrow explanations for these runs: inactive
new branches, and an immediately repairable excess update amplitude at frozen
inference. It was consistent with co-adaptation and a nonportable allocation of
benefits across molecules, but did not separate optimization, coverage,
representation interference and repeated-development-selection effects.

The ranking reversal had already occurred with the **same 100K-trained weights**
on different internal molecules. Reduced exposure during a hypothetical 500K
training run was therefore not necessary to cause this observed failure. This
did not establish how a properly matched 500K training run would converge;
extra epochs alone were not justified by this diagnostic.

Receiver-specific communication remained a narrowly supported research signal;
Triplet and RRWP were not established as portable incremental improvements.
Any later experiment would need a new hypothesis distinguishing coverage or
co-adaptation from raw capacity, explicit authority, and a prospectively frozen
evaluation. This terminal decision released none. The existing Luna handoff
was closed with no successor; the earlier three training replay entries were
preserved and this diagnostic added only its independent NO_TRAIN evidence.
