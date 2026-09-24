# Decision: frozen 500K GPTrans readout attribution

## Outcome

`INCONCLUSIVE` for the broad claim that the GPTrans readout is the cause of
weak 100K-to-500K transfer. The narrow frozen mean-atom readout probe does not
show a consistent advantage over a similarly sized head using only the
existing virtual state. No 500K encoder retraining, full-scale release, or
official-role access follows from this diagnostic.

## Existing 500K endpoint evidence

All four accepted predictions align on 50,000 internal-development source
indices and identical labels. Relative to GPTrans-T `0.106868 eV`, Pair Update
Norm, Noisy Nodes, and the joint variant gain `0.001140`, `0.001811`, and
`0.002056 eV`, respectively. The joint candidate wins on 51.68% of rows.
Its gain varies by Gap quintile from `0.00443 eV` in the lowest quintile to
`0.00090 eV` in the highest. This is post-hoc subdivision of a development
role used repeatedly for selection, not an independent subgroup finding.

In matched 500K trace observations, the joint candidate leads the baseline
by `0.01847 eV` on live development at zero-based epoch 9, but only
`0.00207 eV` at epoch 59. The corresponding online-training differences are
`0.01345` and `0.00847 eV`. This is compatible with a strong early-learning
advantage that the baseline partly catches, plus limited retention on
development rows. Online train and development metrics have different
measurement semantics; the trace cannot isolate architecture, data scale,
optimizer horizon, or seed variation as the cause.

## Frozen-head falsifier

The accepted baseline and Pair Norm checkpoints were frozen. Their original
predictions were reconstructed on 10,000 fixed internal-development rows with
maximum absolute differences of `1.43e-6` and `1.91e-6 eV`. Small residual
heads used train-role rows `450000:490000`; head selection used train-role
rows `490000:500000`. Only then was the 10K development subset checked.

| Frozen encoder | Original MAE | Virtual width 256 | Virtual width 512 | Virtual + atom mean width 256 |
| --- | ---: | ---: | ---: | ---: |
| GPTrans-T | 0.101699 | **0.100843** | 0.101000 | 0.100852 |
| Pair Update Norm | 0.099974 | 0.099164 | **0.098905** | 0.099138 |

The atom-aware head and wide virtual-only head have similar parameter counts
(`139,777` versus `148,481`). Atom mean minus wide-head gain is `+0.000148
eV` for the baseline with row-bootstrap 95% interval `[-0.000021,
+0.000316]`; it is `-0.000233 eV` for Pair Norm with interval
`[-0.000460,-0.000002]`. The best observed reference-minus-candidate gain
on this 10K subset is `0.002095 eV` after virtual-only width-512 corrections,
still below the historical 0.003 eV point gate. This is not a preregistered
promotion comparison and its bootstrap does not cover training or selection
uncertainty.

## Limits and next decision

The base encoders were trained end-to-end with their original heads. A frozen
mean-atom adapter cannot falsify an end-to-end attentive readout such as DSAR,
nor does this single-seed subset prove a generic GPTrans information ceiling.
The previously staged DSAR 500K route must be reconciled through its own
accepted evidence if it is pursued; the positive DSAR 100K result alone does
not resolve this 500K question. Do not spend another full 500K run solely on
the readout-bottleneck assertion from these data. A future readout experiment
requires a separately frozen causal question, matched comparator and budget.

Machine evidence: `results/prediction_attribution.json`,
`results/full/baseline/extraction.json`,
`results/full/pair_norm/extraction.json`, and
`results/full/heads/head_probe.json`. Local feature/head `.pt` artifacts are
ignored and not portable; the input graph/cache and selected checkpoints are
SHA-pinned in `protocol.md` and the extraction records. Official validation,
test-dev and challenge were untouched.

The two feature extractions and residual-head fitting took `25.557`, `27.038`,
and `10.704` measured local wall seconds, respectively. These are not measured
RTX 5060 device-hours; GPU occupancy/utilization was not separately metered.
