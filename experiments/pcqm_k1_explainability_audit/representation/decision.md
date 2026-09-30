# Frozen K1 representation diagnostic — 2026-09-30

## Accepted execution and evidence

Kunshan job `123315282` completed with scheduler state `COMPLETED`, exit
`0:0`, 375 allocated seconds on one DCU. All 22 downloaded JSON files matched
remote SHA256. Independent saved-JSON acceptance checked 16 row chunks,
the prospectively frozen 1,024-row panel in both scales, targets, descriptors,
source/checkpoint identities, reproduction tolerances, role events and costs.
The retained [acceptance](results/acceptance.json) and
[analysis](results/analysis.json) contain the complete checks and metrics.

Both checkpoints reproduced their respective 50,000-row historical development
predictions: maximum differences were 0.0000085831 eV for 100K and zero for
500K, with zero reported MAE differences. Remote inference used frozen models,
FP32 and physical BS128. No training or protected-role access occurred.
The acceptance executed no local model inference.

## Findings

On the same fixed500K development panel, the 100K-trained K1 checkpoint had
MAE 0.1429408414 eV and the matched500K checkpoint had MAE 0.1035725083 eV.
This is a contrast of training scales/exposures and selected checkpoints of
one architecture, not a causal architecture comparison or independent test.

| Exchange layer | Mean normalized effective rank after exchange, 100K | 500K | Mean rank change within exchange, 100K | 500K |
|---|---:|---:|---:|---:|
| 3 | 0.588072 | 0.646271 | -0.006712 | -0.002843 |
| 6 | 0.427383 | 0.573886 | -0.012383 | -0.018746 |
| 9 | 0.241644 | 0.274997 | -0.010174 | -0.028988 |

The measured residual exchanges reduced mean normalized effective rank
slightly within each checkpoint, but the 500K checkpoint's retained node
states had higher average rank at every audited layer. These observations do
not support a general cross-scale collapse of local node representations.
Rank is an energy-spectrum statistic, not an information or accuracy metric.

Mean dispersion ratios were 1.040/1.160/1.165 at layers 3/6/9 for 100K and
1.015/1.077/1.223 for 500K. Layer 6's mean update-direction sensitivity fell
from 0.182326 to 0.130940 eV, while its gradient/update cosine rose from
0.119156 to 0.187499. The other layers and all prespecified metrics are retained
in analysis.json; these derivatives are local sensitivities, not deletion
effects or proof of a repair.

All 15 prespecified atom-count-adjusted correlations between representation
changes and absolute-error changes were weak: absolute Pearson correlations
ranged from 0.00656 to 0.08989. No coherent per-row rank/sensitivity-error
association identified a supported intervention.

| Input-visible group | Rows | Mean absolute-error change, 500K minus 100K (eV) |
|---|---:|---:|
| Small, sparse, low conjugation | 66 | -0.068055 |
| High ring fraction | 177 | -0.055312 |
| High RWSE | 225 | -0.037711 |
| Middle control | 602 | -0.033306 |

Groups overlap and are descriptive. In the small/sparse group, layer 9's mean
normalized rank decreased by 0.026682 across scales despite improved error;
rank therefore cannot be treated as a monotonic quality proxy. Layers 3/6
increased in every prespecified group.

## Decision and limits

The outcome was an accepted `NO_TRAIN` / `CONTEXT_ONLY` diagnostic with no
supported representation-collapse repair. It did not select a new architecture,
release a rank regularizer, reopen scalar gates, or authorize more training.
K1's own improvement with more data does not prove that PairToken or other
candidate gains transfer. Earlier frozen-population reversals remain separate
evidence and are not explained by this endpoint contrast.

Two selected checkpoints do not isolate data size, exposure, selection or
training randomness. Linear atom-count adjustment does not control scaffolds
or chemistry. The producer passed source-bound hook invariance, cross-graph
gradient isolation and unchanged-weight assertions; per-batch assertion maxima
were not retained and cannot be independently recomputed from these outputs.

Native process time was 220.688078 seconds (0.061302 DCU-hours); the scheduler
allocated 375 seconds (0.104167 DCU-hours). Total scheduler CPU time was
152.876 seconds. Costs are recorded separately without cross-hardware conversion.

## RML finalization limitation

The original prospective trajectory's A001 `run_ids` was empty before
submission. Its actual run is established by the unchanged submission receipt
and scheduler evidence. The existing RML finalizer requires the terminal run
to already occur in that frozen action, and offers no postlaunch receipt binding.
Consequently this result's scientific acceptance and compact evidence are
retained, but terminal RML publication is blocked. The original prospective
snapshot was not rewritten and no training replay-ready claim was made.
Resolving that metadata-binding gap requires a separate, tested infrastructure
change; it does not require repeating this diagnostic.
