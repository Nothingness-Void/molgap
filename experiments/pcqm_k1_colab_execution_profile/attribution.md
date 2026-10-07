# Attribution — A100 execution diagnostic

## Packaging failure

This was an omitted serialized graph dependency, not a training/model failure.
The prepared sample retained `WedgeData` after geometry stripping; the original
allowlist omitted its defining module. Earlier synthetic checks did not inspect
this actual pickle dependency. The repair used the existing package-only release
import bootstrap with the actual pickle GLOBAL binding, preserving model/data
bytes. `prepare.py` now includes this dependency for future preparation.

## Observed execution

| Case | Median step seconds |
|---|---:|
| single_w2 | 0.156752 |
| double_w0 | 0.274746 |
| double_w2 | 0.267762 |
| double_w4 | 0.268896 |

Double-forward workers2 is 70.82% slower than
single-forward workers2; the single-pass counterfactual reduces per-step time by
41.46%. Workers0 is only
2.61% slower than workers2; workers4 is
0.42% slower than workers2. This is one
ordered bounded sample, not repeated timing trials or a native T4 speed claim.

| Synchronized phase | Mean milliseconds | Share of measured phase sum |
|---|---:|---:|
| loader | 0.480 | 0.18% |
| h2d | 0.477 | 0.18% |
| forward_loss | 100.008 | 36.76% |
| backward | 134.315 | 49.37% |
| clip | 5.514 | 2.03% |
| optimizer | 31.273 | 11.49% |

Forward/loss plus backward dominate the synchronized phase sum. Loader and H2D
waits are small with4096 in-memory graphs. This does not measure ten-shard epoch
loading, paired-worker CPU contention, development evaluation or publication.
The operator table has substantial indexing/scatter and many small matmul calls;
its nested profiler range attributed to AdamW is not an exclusive wall-time
percentage. Use the separate synchronized optimizer phase for that proportion.
No kernel/optimizer equivalence or optimization is established here.

## Accuracy discriminator

Weighted consistency/supervised gradient norm ratios were0.00366,0.00698,
0.01376,0.01111; cosines were0.0548,0.8294,-0.0992,0.1131. Three batches align
weakly/strongly and one conflicts weakly. The selected epoch49 state's consistency
term is small in these batches, with no sustained large opposing signal. This
weakens the specific selected-state interference explanation; it does not prove
that regularization is harmless throughout training or that removing it improves
MAE. No development labels/predictions were consumed.

Scratch BN buffers changed in train mode, as expected. Together with the prior
accepted BN diagnostic this keeps state/calibration as a separate plausible
question; these gradient observations alone do not prove BN caused the500K gap.
Underfitting, insufficient exposure, capacity and missing100K teacher effects
remain `insufficient_evidence` in this execution-only study.

## Next justified decision

Use the already authorized Kaggle paired result to determine the weight0 versus
0.1 accuracy effect under its exact contract. Both arms use two forwards, so it
cannot answer single-pass MAE equivalence. If the regularizer lacks a material
benefit there, a separately frozen single-forward supervised question is the
cheapest direct speed/quality discriminator. Do not launch it from this diagnostic.

## Durability and cost

Drive retains attempt001 failure and attempt002 complete artifacts. Worker wall
was8.719s for the failed attempt and63.702s for the complete attempt; successful
worker body was58.803s. These are allocated A100 worker intervals, not full-session
billing or GPU busy time. Setup wall observations are separate; idle allocation,
queue and local staging/native CPU-child costs remain unknown.
