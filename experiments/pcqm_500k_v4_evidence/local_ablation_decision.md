# Matched 500K V4 global-communication ablation, 2026-09-17

Both locally trained arms completed all 60 epochs and passed the frozen
no-inference acceptance checks. They used the accepted 500K/50K row identity,
seed 42, physical BS128, `drop_last`, deterministic FP32/no TF32, 234,360
optimizer steps and 29,998,080 sample presentations. Predictions were aligned
exactly on all 50,000 internal-development source indices and targets. No
official validation, test-dev or test-challenge role was read.

| Arm | Global mechanism | Parameters | Best epoch | MAE |
|---|---|---:|---:|---:|
| Neural-Atom K1 | one molecular slot at 3/6/9 | 3,658,817 | historical V4 | **0.104860 eV** |
| GPTrans-T core | relation-flow global path | 5,246,817 | historical V4 | 0.106868 eV |
| Edge local-only | none | 3,433,601 | 48 | 0.107227 eV |
| Edge sparse-global | dense attention at 3/6/9 | 3,879,425 | 53 | 0.109116 eV |
| EdgeState | dense attention at all 9 layers | 4,771,073 | historical V4 | 0.111349 eV |

## Attribution

The strictly paired local comparison showed that adding dense global attention
at layers 3/6/9 worsened the local backbone by `0.001889 eV` (95% paired
bootstrap interval `0.001309-0.002481 eV`). The aligned comparison against the
historical EdgeState prediction showed a `0.004122 eV` gain from removing
every-layer dense attention (interval `0.003460-0.004776 eV`). Sparse dense
attention recovered part of that loss relative to all-nine-layer EdgeState but
did not clear the predeclared `0.003 eV` materiality floor and remained worse
than no global attention.

K1 beat the local-only backbone by `0.002367 eV` (interval
`0.001787-0.002961 eV`), a favorable but sub-threshold increment. It beat the
sparse dense-attention arm by a material `0.004256 eV` (interval
`0.003651-0.004854 eV`). The arithmetic EdgeState-to-K1 gain therefore
decomposed into `0.004122 eV` from deleting dense every-layer attention and a
further `0.002367 eV` from the low-bandwidth molecular slot. These terms are an
architecture attribution, not independent estimates of population causality.

K1 remained the strongest standalone arm. Dense node-to-node global attention
was closed for this backbone. Any later global mechanism should preserve the
local persistent EdgeState path and use a constrained molecular bottleneck or
relation-flow mechanism rather than restoring dense GPS attention.

## Residual evidence

Five-fold source-index cross-fitting of K1 and GPTrans-T produced
`0.100599 eV`. Adding local-only as a third branch reduced the cross-fitted
development score to `0.099902 eV`, an additional `0.000697 eV`; adding the
sparse-global arm reached `0.100188 eV`. The local-only branch therefore carried
some complementary residual signal, but the increment was below the
materiality floor and reused the architecture-selection role. It did not
authorize a router, a full-scale run or a production change.

## Cross-platform target audit

The two local arms had an identical target-transform fingerprint and form the
strict causal comparison. Their target mean matched the historical Kaggle V4
arms exactly. The recomputed standard deviations differed by only
`1.3322676295501878e-14 eV`, causing at most `1.0836e-13` difference in a
normalized target, but their raw transform hashes consequently differed.
This was recorded as CPU reduction-order noise. Cross-platform scalar and
row-aligned comparisons were scientifically equivalent, but they were not
represented as literal raw-fingerprint identity under the written V4 policy.
Future fixed datasets should publish exact target mean/std constants and hash
that immutable asset instead of recomputing reductions per runtime.

Mechanical evidence is in `local_ablation_acceptance.json`; machine-readable
paired and blend analysis is in `local_ablation_analysis.json`.
