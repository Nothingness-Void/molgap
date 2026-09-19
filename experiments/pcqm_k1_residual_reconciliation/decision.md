# K1 recovered-prediction attribution, 2026-09-19

Two original local Stage-5 K1 prediction copies matched the frozen SHA256
`68fba785a0b028a8445fe8d94d348df2c3a8d7d8caab708acb574d13cac9831b`.
The reconstructed copy had a different hash and was excluded. This resolves
the K1 asset gap recorded by the earlier two-model module-attribution analysis.

The reused analyzer verified all three frozen prediction hashes, the topology
shard hash, 50,000 source indices, exact label alignment and finite predictions.
No encoder inference or training occurred. CPU diagnostic regressions/classifiers
used source-index five-fold cross-fitting. Analysis wall time was 26.30 seconds,
excluding imports and prior discovery. No accelerator or protected role was used.

| Arm | Development MAE (eV) |
|---|---:|
| K1 | 0.104859870 |
| GPTrans-T | 0.106867528 |
| EdgeState | 0.111348806 |

Across 22 graph summaries, the largest absolute univariate correlation with
K1-vs-GPTrans gain was only 0.02624 (heteroatom fraction). Ring and conjugated
bond fractions followed at 0.02464 and 0.02013. The diagnostic winner AUC was
0.53428 and gain regression R2 was 0.001685. The top-1% hard-set Jaccard
overlap was 0.48148; the top-10% overlap was 0.50421. Complementary errors
therefore exist, but these simple graph summaries explain little of them.
This does not rule out every learned router; it fails to justify one here.

## Decision

No training is released. Preserve local EdgeState and the single molecular
slot. Coarse structure routing and added dense attention remain unsupported.
These residual correlations cannot identify a faulty layer, prove missing
geometric information, or establish causal benefits of a new relation module.

PairToken 100K-to-500K mechanism attribution remains incomplete: matching
per-row PairToken and 100K K1 assets were not found in the inspected local
locations. Scalar gain decay alone cannot distinguish redundancy, overfitting,
or different subpopulation effects. Obtain those immutable assets/contracts
before proposing another PairToken variant. Do not duplicate the separately
active server MoSE residual experiment. Full K1 convergence remains pending.

The analysis reused a consumed development role and is exploratory, not a
promotion gate. Existing 100K and 500K protocols also differ in exposure;
row alignment alone would not establish a pure data-scale causal comparison.

Exact results, analyzer hash and absolute input locators: `analysis.json`.
