# Frozen 100K checkpoints on the accepted 500K development role

Question: Did PairToken's 100K advantage disappear merely because the 500K
development molecules differ, or only after the models were trained at 500K?
This is a server-owned, prospective `NO_TRAIN` diagnostic. It is not an
architecture promotion, training comparison, independent test, or official
validation claim.

Inputs are immutable: the accepted PCQM fixed 100K and 500K graph manifests,
the 100K K1 and PairToken best-model weights and 50K development payloads,
and the 100K training-only target transform. Exact SHA-256 values are checked
by `src/molgap/pcqm_k1_cross_scale_diagnostic.py`. Inference is FP32,
TF32 disabled, physical batch 128, one allocated Kunshan DCU. No optimizer,
weight update, new seed, cache build, or target fitting occurs.

The first gate reproduces both frozen checkpoints on the original 100K
development role (source indices 100000–149999): target/source indices exact,
maximum absolute prediction discrepancy at most 0.001 eV, and full-role MAE
discrepancy at most 0.0001 eV. If either model fails, stop without opening the
500K development shard. The separate 500K development role consists of source
indices 500000–549999, disjoint from both 100K and 500K training prefixes.
Never evaluate a 500K-trained model on the 100K development role: those rows
were included in its training role.

The predeclared comparison matrix is:

| Training prefix | Evaluation role | K1 / PairToken |
|---|---|---|
| 100K | 100K development | existing result and reproduction gate |
| 100K | 500K development | this diagnostic |
| 500K | 500K development | accepted matched60-v4 results, no rerun |

Measure paired row gain (K1 absolute error minus PairToken absolute error),
bootstrap 95% CI, prediction shift and disagreement, and strata based only on
input-visible graph descriptors. Report the contrast between the 100K-trained
pair's gain on each development distribution, then the contrast with the
500K-trained pair on the *same* 500K development distribution. These contrasts
are diagnostic, not a unique causal attribution: checkpoint selection and
training stochasticity remain. The 100K and 500K development roles were used
previously, so neither is an untouched generalization test. A small delta
must not be promoted as a new winner.

All output is atomically saved in independently retrievable 5K-row chunks,
with progress and terminal SHA-256 manifests. Only the two named development
shards may be read. Official validation, test-dev, and test-challenge flags
must remain false. One preflight DCU job precedes the full bounded job. Stop
on any input hash, reproduction, role, row, or finite-value failure; do not
relax this contract to obtain a result. No automated successor training.
