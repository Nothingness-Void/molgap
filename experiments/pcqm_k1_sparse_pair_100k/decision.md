# Decision: reject K1 topology-conditioned sparse pairs

## Verdict

The seed-42 100K/50K V5 run was mechanically accepted but did not establish a
material or statistically reliable improvement over frozen K1. This mechanism
is closed without another seed, 500K expansion, full training, or protected-role
evaluation.

## Evidence

| Model | Parameters | Best epoch | Development Gap MAE |
|---|---:|---:|---:|
| Frozen K1 | 3,658,817 | 39 | 0.1413736343 eV |
| K1 SparsePair-SPSE proxy | 3,681,233 | 36 | 0.1411096901 eV |

The observed gain was `0.0002639443 eV`, below the predeclared `0.003 eV`
threshold. The paired candidate-minus-K1 bootstrap interval was
`[-0.0011787419, 0.0006223918] eV`; it crosses zero. The probability that the
candidate was better was `0.7144`, which is insufficient evidence for
promotion.

All 40 epochs and 31,240 optimizer steps completed under FP32/no TF32 and
physical BS128. Source, fixed-data, row-order, runtime, checkpoint, prediction,
and recovery hashes passed. No official validation, test-dev, or challenge
role was read.

## Interpretation

The zero-initialized sparse pair return trained correctly and remained cheap,
but it did not improve molecules uniformly. It substantially improved K1-hard
rows while damaging K1-easy rows, leaving a `49.73%` row win rate and only the
small net gain above. A fixed 50:50 average was exploratory-positive at
`0.135340 eV`, but requires two encoder passes and cannot distinguish useful
sparse-pair information from ordinary optimizer-trajectory diversity without
an inert-branch or independent K1 control. It therefore does not reverse the
standalone rejection or release another run.

Machine evidence is in `results/accepted_kaggle_s42_v1/acceptance.json` and
`results/accepted_kaggle_s42_v1/completion_manifest.json`. Remote provenance is
in `results/REMOTE_LOG.md`. The no-inference residual attribution is in
`attribution.md` and
`results/accepted_kaggle_s42_v1/residual_attribution.json`.
