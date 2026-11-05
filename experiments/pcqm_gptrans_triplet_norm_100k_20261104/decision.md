# Terminal decision — 2026-11-05

Both candidates close NEGATIVE_UNDER_CONTRACT: neither passes the frozen
3 meV improvement and negative upper paired-interval gate against the retained
PairNorm reference. No adoption, 500K release, additional seed or retry follows.
This disposition rejects promotion under this contract; it does not establish
that the separate-return mechanism is universally harmful.

Kaggle1 job 142747352/version 1, kernel
nothingnessvoid/molgap-triplet-norm-100k-s42-r3, completed both arms.

| Predictor | Best development MAE, eV | Selected pass | Delta vs parent, meV | Paired 95% interval, meV |
|---|---:|---:|---:|---|
| Retained PairNorm reference | 0.139440192079 | 42 | — | — |
| joint_norm | 0.140089593963 | 54 | +0.649402 | [-0.277862, +1.584633] |
| separate_norm | 0.138816206827 | 43 | -0.623985 | [-1.627374, +0.280935] |

Secondary separate-minus-joint contrast is -1.273387 meV with paired 95%
interval [-2.214384, -0.365840]. This favors the separate placement in this
realization, but is not evidence that either candidate beats the parent.
All comparisons use identical 50,000 development source IDs and targets,
2,000 paired-row resamples and seed 42. These intervals condition on one
training seed and a reused selection role; they do not measure seed variance.

Each candidate completes 60 passes, 46,860 updates and 5,998,080 presentations.
The accepted reference is reused; no baseline is retrained. Qualification
and terminal RML are reported separately by the shared strict acceptance.
See [attribution](analysis.md), [retained-parent comparison](parent_comparison.json),
and [direct comparison](direct_comparison.json).
