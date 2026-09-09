# Adaptive local denoising decision

Decision date: 2026-09-10.

## Decision

The adaptive atom-local denoising arm was numerically best on the frozen
QM9-30K validation role but failed both predeclared promotion margins. The exact
noise generator, prior, KL weight, 10/30 allocation, architecture, and seed are
closed without PCQM-100K transfer, another seed, or parameter tuning.

This result does not reject local geometry denoising in general. It establishes
a weak same-direction signal from fixed denoising and a smaller additional
signal from adaptive noise, neither large enough to separate the mechanism from
the project's material-effect threshold.

## Accepted comparison

All arms used the same Kaggle2 Tesla T4 task, split fingerprint
`62f1cdefdaec6877`, seed 42, FP32, physical batch 128, initial encoder tensor
hash, optimizer, schedule, 30,000 training graphs, 3,000 validation graphs, and
40 encoder exposures. No QM9 test or official PCQM role was read.

| Arm | Validation Gap MAE | Best Gap epoch |
|---|---:|---:|
| scratch40 | 0.1263165623 eV | 38/40 |
| fixed10_gap30 | 0.1253272146 eV | 29/30 |
| adaptive10_gap30 | **0.1246931255 eV** | 28/30 |

The adaptive improvement over scratch was `0.0016234368 eV`, below the required
`0.003 eV`. Its improvement over fixed denoising was `0.0006340891 eV`, below
the required `0.001 eV`.

## Attribution

Fixed local denoising improved over scratch by `0.0009893477 eV`, while the
adaptive component supplied only another `0.0006340891 eV`. During adaptive
pretraining the learned scale remained close to the fixed prior:
`sigma=0.1005453 +/- 0.0027528`. Its denoising loss was only about `0.00182`
below the fixed arm. The adaptive generator therefore behaved mostly like the
fixed-noise control rather than discovering materially distinct chemical noise
regimes.

All inference models contain 4,891,057 parameters. Mean Gap-training throughput
was 1,038.2, 1,075.0, and 1,112.9 graphs/s for scratch, fixed, and adaptive;
peak Gap-training memory stayed below 529 MiB. The negative decision is based
on effect size, not compute or instability.

## Evidence

- no-model acceptance: `results/gpu_seed42_acceptance.json`
- compact arithmetic: `results/summary.json`
- cache acceptance: `results/cpu_cache_v4_acceptance.json`
- ignored full artifact root:
  `platforms/_records/kaggle/training/qm9_adaptive_denoising_s42_v1`
