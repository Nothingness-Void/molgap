# Gated Structural GPS 100K Decision

Decision date: 2026-08-25

## Question

This screen replaced each Structural GPS9 GINE local branch with PyG
`ResGatedGraphConv` while retaining RWSE16, global attention, mean pooling,
three-target supervision, the immutable 100K scaffold split, and the fixed
seeds 42/43/44.

## Acceptance

All three independent Kaggle kernels reached terminal `COMPLETE`. Their model,
last checkpoint, metrics, 9,997-row prediction payload, source identity,
targets, finite values, graph/split hashes, and runtime contracts passed local
read-only acceptance. Machine evidence is in `acceptance.json`.

| Seed | Validation delta | Test average delta | Test Gap delta | Training time |
|---:|---:|---:|---:|---:|
| 42 | -0.007340 eV | -0.005732 eV | -0.006966 eV | 2730.93 s |
| 43 | +0.001733 eV | +0.002928 eV | +0.002692 eV | 2747.02 s |
| 44 | -0.005832 eV | -0.004308 eV | -0.006006 eV | 2748.29 s |
| Mean | -0.003813 eV | -0.002371 eV | -0.003427 eV | 2742.08 s |

The three-seed equal-prediction ensemble improved test average MAE from
`0.130774` to `0.127978 eV` and Gap MAE from `0.156568` to `0.152955 eV`.
However, seed 43 regressed on validation and on every test target.

## Decision

The predeclared gate failed because validation direction was not consistent
across all three seeds. The positive mean and ensemble result were retained as
evidence that edge-aware gating can add useful diversity, but they did not
authorize repaired-2M scale-up, production promotion, or external evaluation.

The result isolated seed stability as the blocker rather than compute: all
three runs completed in less than 46 minutes and passed the 75-minute limit.
The architecture was closed as a standalone scale-up candidate and retained as
component evidence for later edge-state designs.
