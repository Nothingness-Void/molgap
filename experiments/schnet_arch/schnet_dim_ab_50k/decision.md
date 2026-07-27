# SchNet Dimension 50K Screen

## Question

Can the SchNet molecular embedding be narrower than 192 dimensions without
materially degrading the final GPS + SchNet gated fusion model?

## Protocol

- One deterministic random sample of 50,000 graphs from the expansion500K
  ETKDG cache; selection identity is recorded in `graphs_50k_manifest.json`.
- Encoder split: 40,000 train / 5,000 validation / 5,000 test.
- SchNet dimensions 192, 160, and 128 use the same graph rows, seed, optimizer,
  scheduler, 30 epochs, and all other architecture parameters.
- Fusion comparison uses the same aligned GPS embedding, labels, split, and
  192-wide gated head for both the 192D and 160D SchNet embeddings.

## Results

| SchNet width | Encoder test MAE (eV) | Gap MAE (eV) | Parameters | Train time |
|---:|---:|---:|---:|---:|
| 192 | 0.215850 | 0.265747 | 1,041,028 | 1,632 s |
| 160 | 0.217513 | 0.268844 | 873,412 | 1,065 s |
| 128 | 0.231264 | 0.284264 | 722,180 | 993 s |

| Fusion input | HOMO MAE | LUMO MAE | Gap MAE | Average MAE |
|---|---:|---:|---:|---:|
| GPS192 + SchNet192 | 0.076181 | 0.075105 | 0.093336 | 0.081541 |
| GPS192 + SchNet160 | 0.076143 | 0.074767 | 0.093620 | 0.081510 |
| Delta (160 - 192) | -0.000038 | -0.000339 | +0.000283 | -0.000031 |

## Decision

SchNet160 passes the 50K screen. It reduced encoder parameters by 16.1% and
measured training time by 34.7%, while the gated fusion average MAE was tied
within 0.0001 eV and slightly lower in this run. SchNet128 is rejected because
its encoder average MAE degraded by 0.0154 eV.

This is an architecture-screen result, not a production-model promotion.
Inference throughput did not show a measured improvement during embedding
extraction (both runs rounded to 17 seconds), and the 160D candidate still
requires a larger-data replicate plus the fixed common/OOD evaluation before
any registry change.

The follow-up compute-shape search is recorded in
`../schnet_arch_screen_50k/decision.md`; it supersedes SchNet160 as the selected
50K candidate.

## Evidence

- Compact encoder comparison: `summary.json`
- Full encoder metrics: `hidden_*/metrics.json`
- Full fusion metrics: `hidden_192/fusion_metrics.json` and
  `hidden_160/fusion_metrics.json`
- Reproduction command: `scripts/phase8/training/run_schnet_dim_ab.py`
