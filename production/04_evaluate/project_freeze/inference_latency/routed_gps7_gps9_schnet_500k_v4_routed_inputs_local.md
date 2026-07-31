# Routed V4 Local Inference Latency

This measures warm end-to-end inference for new SMILES, including graph construction. It is not a precomputed-catalog lookup benchmark.

- Model: `routed_gps7_gps9_schnet_500k_v4`
- Device: `cuda`
- Model load: `4.074 s` (excluded from warm timings)
- Timed repeats per batch: `3`

| Inputs | Median batch s | P95 batch s | Median ms/mol | Molecules/s | Routed fraction | Peak GPU MiB |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0485 | 0.0519 | 48.46 | 20.63 | 1.000 | 41.3 |
| 16 | 0.6582 | 0.6789 | 41.14 | 24.31 | 1.000 | 80.1 |

The repaired-2M dense/equal pure-2D presets are measured separately; this table is the routed-v4 baseline only.
