# Routed V4 Local Inference Latency

This measures warm end-to-end inference for new SMILES, including graph construction. It is not a precomputed-catalog lookup benchmark.

- Model: `routed_gps7_gps9_schnet_500k_v4`
- Device: `cuda`
- Model load: `4.006 s` (excluded from warm timings)
- Timed repeats per batch: `3`

| Inputs | Median batch s | P95 batch s | Median ms/mol | Molecules/s | Routed fraction | Peak GPU MiB |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0221 | 0.0258 | 22.05 | 45.34 | 0.000 | 40.5 |
| 16 | 0.0932 | 0.1125 | 5.83 | 171.65 | 0.000 | 51.3 |
| 64 | 0.3768 | 0.3790 | 5.89 | 169.87 | 0.000 | 87.9 |

The repaired-2M dense/equal pure-2D presets are measured separately; this table is the routed-v4 baseline only.
