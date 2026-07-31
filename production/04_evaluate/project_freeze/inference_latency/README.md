# Local Inference Latency

This directory contains reproducible local latency evidence required by
`PACKAGE-A`. It distinguishes a new-SMILES model pipeline from a query against
a precomputed catalog; the latter is a database benchmark and is not measured
here.

All records were produced on the same local machine (Ryzen 9700X, RTX 5060 8 GB,
CUDA) with warm caches, one warmup batch, and three timed repeats per size.

## Runners

Registered routed-v4 baseline:

```powershell
.venv\Scripts\python.exe production\04_evaluate\scripts\evaluation\benchmark_routed_v4_latency.py
```

The default small-molecule suite may not cross the routed-v4 `Gap < 4 eV`
threshold. Measure the conditional GPS9 path separately with the fixed
accepted-prediction-derived suite:

```powershell
.venv\Scripts\python.exe production\04_evaluate\scripts\evaluation\benchmark_routed_v4_latency.py `
  --smiles-file production\04_evaluate\project_freeze\inference_latency\routed_v4_trigger_smiles.smi `
  --output production\04_evaluate\project_freeze\inference_latency\routed_gps7_gps9_schnet_500k_v4_routed_inputs_local.json
```

Frozen repaired-2M pure-2D presets:

```powershell
.venv\Scripts\python.exe production\04_evaluate\scripts\evaluation\benchmark_repaired_2m_latency.py --key repaired_2m_dense_2d
.venv\Scripts\python.exe production\04_evaluate\scripts\evaluation\benchmark_repaired_2m_latency.py --key repaired_2m_equal_2d
```

Each runner records warm end-to-end new-SMILES latency, throughput, GPU peak
memory, hardware metadata, input-suite hash, and every checkpoint hash, plus a
Markdown companion.

## Encoder-Pass Accounting

| Model | 2D graph | ETKDG conformer | GPS passes | SchNet passes | Gate/fusion | Parameters |
|---|---|---|---:|---:|---|---:|
| Routed-v4 500K, base only | yes | yes | 1 | 1 | 1 fusion head | 8,140,432 |
| Routed-v4 500K, routed row | yes | yes | 2 | 1 | 2 fusion heads | 8,140,432 |
| Repaired-2M GPS7/GPS9 equal | yes | no | 2 | 0 | fixed equal average | 6,654,726 |
| Repaired-2M three-GPS dense | yes | no | 3 | 0 | 3 dense-gate seeds | 9,841,380 |

Parameter counts are the sum of all loaded checkpoints for that path. Routed-v4
loads the conditional GPS9 whether or not a given row triggers it, so its count
is the same in both rows; only its per-row compute differs.

The decisive cost difference is not encoder count. Both repaired-2M presets are
pure 2D and skip ETKDGv3 conformer generation, which dominates routed-v4's warm
per-molecule time at batch sizes above one.

## Measured Latency

Median warm values from the records in this directory:

| Model | 1 mol (ms) | 16 mol (ms/mol) | 64 mol (ms/mol) | 64 mol throughput | Peak GPU MiB at 64 |
|---|---:|---:|---:|---:|---:|
| Routed-v4 500K, default suite (routed fraction 0.000) | 22.05 | 5.83 | 5.89 | 169.9 mol/s | 87.9 |
| Routed-v4 500K, forced-route suite (routed fraction 1.000) | 48.46 | 41.14 | not measured | 24.3 mol/s at 16 | 80.1 at 16 |
| Repaired-2M GPS7/GPS9 equal | 27.43 | 1.78 | 0.56 | 1,793.0 mol/s | 40.1 |
| Repaired-2M three-GPS dense | 46.06 | 3.19 | 0.88 | 1,139.9 mol/s | 52.4 |

At batch 64 the equal preset is `10.5x` and the dense preset `6.7x` the
routed-v4 default-suite throughput, despite the dense preset running three GPS
encoders instead of one. Single-molecule latency is dominated by fixed per-call
overhead and does not favor either family.

The forced-route suite is a worst case for routed-v4 by construction: every row
crosses the `Gap < 4 eV` threshold, so the conditional GPS9 and dual fusion run
for all of them. It was not measured at batch 64.

## Records

| File | Path |
|---|---|
| Routed-v4 default suite | `routed_gps7_gps9_schnet_500k_v4_local.json` / `.md` |
| Routed-v4 forced-route suite | `routed_gps7_gps9_schnet_500k_v4_routed_inputs_local.json` / `.md` |
| Repaired-2M dense | `repaired_2m_dense_2d_local.json` / `.md` |
| Repaired-2M equal | `repaired_2m_equal_2d_local.json` / `.md` |

Accuracy comparison for the same models is in
`../track_a_final_decision.md`. Public-path correctness evidence is in
`../public_inference_consistency/` and `../public_api_smoke_test/`.
