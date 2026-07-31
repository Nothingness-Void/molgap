# Local Inference Latency

This directory contains reproducible local latency evidence required by
`PACKAGE-A`. It distinguishes a new-SMILES model pipeline from a query against
a precomputed catalog; the latter is a database benchmark and is not measured
here.

Run the registered routed-v4 baseline:

```powershell
.venv\Scripts\python.exe production\04_evaluate\scripts\evaluation\benchmark_routed_v4_latency.py
```

The runner records warm end-to-end new-SMILES latency, throughput, routed
fraction, GPU peak memory, hardware metadata, input-suite hash, and all five
checkpoint hashes. Its default output is
`routed_gps7_gps9_schnet_500k_v4_local.json` plus a Markdown table.

The default small-molecule suite may not cross the routed-v4 `Gap < 4 eV`
threshold. Measure the conditional GPS9 path separately with the fixed
accepted-prediction-derived suite:

```powershell
.venv\Scripts\python.exe production\04_evaluate\scripts\evaluation\benchmark_routed_v4_latency.py `
  --smiles-file production\04_evaluate\project_freeze\inference_latency\routed_v4_trigger_smiles.smi `
  --output production\04_evaluate\project_freeze\inference_latency\routed_gps7_gps9_schnet_500k_v4_routed_inputs_local.json
```

The repaired-2M dense and equal models require their accepted local checkpoints
and a public inference loader before they can be measured and compared here.
