# QM9 cardinality-channel seed-42 terminal evidence

- Kernel: `kaseichou/molgap-qm9-cardinality-channel-s42`, version 1.
- Final Kaggle2 status: `ERROR`.
- Read-only status/output commands used the project CLI with `KAGGLE_CONFIG_DIR=C:\Users\Adminn\Desktop`.
- Retrieved evidence root: `platforms/_records/kaggle/training/qm9_cardinality_channel_s42_v1`.
- The frozen acceptance script was **not run**: the job is incomplete and has no `metrics.json`, `completion_manifest.json`, or candidate outputs. No local model execution was performed.

## Frozen identity and contract

- Launch manifest: `experiments/qm9_cardinality_channel/results/gpu_seed42_launch.json`.
- Launch manifest SHA-256: `fb580dfcae94a27b52e36c2da5b1724844b03bf2522a7322ea276ee53e901097`.
- Source commit: `f40e26e525a12efb065b1c9074a381f68bf6f15f`.
- Cache aggregate SHA-256: `80a2d256ccbbde8de5862e7ad4fd61ea7c1255a19217e590033a94cabe1c6340`.
- Source tree/cache identity, seed 42, FP32, physical batch 128, AdamW `4e-4` / weight decay `1e-5`, cosine schedule to `1e-6`, 40 direct-Gap epochs, and the `baseline` / `size_control` / `cpa` arm contract were not changed by this monitor.
- `official_pcqm_roles_read=false`, `test_role_read=false` in the frozen launch contract.

## Mechanical diagnosis

The candidate worker failed during the remote architecture preflight before either candidate arm trained:

```text
RuntimeError: Zero-return identity failed for size_control
```

The traceback identifies `src/molgap/qm9_cardinality.py:_preflight` at the `torch.equal(output, baseline_output)` zero-return identity check. The top-level script then reported:

```text
RuntimeError: Cardinality worker failure: [0, 1]
```

The baseline worker independently reached 40 epochs and published partial artifacts, but the candidate worker did not publish `candidate_worker.json`; therefore the paired screen has no complete three-arm result and no scientific comparison or transfer nomination is valid. This is retained as a terminal implementation/preflight failure. Per the monitoring contract, it is not repaired or retried here.

## Partial baseline evidence

- Completed epochs: `40` (`ep00` through `ep39`).
- Best epoch: `38`.
- Best validation Gap MAE: `0.13004589080810547 eV`.
- Final logged epoch-39 validation Gap MAE: `0.130285635590553 eV`.
- Parameter count: `4,771,073`.
- Mean epoch time: `17.266737164350005 s`.
- Peak memory: `406.89453125 MiB` of `14,911.6875 MiB` (`0.9727130459748435` reserve fraction).
- Last visible log time: `754.49921073 s`.
- `size_control` and `cpa`: no training metrics, checkpoints, or payloads published.

## Retrieved artifact hashes

| artifact | size (bytes) | SHA-256 |
|---|---:|---|
| `molgap-qm9-cardinality-channel-s42.log` | 7,737 | `31261209f4fb5c3bd0886b439b33e8da8756785cf91166984cdba4d8d96c9e49` |
| `qm9_cardinality_channel_s42/baseline_worker.json` | 1,270 | `541aaa2e3317c9371c477483155bae47d6e1afaf8e752c29cf0ebd614ebc383d` |
| `qm9_cardinality_channel_s42/baseline/best_model.pt` | 19,265,612 | `b6fdd13f4debd897b677c5a1d0eccd7d01076ce295b7dae5db72302386dbcfb0` |
| `qm9_cardinality_channel_s42/baseline/best_validation_payload.pt` | 26,053 | `df479e68394dd0d6b796a1a6c6944cc849fb323671e7686f4c26e603b157280b` |
| `qm9_cardinality_channel_s42/baseline/last_checkpoint.pt` | 57,711,010 | `da4d552b39b3fe0b774e376fe070e2cf52eccdc2f44ebcd6bd03e91f34e5bbf7` |
| `qm9_cardinality_channel_s42/baseline/trace.json` | 9,625 | `33ba2300761974ece137128ebdbc1cadbc515fd3bb48b0d78041d24015c23ed1` |

Missing terminal artifacts: `candidate_worker.json`, `metrics.json`, `completion_manifest.json`, all `size_control` outputs, and all `cpa` outputs.

No GPU retry, successor submission, PCQM transfer, seed expansion, official-role access, SCNet/desktop/IMS access, or scientific decision was made by this monitor.
