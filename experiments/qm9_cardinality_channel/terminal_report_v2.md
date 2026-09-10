# QM9 cardinality-channel seed-42 version-2 terminal evidence

- Kernel: `kaseichou/molgap-qm9-cardinality-channel-s42`, version 2.
- Final Kaggle2 status: `ERROR`.
- Read-only status/output commands used the project CLI with `KAGGLE_CONFIG_DIR=C:\Users\Adminn\Desktop`.
- Retrieved evidence root: `platforms/_records/kaggle/training/qm9_cardinality_channel_s42_v2`.
- The frozen acceptance script was **not run**: the paired job is incomplete and has no `metrics.json`, `completion_manifest.json`, or candidate outputs. No local model execution was performed.

## Frozen identity and contract

- Launch manifest: `experiments/qm9_cardinality_channel/results/gpu_seed42_v2_launch.json`.
- Launch manifest SHA-256: `792c14376f8750a8e1d78ee756826a28b971ae7f2ec297c20c59952567c98275`.
- Source commit: `db0c458e1f2ca3df6f077547cc1fc8b07f8937a1`.
- Source tree SHA-256: `1c1003bbe1d50eaf3d240a97abd6d39268f696e53a7b2b2bf32749f7e1d0095b`.
- Cache aggregate SHA-256: `80a2d256ccbbde8de5862e7ad4fd61ea7c1255a19217e590033a94cabe1c6340`.
- Split fingerprint: `62f1cdefdaec6877`.
- Retry class: `infrastructure_preflight_only`; repair: replace bitwise CUDA forward equality with `atol=1e-7`, `rtol=1e-6`.
- The source, data roles, split, seed 42, FP32, physical batch 128, AdamW/schedule, 40 direct-Gap epochs, and `baseline` / `size_control` / `cpa` arm contract were not intentionally changed.
- `official_pcqm_roles_read=false`, `qm9_test_role_read=false`, and `successor_submitted=false` in the launch manifest.

## Mechanical diagnosis

The v2 candidate worker again failed during remote architecture preflight before either candidate arm trained:

```text
RuntimeError: Zero-return identity failed for size_control: max_abs_difference=6.103515625e-05
```

The traceback identifies `src/molgap/qm9_cardinality.py:_preflight` at the repaired `torch.allclose` check. The observed maximum absolute difference (`6.103515625e-05`) exceeds the frozen tolerances (`atol=1e-7`, `rtol=1e-6`), so the candidate worker stopped. The top-level script then reported:

```text
RuntimeError: Cardinality worker failure: [0, 1]
```

This v2 run therefore did not produce a complete paired screen. It is retained as a terminal preflight/implementation failure; this monitor did not alter the source, widen the tolerance, repair, or retry.

## Partial baseline evidence

- Completed epochs: `40` (`ep00` through `ep39`).
- Best epoch: `38`.
- Best validation Gap MAE: `0.1304904967546463 eV`.
- Final logged epoch-39 validation Gap MAE: `0.130753` eV.
- Parameter count: `4,771,073`.
- Mean epoch time: `19.68344855072497 s`.
- Peak memory: `406.89453125 MiB` of `14,911.6875 MiB` (`0.9727130459748435` reserve fraction).
- Last visible log time: `851.199397893 s`.
- `size_control` and `cpa`: no training metrics, checkpoints, payloads, timing, or memory statistics published.
- The partial worker contract records `task_id=qm9-cardinality-channel-s42-v1`; no scientific interpretation is made from this incomplete artifact.

## Retrieved artifact hashes

| artifact | size (bytes) | SHA-256 |
|---|---:|---|
| `molgap-qm9-cardinality-channel-s42.log` | 7,726 | `b22ac1034408259ecccfaaab13e1caabb44cf9f2e7b57ffed76a57f1c0fa348c` |
| `qm9_cardinality_channel_s42/baseline_worker.json` | 1,268 | `093e1f6b512b56b19974caeed12deacb6bd4f8f92cd454afec30e4a74e9937cb` |
| `qm9_cardinality_channel_s42/baseline/best_model.pt` | 19,265,612 | `a56f17f23c561b3183863c633dd37e0890537ac3c02983ce6070fc4bbb88fbc9` |
| `qm9_cardinality_channel_s42/baseline/best_validation_payload.pt` | 26,053 | `4197357c76ad20c5605f61df97ba2944f3c374ef1e4279a0b25037851d396964` |
| `qm9_cardinality_channel_s42/baseline/last_checkpoint.pt` | 57,711,010 | `64ca07cf6e157cf756976b070ba6d09032e32c2b2879818200badf1cf8169620` |
| `qm9_cardinality_channel_s42/baseline/trace.json` | 9,630 | `b49af2774819140934d7d750066f7a4ebbd11174ea6128854cfe1865722c313f` |

Missing terminal artifacts: `candidate_worker.json`, `metrics.json`, `completion_manifest.json`, all `size_control` outputs, and all `cpa` outputs.

No GPU retry, successor submission, PCQM transfer, seed expansion, official-role access, SCNet/desktop/IMS access, or scientific decision was made by this monitor.
