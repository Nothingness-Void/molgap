# QM9 cardinality-channel seed-42 version-3 terminal evidence

- Kernel: `kaseichou/molgap-qm9-cardinality-channel-s42`, version 3.
- Final Kaggle2 status: `COMPLETE`.
- Read-only status/output commands used the project CLI with `KAGGLE_CONFIG_DIR=C:\Users\Adminn\Desktop`.
- Retrieved output root: `platforms/_records/kaggle/training/qm9_cardinality_channel_s42_v3`.
- Accepted output root: `platforms/_records/kaggle/training/qm9_cardinality_channel_s42_v3/qm9_cardinality_channel_s42`.

## No-model acceptance

Command:

```text
.venv\Scripts\python.exe experiments/qm9_cardinality_channel/accept.py --root <accepted-output-root> --source-commit 9b0393bc9212671cc727a8e23391d7d882c472ff --cache-sha256 80a2d256ccbbde8de5862e7ad4fd61ea7c1255a19217e590033a94cabe1c6340 --output <accepted-output-root>\acceptance.json
```

- Exit code: `0`.
- Acceptance: `true`.
- Acceptance output records `model_inference_executed=false`, `official_pcqm_roles_read=false`, and `test_role_read=false`.
- The remote preflight necessarily records its own model-forward check as `model_inference_executed=true`; no model was run locally by this monitor.

## Frozen identity and comparability

- Source commit: `9b0393bc9212671cc727a8e23391d7d882c472ff`.
- Source tree SHA-256: `1c1003bbe1d50eaf3d240a97abd6d39268f696e53a7b2b2bf32749f7e1d0095b`.
- Cache aggregate SHA-256: `80a2d256ccbbde8de5862e7ad4fd61ea7c1255a19217e590033a94cabe1c6340`.
- Split fingerprint: `62f1cdefdaec6877`.
- Arms: `baseline`, `size_control`, `cpa`.
- Shared task/platform: `qm9-cardinality-channel-s42-v1` / `kaggle2` / `Tesla T4`.
- Seed: `42`; precision: `fp32`; physical batch: `128`; no accumulation.
- Optimizer/schedule: AdamW `4e-4`, weight decay `1e-5`, clip `1`, cosine40 to `1e-6`.
- Exposure: `qm9-train30000-gap40`; each arm completed 40 epochs.
- Preflight: exact-zero new return, zero output projection, shared state hash, finite nonzero return gradient, exact K<=3 support, channel layers `[3,6,9]`; all accepted.
- `official_pcqm_roles_read=false`; `test_role_read=false`.

## Arm metrics

| arm | best epoch | validation Gap MAE (eV) | parameters | mean epoch (s) | peak memory (MiB) | reserve |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 38 | 0.12806478142738342 | 4,771,073 | 18.105581655675 | 406.89453125 | 0.9727130459748435 |
| size_control | 39 | 0.12892957031726837 | 4,845,957 | 20.720047994475 | 414.99267578125 | 0.9721699723266565 |
| cpa | 38 | 0.12822422385215760 | 4,845,957 | 20.023011480000 | 415.18603515625 | 0.9721570053586323 |

Stored acceptance arithmetic:

- CPA gain versus baseline: `-0.00015944242477416992 eV`.
- CPA gain versus size control: `0.0007053464651107788 eV`.
- Required gains: `0.003 eV` and `0.001 eV` respectively.
- CPA epoch-time ratio: `1.1059026912689116` (maximum `1.35`).
- Stored `pcqm_transfer_nominated`: `false`.

## Evidence hashes

- v3 launch manifest SHA-256: `1982c5ec04a2b41c7985f16f77b51dcfe6318927d2abe64a9729ae2b3c2a0400`.
- `acceptance.json`: `cb06353c53dc5a3f7f04cbd43c3fd69d2404a6f1eb2e368a5c75674c9f77c5dd`.

| artifact | SHA-256 |
|---|---|
| `metrics.json` | `f02a84b4bfbf8b7090f62f5f5920e02c17d223b0e06e3b33bdcba6dd4be19a36` |
| `completion_manifest.json` | `b7fcf3c9ddc6b52094c717fdc9693c49adac3e52eed56e0fe5bed619e6af88db` |
| `preflight.json` | `76e56038a1ef54be0ba2ea011bf7e11211ea3fbd2e0f903e5492f0d140f2e8ba` |
| `baseline_worker.json` | `4f90d3639ac9643aee72e6e94478fa561b6497010d6fc9990a682f104729a9e6` |
| `candidate_worker.json` | `86fa24e65f16c2b4d75c414e42f6f80074b563374280e97332de762d219ce49e` |
| `baseline/best_model.pt` | `f9e9fd208a41ff14b939ce55aec373d41a9b69ff20c336ead1464867f014158b` |
| `baseline/best_validation_payload.pt` | `22ed81348c16cc39c77b15baf643a10bf2b11728a8ba15f0b84e9fe9dc7cfa02` |
| `baseline/last_checkpoint.pt` | `977102032348ee348aa8a2179bf443f896b1f4659329bbbaf7f865683d3f2006` |
| `baseline/trace.json` | `f86fc815aba0389181378211f2a3ba6bf73cc6559b316ced08832b7afacfe711` |
| `size_control/best_model.pt` | `5344a6793c5069de31c363b5c2ce6a4035f36a241707fa222786b78c9aeffacc` |
| `size_control/best_validation_payload.pt` | `f1287a1ac0454da4629c6d16c0678aa7772dff29c5e20d5cc60e9fc20d4f896b` |
| `size_control/last_checkpoint.pt` | `d1600fb0e1d05de4e6599a1b56ca9060168c4d66cdb86d8f53073efa1f82000e` |
| `size_control/trace.json` | `6b5bc95611fe34e1aed65a371353f6d68ec759841e408060383ec4a15efdc9b9` |
| `cpa/best_model.pt` | `fbd0c555b151260b6b582770135415c66518350574ccd462714734a4b61df334` |
| `cpa/best_validation_payload.pt` | `5a8c63e92e5f9cf8ef7986568c2ebe8e355a2ff71ae115478478f3a698f1bbc0` |
| `cpa/last_checkpoint.pt` | `5c354c01ead703b4421ef74337e0e6894d4fe1651da39c890650b2973e57ff9c` |
| `cpa/trace.json` | `41c8e3c33244f2f2bede0f4cb4b310b22d133cdf5b1aa340b0faf637ef221c2f` |
| `molgap-qm9-cardinality-channel-s42.log` | `d8b731d5bce73d066355ce067055fd851b245ae20a0f1e8453333479f809ee16` |

All completion-manifest artifact hashes were recomputed locally and matched (`0` mismatches). No successor, PCQM transfer, seed expansion, official-role access, SCNet/desktop/IMS access, or scientific route decision was made by this monitor.
