# QM9 Fourier-Edge seed-42 terminal evidence

- Kernel: `kaseichou/molgap-qm9-fourier-edge-s42`, version 1.
- Final Kaggle2 status: `COMPLETE`.
- Read-only status/output commands used the project CLI with `KAGGLE_CONFIG_DIR=C:\Users\Adminn\Desktop`.
- Retrieved output root: `platforms/_records/kaggle/training/qm9_fourier_edge_s42_v1`.
- Accepted output root: `platforms/_records/kaggle/training/qm9_fourier_edge_s42_v1/qm9_fourier_edge_s42`.

## No-model acceptance

Command:

```text
.venv\Scripts\python.exe experiments/qm9_fourier_edge/accept.py --root <accepted-output-root> --source-commit 50fdaf00abee6a24a7285908dd830da243de33e9 --cache-sha256 80a2d256ccbbde8de5862e7ad4fd61ea7c1255a19217e590033a94cabe1c6340 --output <accepted-output-root>\acceptance.json
```

- Exit code: `0`.
- Acceptance: `true`.
- Acceptance output records `model_inference_executed=false`, `official_pcqm_roles_read=false`, and `test_role_read=false`.
- No model was run locally by this monitor. The remote preflight's forward/backward checks are retained in `preflight.json`.

## Frozen identity and contract

- Source commit: `50fdaf00abee6a24a7285908dd830da243de33e9`.
- Cache aggregate SHA-256: `80a2d256ccbbde8de5862e7ad4fd61ea7c1255a19217e590033a94cabe1c6340`.
- Roles: QM9 train 30,000 and internal validation 3,000; split fingerprint `62f1cdefdaec6877`.
- Arms: `full_gps`, `neural_atom_k1`, `fourier_edge_k1`.
- Shared task/platform: `qm9-fourier-edge-k1-s42-v1` / `kaggle2` / `Tesla T4`.
- Seed 42, FP32, physical batch 128, no accumulation, AdamW `4e-4`, weight decay `1e-5`, clip 1, cosine40 to `1e-6`, 40 direct-Gap epochs.
- Fourier contract: one harmonic (`harmonics=1`), manual equation match, all Fourier gradients nonzero, and shared K1/Fourier state identity accepted.
- `official_pcqm_roles_read=false`; `test_role_read=false`.

## Arm metrics

| arm | best epoch | validation Gap MAE (eV) | parameters | mean epoch (s) | peak memory (MiB) | reserve |
|---|---:|---:|---:|---:|---:|---:|
| full_gps | 38 | 0.12808437645435333 | 4,771,073 | 18.723063583500 | 406.89453125 | 0.9727130459748435 |
| neural_atom_k1 | 39 | 0.12848038971424103 | 3,658,817 | 15.721814742500 | 337.72949218750 | 0.9773513566330102 |
| fourier_edge_k1 | 38 | 0.12419757992029190 | 3,658,241 | 15.815168392125 | 364.15820312500 | 0.9755790078671510 |

Stored acceptance arithmetic:

- Fourier-Edge gain versus full GPS: `0.003886796534061432 eV` (required `0.003 eV`).
- Fourier-Edge gain versus neural-atom K1: `0.004282809793949127 eV` (required `0.001 eV`).
- Fourier-Edge epoch-time ratio versus K1: `1.0059378418556628` (maximum `1.25`).
- Stored `pcqm_transfer_nominated`: `true`.

## Evidence hashes

- Launch manifest SHA-256: `a316aad7ec2da5aff6afe15848b13428fe0b76e40c1691a0f0cf3e07b5cbd341`.
- `acceptance.json`: `65382d542752d56db1f3c0745a0a2767ff21feba82ee05d6daef1b6ac0245259`.
- `metrics.json`: `1c38494271b878ce81a593d5d96c485ed5c4a4d214dfb6fdb0507544de6db3db`.
- `completion_manifest.json`: `a9375dfe5aac6e91e88126376cb702e1ca5cd4026456285df14fa5eeb5de7090`.
- `preflight.json`: `3df28c334160baa069a0dddac09665cffb7862b535073fdd544de12eeccb8725`.
- `anchor_worker.json`: `75b381f7a8ed4730007c79192fd7d581454f1b1cdc24ce6e276eb649c4a49c07`.
- `edge_worker.json`: `d2aeba864d975c312fd9d7595b43ffe64d78da55a897bbd460885137e9a2b124`.
- Kernel log: `50aa1b1448062d660216da041c46b4c93c037dc91c430f48a06c75b9a08c9add`.

| artifact | SHA-256 |
|---|---|
| `full_gps/best_model.pt` | `e1c11517474420cf48434f32d5ceb6e94f4d5b8e04f4ce2baa2ac783710c4c40` |
| `full_gps/best_validation_payload.pt` | `c39dc11a9744e2214cbb4c16d045acd44f5715aacece1b736cc6a8ac8ac60f6d` |
| `full_gps/last_checkpoint.pt` | `d55feed52300f4da04b7800642827fd9999daea208f9177de738c491a66fda36` |
| `full_gps/trace.json` | `bced3a048a7df054e5db49744b89d3f1813a0a86f26e8b86b9988fc6b1dadd98` |
| `neural_atom_k1/best_model.pt` | `c0c7199afaa22dfa96e36811354f17ae7c592f29e6ec2212192881a287367ea8` |
| `neural_atom_k1/best_validation_payload.pt` | `e53e16cff9f44c561123bd6ca9e1f6950d66f32aa40358f24df6a84789e18fe5` |
| `neural_atom_k1/last_checkpoint.pt` | `338999248b57ca6cdb41b7f145e81f48870f09a3892a6cf581864c6e93ba38b3` |
| `neural_atom_k1/trace.json` | `228769a5356507408598d55f6e0af74ea327221de1e9c145a77c140b984d4f31` |
| `fourier_edge_k1/best_model.pt` | `f94ab5b59d5dee0e01a9f915cb57f7a74847b81dd6b11f8f807e4e5d748bb076` |
| `fourier_edge_k1/best_validation_payload.pt` | `4b439635c8cdae0519b60a76fee02ea15c1973f11618579f71c76341c9476349` |
| `fourier_edge_k1/last_checkpoint.pt` | `ae4047d33848564b7996e2875ce39044b97eb4a657ac820965fca4c09ffc6b48` |
| `fourier_edge_k1/trace.json` | `f6db5c1635359cc6a4ab539a8d24e517aa404fc50a6bff6f20245910a1cfec8c` |

All 16 completion-manifest artifact hashes were recomputed locally and matched (`0` mismatches). No successor, PCQM transfer, seed expansion, official-role access, desktop/SCNet/IMS access, or scientific route decision was made by this monitor.
