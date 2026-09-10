# PCQM-100K Fourier-Edge transfer terminal evidence

- Kernel: `kaseichou/molgap-pcqm-fourier-edge-s42`, version 1.
- Final Kaggle2 status: `COMPLETE`.
- Read-only status/output commands used the project CLI with `KAGGLE_CONFIG_DIR=C:\Users\Adminn\Desktop`.
- Retrieved output root: `platforms/_records/kaggle/training/pcqm_fourier_edge_s42_v1`.
- Accepted output root: `platforms/_records/kaggle/training/pcqm_fourier_edge_s42_v1/pcqm_gap100k_fourier_edge_s42`.

## No-model acceptance

Command:

```text
.venv\Scripts\python.exe experiments/pcqm_fourier_edge_transfer/accept.py --root <accepted-output-root> --source-commit 47f99cf9da7fee306f5165175b4020c6c4aa9fb3 --output <accepted-output-root>\acceptance.json
```

- Exit code: `0`.
- Acceptance: `true`.
- Acceptance output records `model_inference_executed=false`, `official_validation_role_read=false`, `test_dev_role_read=false`, and `shadow_audit_read=false`.
- No model was run locally by this monitor. The remote preflight's required forward/backward checks are retained in `preflight.json`.

## Frozen identity and contract

- Source commit: `47f99cf9da7fee306f5165175b4020c6c4aa9fb3`.
- Geometry cache aggregate SHA-256: `3e4206fd239942ab79f9c4978cd4334f6025ba14e8f1cd1df78c20060a0d1d22`.
- Parent graph cache aggregate SHA-256: `eb7c843e33f430ac755bc575d80153aba87677cea1ad5bb0dcf73cca906e2c21`.
- Roles: exactly 100,000 official-train-derived training graphs and 10,000 internal validation graphs; split/data-role fingerprint is the parent graph-cache identity above.
- Pure-2D inputs: `ogb_atom`, `ogb_real_bond`, `rwse16`; geometry attributes were removed before batching.
- Arms: `full_gps`, `neural_atom_k1`, `fourier_edge_k1`.
- Shared task/platform: `pcqm-gap100k-fourier-edge-s42-v1` / `kaggle2` / `Tesla T4`.
- Seed 42, FP32, physical batch 128, no accumulation, AdamW `4e-4`, weight decay `1e-5`, clip 1, cosine40 to `1e-6`, 40 direct-Gap epochs.
- `official_validation_role_read=false`; `test_dev_role_read=false`; `shadow_audit_read=false`.

## Remote preflight

- Preflight accepted finite forward/backward checks, exact one-harmonic manual Fourier equation, nonzero Fourier gradients, and matched shared K1/Fourier initialization.
- Full-GPS parameters: `4,771,073`; neural-atom K1: `3,658,817`; Fourier-edge K1: `3,658,241`.
- Shared K1/Fourier state SHA-256: `f83d4566b2393eea5342d78ec7c39c6768a721093b58be625e201d3ff5b59bb2`.
- Full-GPS initial state SHA-256: `947ef0d4c1d77afe8369e6aa1cfce8128fe8210c6c6501f6ab595532ebe6ec2e`.
- Neural-atom K1 initial state SHA-256: `d5c4fe849ff88634f8324bc3767c72f472068c530819f79435c3934d1ecf14b4`.
- Fourier-edge K1 initial state SHA-256: `079223cf91b3f5b756f43c3734ed9f663809ee0c033910af1a0c2f35591efe83`.

## Arm metrics

| arm | best epoch | validation Gap MAE (eV) | parameters | mean epoch (s) | peak memory (MiB) | reserve |
|---|---:|---:|---:|---:|---:|---:|
| full_gps | 38 | 0.13605749607086182 | 4,771,073 | 75.232738426725 | 599.6767578125 | 0.959784782385461 |
| neural_atom_k1 | 38 | 0.12659579515457153 | 3,658,817 | 67.327547793475 | 466.50244140625 | 0.968715650632683 |
| fourier_edge_k1 | 38 | 0.12695240974426270 | 3,658,241 | 65.588359501350 | 493.5722656250 | 0.966900307854158 |

Stored acceptance arithmetic:

- Fourier-edge gain versus full GPS: `0.009105086326599121 eV` (required `0.003 eV`).
- Fourier-edge gain versus neural-atom K1: `-0.0003566145896911621 eV` (required `0.001 eV`).
- Fourier-edge epoch-time ratio versus K1: `0.9741682513454983` (maximum `1.25`).
- Stored `shadow_audit_authorized`: `false`.

## Evidence hashes

- Launch manifest SHA-256: `5c99f08856b191e182c22485ed93d1e960472e475da57bb3d944250602af7b3f`.
- `acceptance.json`: `b9520ebb36f632303f7664f803e49846ebc49a2025f9ef4360f9d0de53d30340`.
- `metrics.json`: `3d46e9873d38176508a0f2651fd0ad55b58430ee01a53821c7ce5c77f00879ff`.
- `completion_manifest.json`: `dfc1d4c64aa96b2ce635ceefcb94ee1038f9e9a2a7197232f6da30f11ec6ce1b`.
- `preflight.json`: `ef372294d34022a9ab8fdac2092969f4632ffbf135cb735967fd77f8b1ae982d`.
- `anchor_worker.json`: `9f9323aafdeaa9991b7be9d2ded91540cb8534bea4e1e5b267079f652ee0c017`.
- `edge_worker.json`: `38b371c78029d9daa2d61450834608d54a0afbd0ad0fd9f5f26d9a85b455a914`.
- Kernel log: `94e7bdb55279f751d639f5a91e62eacf719fbeda410bf812882384041c4ebb77`.

| artifact | SHA-256 |
|---|---|
| `full_gps/best_model.pt` | `467753c8caa26e3e8d537aa7933e693073fcc45851caf561174d604742d48e6a` |
| `full_gps/best_validation_payload.pt` | `21ceb06b237930dd12b3c63d2b453ef34b7c0ac3cf0462ff094f49e8638ada9b` |
| `full_gps/last_checkpoint.pt` | `52a055a2ac2a387bd8586475ff05547bc00656b33ae013663875aa4931d173ca` |
| `full_gps/trace.json` | `563557bc5f8cf673283b8213edea20290a786eb42f414ad1a901c283eb09ac97` |
| `neural_atom_k1/best_model.pt` | `9e9ac63a9887030dcfbdc795d843cc10e70f8d9dcec7421a13cff98970203784` |
| `neural_atom_k1/best_validation_payload.pt` | `596e7e9d49cb9ad9af4cf42d7bd240057e3c410aa75a22769f19fd3a92f7625f` |
| `neural_atom_k1/last_checkpoint.pt` | `413f2bfeb3de97c8403bd77f9e9d48ff533700d810deca0f50f0aea647f6a73a` |
| `neural_atom_k1/trace.json` | `3321c9a82f1292bd9c7dd96545c9ddff246858a77a73eca0b2714275fe0c409d` |
| `fourier_edge_k1/best_model.pt` | `e56dc991fe7bac3e53e7e89327a78c69078716b5f99790f2a8ca7f89ac87c0b8` |
| `fourier_edge_k1/best_validation_payload.pt` | `8919fd67fa2b091c8604e4bd6c772a773eed5368fbf7b44dafee5bc919e59aac` |
| `fourier_edge_k1/last_checkpoint.pt` | `c6b848b4f571c1832f5d454a6b20efee66e46d2718b55dbdadb324a3c7d2dd6f` |
| `fourier_edge_k1/trace.json` | `5dec3a07ecc4d35c674fad82ed69f4c83aad7075b873f85046c9597429c90242` |

All 16 completion-manifest artifact hashes were recomputed locally and matched (`0` mismatches). No shadow audit, official-role access, successor, seed expansion, desktop/SCNet/IMS access, or scientific route decision was made by this monitor.
