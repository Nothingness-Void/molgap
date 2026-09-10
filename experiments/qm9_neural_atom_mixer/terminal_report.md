# QM9 Neural-Atom mixer seed-42 terminal evidence

- Kernel: `kaseichou/molgap-qm9-neural-atom-s42`, version 1.
- Final Kaggle2 status: `COMPLETE`.
- Read-only status/output commands used the project CLI with `KAGGLE_CONFIG_DIR=C:\Users\Adminn\Desktop`.
- Retrieved output root: `platforms/_records/kaggle/training/qm9_neural_atom_s42_v1`.
- Accepted output root: `platforms/_records/kaggle/training/qm9_neural_atom_s42_v1/qm9_neural_atom_s42`.

## No-model acceptance

Command:

```text
.venv\Scripts\python.exe experiments/qm9_neural_atom_mixer/accept.py --root <accepted-output-root> --source-commit f2d760b584023e53b3b77aef4d9348c21e6b36f4 --cache-sha256 80a2d256ccbbde8de5862e7ad4fd61ea7c1255a19217e590033a94cabe1c6340 --output <accepted-output-root>\acceptance.json
```

- Exit code: `0`.
- Acceptance: `true`.
- Acceptance output records `model_inference_executed=false`, `official_pcqm_roles_read=false`, and `test_role_read=false`.
- No model was run locally by this monitor. The remote preflight's own forward/backward checks are retained in `preflight.json` and are not local inference.

## Frozen identity and contract

- Source commit: `f2d760b584023e53b3b77aef4d9348c21e6b36f4`.
- Cache aggregate SHA-256: `80a2d256ccbbde8de5862e7ad4fd61ea7c1255a19217e590033a94cabe1c6340`.
- Roles: QM9 train 30,000 and internal validation 3,000; split fingerprint `62f1cdefdaec6877`.
- Arms: `full_gps`, `neural_atom_k1`, `neural_atom_k4`.
- Shared task/platform: `qm9-neural-atom-mixer-s42-v1` / `kaggle2` / `Tesla T4`.
- Seed 42, FP32, physical batch 128, no accumulation, AdamW `4e-4`, weight decay `1e-5`, clip 1, cosine40 to `1e-6`, 40 direct-Gap epochs.
- `official_pcqm_roles_read=false`; `test_role_read=false`.

## Remote preflight

- Preflight accepted exact-zero mixer return, zero return projections, finite nonzero return-projection gradients, assignment mass over valid atoms, zero padding mass, and the K1/K4 shared-backbone and initial-state identities.
- Mixer layers: `[3, 6, 9]`; K1 active slots: `1`; K4 active slots: `4`; latent channels: `64`; maximum slots: `4`.
- Full-GPS parameter count: `4,771,073`.
- K1/K4 parameter count: `3,658,817` each (below the 4.2M ceiling).
- Shared backbone SHA-256: `26c241055b745899cf289d4d8a32aa678197cd5b4a802861b8177a551e4e96c9`.
- K1/K4 initial-state SHA-256: `ce878c7a6e71519e4bd25ef10e45e6c9f693687f7f8c76dbf37c66f3c5708dc3`.

## Arm metrics

| arm | best epoch | validation Gap MAE (eV) | parameters | mean epoch (s) | peak memory (MiB) | reserve |
|---|---:|---:|---:|---:|---:|---:|
| full_gps | 38 | 0.12945686280727386 | 4,771,073 | 17.758207325100 | 406.89453125 | 0.9727130459748435 |
| neural_atom_k1 | 38 | 0.12778982520103455 | 3,658,817 | 15.125080092525 | 337.72949218750 | 0.9773513566330102 |
| neural_atom_k4 | 38 | 0.12744142115116120 | 3,658,817 | 14.758360711725 | 364.67236328125 | 0.9755445275203595 |

Stored acceptance arithmetic:

- K4 gain versus full GPS: `0.002015441656112671 eV` (required `0.003 eV`).
- K4 gain versus K1: `0.00034840404987335205 eV` (required `0.001 eV`).
- K4 epoch-time ratio versus full GPS: `0.8310726663757944` (maximum `1.15`).
- Stored `pcqm_transfer_nominated`: `false`.

## Evidence hashes

- Launch manifest SHA-256: `c0b81e58c87ea11c7e226e97b4c58bcb3bb450c51a877a8cbbea33b1fcb01c72`.
- `acceptance.json`: `ae2a204f3f6c76e69cf5c1dc56a761ca5bcabdd9e291327f2e9b2951522b24a5`.
- `metrics.json`: `0d946e5ad19cd6f46c2bcd0106a07c5b44bbdaf8c7ae03ad21ed99758e441b60`.
- `completion_manifest.json`: `4161bd9f8cb4649b6c534270c545aac6ebebd7af7e189d4a689e8ea965a4bc6c`.
- `preflight.json`: `eb458aa1d5bb460c399980c2d6ef0cb15a864578ef569aff8c668e52ae7155c4`.
- `baseline_worker.json`: `d87ebcccaef8415eda55cd0882aa4373b1f8ebb0c832b6bcb3bb9ffa8a3e8a32`.
- `mixer_worker.json`: `df7959a223a2408a3ff5bbdefbe0ce1fe0c72a0aff9712a2fb5382a5b438e499`.
- Kernel log: `60c481aed2f4855f47e019fa335eae31be97beacaf2bd2771d53643de9aa7332`.

| artifact | SHA-256 |
|---|---|
| `full_gps/best_model.pt` | `e00f69c6ee2f7cc46656b9b7bd50d7d4396a1bd47e0e282e5b74d7c74dac184b` |
| `full_gps/best_validation_payload.pt` | `91c01315fb46c24d74b6a72ce1e68864a82ae8c16c74ec6de086a9b8e5e04f4d` |
| `full_gps/last_checkpoint.pt` | `2cc515bcd0dd8c5ccdee1c6252787ba54c1cc56b09ed4996aa03e0a4b88da3ee` |
| `full_gps/trace.json` | `1e8474c1b1140db293a0b1b2ef3057089eade1fc914c113a415f35f207defd09` |
| `neural_atom_k1/best_model.pt` | `29239141bd05f102a4b6ce97344f3686a584c3ba6f50dc2da41c9941702cbd82` |
| `neural_atom_k1/best_validation_payload.pt` | `414c5ddf9d43af04648a970715debb15b2aad080fdb53ff31a3e35b8e728a0c9` |
| `neural_atom_k1/last_checkpoint.pt` | `7815b3953621b64b9337240928c3dbd10954ffcf2a5126b21a914cd3d27a58fb` |
| `neural_atom_k1/trace.json` | `d59395449baea04e2d4c95a36aa960292a12adffc43651784c96645f576d22f4` |
| `neural_atom_k4/best_model.pt` | `126157407bc5fe5e14eb36c860dd96d553708096820e3eae63864501a9ccfe25` |
| `neural_atom_k4/best_validation_payload.pt` | `204b0ca0f18fb0c1aafb9a16e89f84de84c03212c73f2a26bbe16d7848a13a64` |
| `neural_atom_k4/last_checkpoint.pt` | `65576e84dba079441b5163c69f46d674ac2dcfedc654356c7c1461fd087304fe` |
| `neural_atom_k4/trace.json` | `1b4f6069e45b97bc84685c664dea290336c47af2b86977e4ff061020d769f5c0` |

All 16 completion-manifest artifact hashes were recomputed locally and matched (`0` mismatches). No successor, PCQM transfer, seed expansion, official-role access, desktop/SCNet/IMS access, or scientific route decision was made by this monitor.
