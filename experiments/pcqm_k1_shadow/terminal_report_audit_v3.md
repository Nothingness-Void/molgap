# PCQM K1 shadow audit v3 terminal evidence

- Kernel: `kaseichou/molgap-pcqm-k1-shadow-audit`, version `3`.
- Final Kaggle2 status: `COMPLETE`.
- Read-only status/output commands used `C:\Users\Adminn\Documents\molgap\.venv\Scripts\kaggle.exe` with `KAGGLE_CONFIG_DIR=C:\Users\Adminn\Desktop`.
- Retrieved evidence root: `platforms/_records/kaggle/training/pcqm_k1_shadow_audit_v3`.
- Frozen acceptance command: `.venv\Scripts\python.exe experiments/pcqm_k1_shadow/accept_audit.py --root <artifact-root> --source-commit fdba2e21e5f9441c83c6259908fd8eec99b7b34e`.
- Acceptance result: `accepted=true`; all acceptance checks passed. The acceptance itself performed no model inference or training. The remote audit intentionally read the 10,000 shadow labels; no official validation/test-dev role was read.

## Frozen identity and sealed-role contract

- Launch manifest: `experiments/pcqm_k1_shadow/launch_audit_v3.json`.
- Launch manifest SHA-256: `6D1322109726EB4F2F489E485D825184352EEA087318ED3EB028D05A9EAC2BEC`.
- Source commit: `fdba2e21e5f9441c83c6259908fd8eec99b7b34e`.
- Frozen candidate: `neural_atom_k1`.
- Cache aggregate SHA-256: `4a9e4361247f497f5cd9911a69e69eddb1e0fecea6e33f318ada48aa594f6a44`.
- Shadow-index SHA-256: `f68f0223dccb1c0f80e76035c79efe08b0959e4e14d5b1f7ebb5153d6cbc27bd`.
- Shadow rows: `10,000`; physical batch per device: `128`.
- Training: `model_training_executed=false`; official validation/test-dev roles: `false/false`.

## Shadow audit metrics

The completed remote audit reported the following paired comparison:

| model | shadow Gap MAE (eV) | elapsed (s) | graphs/s | peak memory (bytes) | memory reserve | parameters |
|---|---:|---:|---:|---:|---:|---:|
| `full_gps` | 0.13400164246559143 | 2.8152690580000126 | 3552.0583624444307 | 57875968 | 0.9966074143418566 | 4,771,073 |
| `neural_atom_k1` | 0.12792469561100006 | 1.7804063309999947 | 5616.695372220641 | 53401600 | 0.9968696937858229 | 3,658,817 |

- Paired delta (`neural_atom_k1 - full_gps`): `-0.00607694685459137 eV`.
- Paired bootstrap 95% CI: `[-0.007948308970332146, -0.0042067339545488365] eV`.
- Bootstrap seed/replicates: `2026091142` / `20000`.
- K1 inference-time ratio: `0.6324107196577539` (K1 / full GPS).
- Memory reserve fractions: `0.9966074143418566` (`full_gps`) and `0.9968696937858229` (`neural_atom_k1`).
- Remote audit flags: `shadow_labels_read=true`, `shadow_label_values_accessed=10000`, `model_inference_executed=true`, `training_executed=false`.

The frozen acceptance recomputed the MAEs and paired statistics, matched the completion manifest and all three audit artifacts, and sealed the official validation/test-dev roles. Scientific coordination/decision is deferred to the coordinator; this monitor did not select or submit a successor, 500K run, or any other task.

## Retrieved artifact hashes

| artifact | bytes | SHA-256 |
|---|---:|---|
| `molgap-pcqm-k1-shadow-audit.log` | 3674 | `4C481D70B65BC32CFADA1676DBE14481B2F91C3345EC9FAFEA5AD6F5FC05A53D` |
| `pcqm_k1_shadow_audit/acceptance.json` | 1107 | `A99E791B73BE91A8376CFB3CAA5C97873B7800B8F568C5C5C68065D3B2C14E10` |
| `pcqm_k1_shadow_audit/audit_payload.pt` | 202625 | `863991C4440C405831975705826B67257BE95A28ECCAD8B4F86306AD13E592C3` |
| `pcqm_k1_shadow_audit/completion_manifest.json` | 2226 | `9FA93917BDA9A3A2EA87F16E26340B25EC8BEA360CC95522F0332660E1BAD3D1` |
| `pcqm_k1_shadow_audit/metrics.json` | 1911 | `15B25E41DF8BA8EE9DDDE0156B9B6268B2BB3FF4DCDA2F499B9CD848AFF5F4A6` |
| `pcqm_k1_shadow_audit/predictions_before_label_read.pt` | 162571 | `65C9AE51368FF62C3DADDA41B91C260422D737FAE53D866B0AC2C7FE5C1A326A` |
