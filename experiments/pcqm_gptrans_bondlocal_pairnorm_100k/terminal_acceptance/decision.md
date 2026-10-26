# Terminal decision: GPTrans BondLocal vs PairUpdateNorm, 100K

Decision date: 2026-10-27 (Asia/Tokyo)  
Run: `pcqm-gptrans-bondlocal-pairnorm-100k-s42` / Kaggle v1, account `nvoid912`  
Spec: `86b0ec8fd0e6ee380b93f5f6c0fb8f4041713699e3eb0489bce1514d2ddb8840`; source commit `32b96f619404160c9c685ba9bd8d1d79d77921f2`; source archive SHA-256 `dc3f82dc53c867bd561168d795c51962e70b8ecc34eb12992eabf0bf566c6ce0`

## Decision

- `bond_local`: **CLOSED**.
- `pair_update_norm`: **POSITIVE** under the frozen 100K gate. The nominated result is limited to this one seed and this 100K run.
- The primary metric is best-development EMA MAE. The frozen selection role is `development`; the development cohort was used for prediction, label reading, metric computation, and checkpoint selection.
- This decision does not establish independent-test performance, adoption, 500K performance, full-scale qualification, or `READY_FOR_DESKTOP`.

## Primary result and exposure

| Arm | Outcome | Selected EMA pass / step | Selected development MAE (eV) | Process wall to acknowledged endpoint (s) | Exposure |
| --- | --- | ---: | ---: | ---: | --- |
| `bond_local` | CLOSED | 52 / 40612 | 0.14286130827367 | 9870.282847470 | 60 passes; 46860 steps; 5998080 presentations |
| `pair_update_norm` | POSITIVE | 40 / 31240 | 0.13971831759870 | 9911.765227471 | 60 passes; 46860 steps; 5998080 presentations |

Paired development comparison, candidate minus control: **-0.003142990674973 eV**; 95% paired row-bootstrap CI **[-0.004056944928527, -0.002271358706117] eV**, `50000` rows, `2000` resamples, seed `42`. The frozen gate requires delta ≤ −0.003 eV and upper CI < 0 eV; both numerical conditions are met.

## Cost and mechanical status

| Observation | Value | Scope |
| --- | ---: | --- |
| Allocation entry wall | 10282.84574055 s | Two T4s; entry through bootstrap completion, including setup, preflight, training/evaluation and idle assigned GPU time |
| Allocation entry device time | 20565.6914811 T4-device-s (5.712692078 T4-device-h) | Pair total across the two observed GPUs; queue, pre-entry allocation and post-entry release are unmeasured |
| Runtime allocation ledger | 10001.231544228 s wall; 20002.463088456 T4-device-s | Sealed Python bootstrap/runtime window; per-arm device seconds: bond_local 10001.231544228, pair_update_norm 10001.231544228 |

The measured entry wall is within the frozen 13,500 s allocation wall cap. Strict mechanical acceptance is `MECHANICALLY_VERIFIED`; native dependencies are `VERIFIED` with no missing files. Runtime qualification and replay readiness remain `NOT_EVALUATED` in the retained diagnostic and are separate from this 100K scientific decision.

## Evidence

- Paired diagnosis: `D:\molgap-artifacts\retrieved\pcqm-gptrans-bondlocal-pairnorm-100k-s42\v1\strict-diagnosis-after-companion.json` (SHA-256 `29aa9a058a8b197c8c890909108a39d1626a57ef682326aa7a959cd413892645`).
- Physical run identity: `D:\molgap-artifacts\retrieved\pcqm-gptrans-bondlocal-pairnorm-100k-s42\v1\physical_runs.json` (SHA-256 `99e1519f7f79f0f3f920ad912e70ef3e70181b5cbc5c7eb5bcaa24f2f363ade9`).
- `bond_local` canonical trace: `D:\molgap-artifacts\retrieved\pcqm-gptrans-bondlocal-pairnorm-100k-s42\v1\bond_local\canonical_trace.json` (SHA-256 `be5f075e65d814cda49d10572dc7b9dbb7f942ba778f99565f7f13cc30429068`).
- `pair_update_norm` canonical trace: `D:\molgap-artifacts\retrieved\pcqm-gptrans-bondlocal-pairnorm-100k-s42\v1\pair_update_norm\canonical_trace.json` (SHA-256 `03573b79d691d020333de714673dfb1f3d1222a76e5a8efefaaeb3bfa1438ce4`).
- Observed role audits: `roles_bond_local_complete.json` and `roles_pair_update_norm_complete.json`, each bound to its exact run context, A001, trajectory, training membership, and development row identity.
- Cost sources: `D:\molgap-artifacts\retrieved\pcqm-gptrans-bondlocal-pairnorm-100k-s42\v1\execution-metadata\experiment\allocation_entry_observation.json` (SHA-256 `0f8924992315a9b4a2692f96580b20ee6ff93a167354d0673030b70b1103b2ac`), `D:\molgap-artifacts\retrieved\pcqm-gptrans-bondlocal-pairnorm-100k-s42\v1\execution-root\bond_local\allocation_ledger.json` (SHA-256 `3826298311d999d8aade241b69dd10c5afe536ab4a7b3d016502536d84930b4d`), and `D:\molgap-artifacts\retrieved\pcqm-gptrans-bondlocal-pairnorm-100k-s42\v1\execution-root\execution_report.json` (SHA-256 `62aa3d3e4ca232ab95c35d4ecee8709bd213203c062f6ceac5962f312bbc80bb`).
