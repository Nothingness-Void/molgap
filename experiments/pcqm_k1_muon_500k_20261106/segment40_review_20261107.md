# Verified second segment — 2026-11-07 JST

Both arms reached the planned 40-pass pause and passed existing build-resume
verification. Muon remains ahead, with a smaller selected difference than at
20 passes. The complete 60-pass scientific gate remains pending.

| Observation | AdamW | Muon |
|---|---:|---:|
| Completed passes / optimizer steps | 40 / 156240 | 40 / 156240 |
| Sample presentations | 19998720 | 19998720 |
| Selected pass / step | 40 / 156240 | 36 / 140616 |
| Selected Live development MAE, eV | 0.106885798275 | 0.102467745543 |
| Pass40 Live development MAE, eV | 0.106885798275 | 0.103142537177 |
| Current invocation process wall, seconds | 11994.112731125 | 26593.988709297 |
| Cumulative process wall, seconds | 24372.356259493 | 53780.476644663 |
| Existing build-resume verification | RESUME_VERIFIED | RESUME_VERIFIED |

Selected Muon-minus-AdamW is -4.418053 meV; pass40 is -3.743261 meV;
the mean matched difference at passes36–40 is -4.446688 meV. Selected states
have different passes. No paired interval or terminal qualification was computed.
Both traces contain 40 observations; their first 20 observations match the
retained version1 traces field for field. Both cursors have next_batch=0 and
pending_rows=0. Their original optimizer/recipe identities remain distinct and
unchanged; this run does not use the separate accelerated optimizer question.

## Actual cost and retained state

The shared physical allocation ledger records version2 wall26947.714836701s
and allocated T4 53895.429673402 device-seconds, with exactly one prior segment:
version1 wall27515.005689762s and allocated T4 55030.011379524 device-seconds.
Count each physical segment once, not once per arm, and do not add a nested
prior ledger again. Total: 15.128533 entry-wall hours and 30.257067 allocated
T4-device-hours. Before version3, the contract retained 11.871467 entry-wall
hours, 23.742933 T4-device-hours and at most two physical sessions. Queue and
provisioning before Python entry remain unknown; balances are not forecasts.

Selective paused retrieval returned PAUSED_RETRIEVED/version2, not terminal
scientific acceptance. Retained root:
`D:/molgap-artifacts/retrieved/pcqm-k1-muon-500k-20261106-s42/v1/version2`.
Each arm contains its trace, checkpoint, pause receipt, selected model,
predictions, runtime and cost records. Curve CSV and execution logs:
`D:/molgap-artifacts/continuations/pcqm-k1-muon-500k-20261106-s42/v2/operations/segment40_curve_20261107.csv`.

Checkpoint SHA256:
- AdamW: `765caddcfd17a3e491161819444211817aec38bf092c77c7b25228fbee21497e`.
- Muon: `bbb46f607c12d2e39cbb6b5a9e23d358353c375cb9aa7b79f7cc01dcb964de5b`.

Verified resume bundles:
`D:/molgap-artifacts/resume-validation/pcqm-k1-muon-500k-20261106-s42/v2/20261107/{adamw,muon}/resume_bundle.json`.
- AdamW SHA256: `4c90199d114232d69055f89f7eb10559c1b4c0972571cfabe34f1d8fd80ef26d`.
- Muon SHA256: `05f82ff0a9163cf4969eb77366eda0f4defb942cbc20062422ef35d3e503a3c5`.

## Final continuation provenance

The preauthorized third physical session is SUBMISSION_BOUND at original
Kaggle2 kernel143011302/version3, source dataset version3. Existing
prepare-resume and submit_workflow preserve the original 60-pass schedule;
absolute bound234360 is the original endpoint. Spec/package/source remain:
- Source commit: `ae61a9153a6bdf4f6e31b40a0d9366d2445c7a1a`.
- Source archive SHA256: `57975626246987aeb607b9d513e12acb836c420b117ab0d4b342c4d83df87383`.
- Spec: `d82519ff3d64e476ccb117171371f2f0725ce2b10e4cbc4358e486359f6e5144`.
- Package: `5bc3e3e70457dec92810d010ee918df66256bfbb18e980bc2ee6b79db32e492d`.

Continuation root:
`D:/molgap-artifacts/continuations/pcqm-k1-muon-500k-20261106-s42/v3`.
Its `resume_plan.json`, `prepared/` and `operations/` retain exact inputs and
publication records. Receipt: `operations/launch_receipt.json`, identity
`b86dbc53d4516d5f9bd70e5f8fb966ec347106ad50c285459140601f33d24f1c`.
The submitter checked 30 GPU quota hours available against9 hours before
publication. Initial SOURCE_PUBLISHED returned ValueError; one re-entry with
the same command/journal reconciled it and returned SUBMISSION_BOUND.
No new slug, scientific attempt, source edit or duplicate submission was made.

Luna observed RUNNING/latest3 matching receipt3 at 2026-11-07 13:19:41–45 JST.
That is scheduler evidence, not proof of resumed native progress or 60-pass
completion. No terminal decision, role closure or RML rebuild belongs to this
phase-progress record. Keep the computer on.
