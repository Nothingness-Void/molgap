# Native T4 diagnostic disposition - 2026-10-09

Disposition: NO_TRAIN, execution diagnostic only; no model promotion, quality
acceptance, training replay, full release or automatic successor. Parent must
independently execute byte acceptance and freeze closure artifacts; this document
does not assert that RML finalization has occurred.

Authority: [protocol](protocol.md), [prospective trajectory](rml/trajectory.json),
[payload plan](payload_plan.json), [publication binding](publication_binding.json),
[submission response](submission_response.json). Parent submitted
`nvoid912/molgap-k1-native-t4-profile-s42-v1`, kernel137849502/version1 and reports
COMPLETE retrieval. The parent's [terminal scheduler snapshot](scheduler_reconciliation_complete.json)
binds that exact kernel/version to COMPLETE. The earlier RUNNING snapshot is
historical and unchanged; this adapter makes no new remote query.
The [completion inventory](../../../platforms/_records/kaggle/training/k1_native_t4_profile_s42_v1/profile/completion.json)
and [entry observation](../../../platforms/_records/kaggle/training/k1_native_t4_profile_s42_v1/profile/entry_observation.json)
record complete execution. `close.py` verifies their retained bindings.

Publication pins manifest
`d06f5ecdb4e399a9ed55bae081d98160cf90bef7bcecc6b4a984404d1dfd75be`,
archive `a5d7bc79a77a71a9b0b94f0cdc3bceb6cae8711a2fa111623da3fea5167a6cde`,
setup `d8f183d41df1a5bd50ae2d2ea4f5ebe9d1b1fb7e55fffde51bb7affff11d05c7`,
and extractor `2c6401136d509b3b4ab3832a8dce4d08b6948ccb34c103754a78182553c082b5`.
The plan/manifest bind source commit, exact source inventory, selected checkpoint,
accepted sample and prospective bytes; acceptance never loads a pickle.
Observed runtime: Torch2.11.0+cu128/CUDA12.8, two Tesla T4s, cuda0 active,
seed42, FP32/noTF32 and deterministic algorithms.

Matched24-step medians: single0.169898624s, mean2 double0.272408326s;
single-step reduction37.630898%, above the prospective25% review threshold.
This warrants only separate cost-quality review, not single-pass MAE equivalence
or global epoch/training-cost extrapolation. Mean2 means mean of two supervised
L1 losses, coefficient0, not a2.272s timing or a two-loss sum.

Full entry307.727669617s *2/3600 =0.170959816454 allocated T4-device-hours,
including verification/setup/bootstrap. Worker56.740963478s and child supervisor
61.663854923s are nested subsets, not additional allocated costs. GPU busy,
queue, remote upload and full50K development time remain unknown. Active work
windows are not GPU-busy measurements.

Both cases perform29 scratch updates (5warmup+24measured), with8 independent
phase updates:66 total, discarded as scientific state. Six synchronized phase
samples and four eval-only batches/512 train members plus one localFS atomic
scratch checkpoint write are retained, including its completion SHA. The
accepted checkpoint is unchanged; retained scratch bytes are publication
evidence, not an accepted trained model.

The original accepted4096-row sample was prebuilt; preparation reused exact
bytes without decoding. This worker decoded only4096 official-train-derived
members, not the entire500K backing labels. Development and protected roles
were untouched. No new scientific training, selection or accuracy metric.

See [attribution](attribution.md) before any separate quality question. Accepted
diagnostic evidence/reusable controls may follow the reviewed non-promotion
desktop route in BRANCHES; this task performs no routing, commit or rebuild.
