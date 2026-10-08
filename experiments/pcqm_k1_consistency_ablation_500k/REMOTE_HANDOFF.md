# Desktop return procedure

Owner checkout: `D:/w/k1-consistency-500k`, branch
`codex/exp/k1-consistency-ablation-500k`; source, Spec and package identities
are in [STATUS](STATUS.md). Do not route to server or create a default monitor.
Use desktop project Python with `PYTHONPATH=D:/w/k1-consistency-500k/src` and
the Kaggle workload skill. Explicit credential file for the latest attempt:
`D:/下载/Key/.kaggle3_cli/kaggle.json`, owner `nvoid912`; never print it.
For historical Kaggle1 reconciliation only, use its explicit credential file
`D:/下载/Key/.kaggle1_default/kaggle.json`, owner `nothingnessvoid`.

1. Query the exact kernel/version from [latest submission response](submission_kaggle3_v1/platform_response.json).
   Healthy RUNNING/QUEUED does not justify downloading training files or retrying.
2. At terminal state verify returned source/version, frozen source dataset and
   Spec. Retrieve only pair-state/invocation-cost and per-arm qualification,
   runtime/data/contract/trace/objective/progress/stage manifests, selected model,
   selected aligned predictions and last checkpoint required by acceptance or
   continuation. Do not retrieve all per-epoch predictions or all outputs.
3. Verify every required retained artifact against the owning stage manifest.
   Preserve actual runtime and cumulative costs from every invocation, including
   partial stages and paired idle capacity. Do not infer unknown queue/billing
   from observed native windows. A bounded incomplete stage is a continuation,
   not a scientific negative.
4. If continuation is required and still within the52T4-hour training ceiling,
   use `prepare.py --continuation-from <previous-prepared-dir> --resume-root
   <verified-stages> --checkpoint-dataset nothingnessvoid/<fresh-resume-slug>
   --output <fresh-dir>`. This reuses the lifted retained continuation owner.
   Keep scientific Spec, initialization, recipes and prospective identity;
   review any executable source change. Private checkpoint publication and
   actual submission stay with the skill. Reconcile unknown submit outcomes.
   Already completed peer arms are reused byte-for-byte without another worker.
5. At two complete60-epoch endpoints, reuse accepted native500K artifact/trace
   checks, paired prediction/bootstrap helpers and RML terminal/finalize owners.
   Explicitly compare weight0/0.1 with the primary gate in protocol. Perform the
   same predeclared BN-buffer calibration on both selected states using
   `k1_bn_calibration.recalibrated_batch_norm`; retain original states, separate
   calibrated predictions/buffers, roles, restoration and actual analysis cost.
6. Publish independent per-arm V5/trace/role/cost records, attribution and
   terminal decision. Rebuild/check RML and inspect both replay_pool entries;
   missing qualification or inherited historical cost stays an exclusion.
   Apply the BRANCHES adoption/archive route only after terminal disposition.

Original version1 preparation lives at
`platforms/_records/kaggle/staging/pcqm_k1_consistency_ablation_500k/v1/`.
Its release, prospective and legacy launch bindings are preserved copies in
`submission_v1/`. Initialization files are retained in its `inputs/` and the
private source dataset. No official/common/OOD/test roles or full-data execution.

## Version2 continuation location

Latest prepared payload is
`platforms/_records/kaggle/staging/pcqm_k1_consistency_ablation_500k/v2/`.
Use the actual version2 response and STATUS before reconciliation. Preserve
version1 archive `acb6c461482c4b73f392b18bd69096bb3c01205877ff5a3c7ab8fbb653af4374`
as the producer of the pinned epoch23 recovery states. Version2 changes only
resume retention/plumbing; Spec, prospective launch, initial state and recipes
are frozen. `selected-and-resume-v1` now verifies required selected/resume inputs
against the original full stage manifest without requesting numbered epoch
predictions. Fresh source publication uses `--source-dataset` on prepare.py.
Continue from the latest verified stage/package; never reuse the epoch23 checkpoint
to duplicate progress after version2 has produced a later stage. No healthy tick
artifact download or automatic resubmission.

## Kaggle3 continuation location

Latest prepared payload:
`platforms/_records/kaggle/staging/pcqm_k1_consistency_ablation_500k/kaggle3_v1/`.
Physical kernel is `nvoid912/molgap-k1-consistency-500k-pair-s42-v1`,
ID137710959/version1. Read STATUS, actual response and
[authorization](continuation_authorization_kaggle3.md), not the historical
Kaggle1 example above. Both arms resume from accepted epoch46. The private
source and checkpoint dataset identities are in
[continuation binding](submission_kaggle3_v1/continuation_binding.json);
published privacy, layout and selected-byte checks are in that submission tree.

The shared continuation adapter now accepts explicit `--account`,
`--run-reference`, `--graph-dataset`, `--platform-id` and
`--max-stage-seconds`, alongside fresh source/recovery dataset arguments.
Account changes require all mounts to have the same explicit owner, an accepted
byte-identical fixed500K mirror and unchanged scientific/prospective inputs.
Frozen config labels the new native certificate `kaggle3-t4x2`; runtime/device
equivalence remains mandatory. No new authorization follows from those options.

SDK `kernels_status(...).to_dict()` omits QUEUED because its enum value is0.
Retain and inspect `response.status.name` as in
`submission_kaggle3_v1/scheduler_observation_v1.json`; an empty dict is not an
unknown or failed job. No retry or local polling trigger is required for QUEUED.
