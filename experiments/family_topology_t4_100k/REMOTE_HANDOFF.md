# T4 Execution Comparison Handoff - 2026-10-14

Owner: `codex/exp/family-topology-t4-100k`, `D:/w/family-topology-t4-100k`.
Use this owner until acceptance and a reviewed terminal routing decision.
The [protocol](protocol.md) owns authorization, unchanged recipes, profiling
window, comparison limits and resource ceiling. No monitor, retry, continuation,
shutdown or server takeover is requested.

## Submission Custody

Kaggle1 API accepted private kernel
`nothingnessvoid/molgap-k1-gptrans-topology-100k-s42`, ID138619974/version1.
At `2026-10-14T06:15:32.123105+00:00`, scheduler state was RUNNING and raw API
source matched the frozen entry bytes. The startup query returned no output
files and an empty log. Actual GPU count, native qualification, formal exposure
and measured improvement were not yet observed; script/session ID is unknown.
API metadata's `lastRunTime` is not used as allocation or cost evidence.

Primary records:

- [Push response](submission/kernel_push_response.json) and
  [scheduler/source snapshot](submission/scheduler_source_observation.json).
- [Verified launch receipt](submission/receipts/38996104dccf38014fae8d43b1472f8946dc1f767083d8b0425d6081666173ac.json);
  accepted submission is not runtime qualification or scientific acceptance.
- [Local release checks](submission/release_report.json) own exact executable
  source commit, spec/package/archive identities, recipes, initial states,
  mounted-input bindings and real-shard checks.
- [Dataset download verification](submission/dataset_roundtrip.json) binds
  all11 paths/sizes/SHA values in the private source dataset.
- [Preparation reconciliation](submission/preparation_reconciliation.json)
  preserves local failed-stage history and reuse of unchanged prospective records.
- [Historical asset supplement](submission/retained_asset_supplement.json)
  records three additional exact-hash GINE files required by whole-repository
  portable validation. This is opaque custody, not new model/role execution.

Actual uploaded root is
`platforms/_records/kaggle/staging/family_topology_t4_100k/release002/source_dataset/`,
not the earlier core-stage `inputs/` copy. The private source dataset and
fixed100K graph dataset are named in the verified scheduler/source snapshot.
Source commit35ddecdc is executable; later custody commits do not change it.
CPU package/release regression:88passed/1host-capability skip. No CPU check
qualifies the requested native T4x2 allocation.

## Reconcile and Compare

Use the Kaggle skill/API to query this exact version before any action.
Reconcile retained best/last state, manifest, predictions, canonical trace,
runtime/native proofs and allocation accounting through existing family/RML
acceptance. Outputs are under `/kaggle/working/experiment/<arm>`.
Also retain `execution_profile_receipt.json`, its named
`profiles/step_*/profile.json` and `trace.json`, and
`profile_qualification/profile.json` with its trace. Verify all hashes/context;
ordinary minimal checkpoint retention does not cover every profiler artifact.
Keep complete transaction directories if retrieval is interrupted.

[Baseline bindings](baseline_binding.json) pin prior version1 artifact hashes,
initial tensor files, exact recipe bytes and parent evidence identities.
The [prior summary](../family_100k_operator_profile/terminal_acceptance/summary.json)
and [attribution](../family_100k_operator_profile/terminal_acceptance/attribution.md)
own its endpoints and timing. They are contextual predecessors, not strict
same-physical-job references.

Before interpreting a speed difference, verify complete matched optimizer
steps/presentations, data/row/init/recipe identities, actual T4 allocation and
runtime. Compare each family to its own predecessor: worker-entry wall,
steady training throughput when retained, forward/backward/optimizer/loader
profile phases, operator/launch counts and checkpoint-inclusive capture wall.
Report pair entry/device-allocation wall separately; it is not billed or busy
GPU time. Do not sum overlapping profiler rows or extrapolate window proportions
to whole-run/evaluation costs. Both API metadata images match; native driver/
runtime and contention may still differ.

Align internal-development predictions by `source_idx`; compare target metrics,
prediction differences and live/EMA selection semantics. Rounded equal MAE
alone does not prove equivalent predictions. This is cross-job execution
delivery evidence, not STRICT_CAUSAL architecture evidence. Missing identities,
partial exposure or small noisy differences stay contextual/inconclusive.

Then write terminal attribution, applicable per-arm V5/RML evidence and a
reviewed routing decision. Official validation/test roles remain forbidden;
no automatic successor, model adoption or full-scale release follows.
