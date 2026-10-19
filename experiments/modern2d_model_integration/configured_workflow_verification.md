# Configured workflow verification, 2026-10-19

The implementation extended existing owners rather than introducing another
training or submission framework:

| Concern | Reused owner | Added behavior |
|---|---|---|
| Scientific input | `experiment_input` | PPGT/3 and ESA/4 catalog addons, optimizer edits, explicit pass horizon, context-only initial screens |
| Release and authority | `modern2d_training_release` | Independently configured one/two-arm indexed-role plans and full-allocation budget |
| Runtime admission | `Modern2DFormalTrainingContext` | Configured flag, exact run/account/role/horizon checks and feature-only bindings |
| Execution | Existing Kaggle pair runtime, native preflight, FamilyKernel and Session | Configured plan routed to existing budget and checkpoint controls |
| Saved evidence | `experiment_family_workflow`, `SessionOutputs` | Formal interfaces dispatched to the Session inspector; pinned configured target manifests |
| Recovery and RML | Existing frozen run, resume and terminal owners | Original identities retained; no replacement recovery or closure engine |

The historical formal60-pass plan retained its ordered pair, exact scientific
recipe and budget checks. The configured plan required separate authority.
No actual authority, source publication, data cache or remote job was created.

## CPU evidence

Main authored code/tests; Luna workers executed the suites, with one CPU thread
per process, CUDA hidden and independent E-drive scratch paths. Only synthetic
molecules and fake indexed metadata were used. Real role payloads were absent.

- `regression-rerun2.xml`: 110 existing formal release/training, addon Session
  and indexed-feature checks passed; the newly added output test exposed a
  generic-inspector dispatch omission and failed. This was retained as failure
  evidence rather than represented as a fully passing run.
- `configured-dispatch-rerun.xml`: after the dispatch repair, 63 output/workflow
  checks passed in 171.694 seconds, including actual tiny addon Sessions,
  independent saved-output inspection and rejection of a changed trace companion.
- `frontdoor-rerun2/junit.xml`: 14 isolated front-door cases passed in 533.837
  seconds. They checked both families' real offline source packages, source-bound
  native metadata context, frozen continuation across HEAD changes, resource/role
  rejection and metadata tampering. This preceded the final metadata-pin check
  and consolidation of repeated negative-case fixture setup.

The final front-door run (`final-frontdoor/junit.xml`) passed 43 checks with
zero failures/errors in 362.136 seconds. It covered the final Modern2D compiler
and authority changes, existing K1/GPTrans scientific-input scenarios and the
historical formal-release regression. The separate same-family dual-arm
compilation extension (`final-frontdoor/added-arm-junit.xml`) passed both family
cases in 80.68 seconds. Thus the final non-overlapping front-door batch had
45 passing checks. The outcome for this bounded implementation was
**INFRA_CPU_ACCEPTED**. No remote execution or scientific promotion followed.

Original reports are retained in
`E:/MolGap/artifacts/configured-modern2d-tests-20261019/`. Counts from overlapping
runs are not added together.

## Interpretation

These checks establish bounded CPU mechanics, not CUDA determinism, native
throughput, full-allocation billing or model quality. Native repeat/resume
preflight remains mandatory for each newly frozen runtime/recipe. Missing
costs stay missing. Initial model screens remain context-only/INCONCLUSIVE;
no ranking, Replay admission, automatic 500K decision or early stop was added.

The public configuration and resource binding are described in
[the workflow guide](../../docs/operations/MODERN2D_CONFIGURED_WORKFLOW.md).
