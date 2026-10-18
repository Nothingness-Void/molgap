# User cancellation and workflow inspection —2026-10-18

## Observed execution, not a scientific verdict

The human reported canceling the job after about2000 seconds without training
logs and redirected attention to mature K1/GPTrans-T for practical workflow
validation. This superseded the earlier continue-to-budget instruction.

Read-only reconciliation verified kernel139370303/version1, still bound to
`kaseichou/molgap-modern2d-formal-100k-v2`. Two scheduler observations returned
`CANCEL_ACKNOWLEDGED` at07:13:04 and07:13:45 UTC. No agent cancel, retry or GPU
POST was issued during this inspection. The retained SDK log ended at worker
time48.504285409 seconds after dependency installation. No visible training
log does not establish zero optimizer steps or identify the remote blocking
phase. The exact-version `experiment/execution_report.json` was unavailable;
no bulk-output fallback was used. Native acceptance, progress, final artifacts,
role consumption and per-job allocation/billing remain unestablished. This
record is not terminal RML finalization or an architecture-negative result.

Both post-cancellation quota observations reported2703.390 used seconds and
zero reservation in the same refresh window. Before this attempt the account
reported535.912 used seconds. The2167.478-second account delta is not a per-job
invoice or a full two-device allocation measurement. It must not silently
replace native/billing evidence or expand the original resource authority.

Raw observations and existing-owner reports were retained at:
`E:/MolGap/artifacts/modern2d-formal-screen/inspection_retry_v2/user_cancel/`:

- `cancellation_observation.json`, `cancellation_confirmation.json`;
- `selected_storage_census.json`, `v2_run_storage_census.json`;
- `v2_workflow_timing.json`.

## What the new front door did and did not prove

The frozen workflow report measured43.0464142 seconds for ordinary local
preparation. The outer preparation observation measured43.6465972 seconds,
including its wider call boundary. Three retained reuse observations took
0.110775–0.237310 seconds at that internal boundary. These do not include
process startup, source-ready waiting, queueing or first optimizer update.
The timing reporter explicitly left end-to-end receipt, source-ready waiting,
isolated upload transfer and training queue unmeasured. Local packaging speed
therefore did not establish a fast or usable remote training workflow.

The complete, read-only census of this attempt's four retained subdirectories
counted121718597 logical bytes in51 files (about116.08 MiB), excluding root-level
files and external initialization originals/shared cache. Preparation accounted
for61646831 bytes and operations60061954 bytes. The leading files were the
38.27MB PPGT and20.11MB ESA initializations, each present in staged inputs and
the byte-verified source roundtrip. Three1.412MB source-archive copies were also
visible. These are logical path counts, not allocated disk space or globally
unique content; the new front door did not copy the graph cache into this run.

The wider35-second census hit its time limit and excluded nested worktrees.
Its partial counts cannot establish all-project size, cleanup eligibility or
absence of historical disk bloat. No files were deleted or migrated. At the
inspection, C: had about22.23GiB free and E:313.30GiB free. Large new run inputs,
reports and scratch remain on E: per the human storage boundary.

## Mature-model trial boundary

Use the existing FamilyRecipe/Core/Session, shared logger, frozen source cache,
Kaggle submission journal and manifest-bound retriever. Do not copy legacy
experiment scripts or develop another training/submission framework.
Using mature models alone does not change a selected indexed data loader:
the immutable manifest chooses the fixed-packed versus indexed loading route.
Any proposed mature-model trial must bind its already accepted fixed data,
finite execution endpoint, log cadence, full T4 allocation ceiling and retention
scope separately. It must measure preparation, publication, receipt, first
optimizer update, recovery/collection and storage independently, without a
new scientific MAE ranking, extra seeds or protected-role access.

The bounded source review identified compatible mature interfaces
`neural_atom_k1/3` and `gptrans_t/2`. In
`src/molgap/family_training_data.py`, the fixed route verifies/opens each shard
once and retains it in a `ConcatDataset`; random sampling does not re-enter
the indexed single-shard pin/open path. This isolates the established indexed
rehashing mechanism, not all possible runtime problems.

The accepted Kaggle2 mirror was already recorded as
`kaseichou/pcqm4mv2-ogb-fixed-100k-v1`, version1, manifest SHA256
`1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d` in
`platforms/_records/kaggle/pcqm_fixed_datasets_kaggle2_v1/acceptance.json`.
This is the existing fixed dataset, not the Unified V2 Common-Dev identity;
its remote availability was not rechecked during this source review.

The shared reference-only `molgap-experiment-input-v1` entry can bind a
dedicated execution Spec/workflow through the same prepare/cache/journal path.
The scientific-delta compiler must not be tricked with a fake learning-rate
change just to create a workflow test. Existing long-horizon contracts are
examples, not release authority or the short test's endpoint. A train-only
short endpoint needs explicit final-only selection and the existing worker's
bounded-allocation hook; an adapter budget field alone is not a runtime cutoff.
Thread/statistics compatibility and logging are also explicit pre-release
checks. No new trial was prepared, published or submitted by this inspection.

The delegated Luna worker independently ran the existing
`tests/test_family_training_data.py` and `tests/test_training_logging.py` suites:
46 passed, zero skipped/failed,27.44 pytest seconds (29.353 process seconds).
Four CPU-only pin-memory warnings were retained. JUnit, log and run summary
are under `E:/MolGap/artifacts/reports/k1-gptrans-workflow-trial-cpu-20261018/`.
These synthetic CPU checks establish neither native GPU throughput nor
end-to-end latency/storage acceptance. The coordinator did not execute tests.
