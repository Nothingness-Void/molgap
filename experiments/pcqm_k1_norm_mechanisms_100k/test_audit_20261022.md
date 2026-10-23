# Audit of the124 passing local recovery cases

The124 passing cases are39 test functions expanded by parametrization. They
contain useful component checks, redundant combinations and weak assertions.
**None executes a native K1/GPTrans training worker or restores a populated
native optimizer through the real continuation preflight.** Their green result
does not establish that the published Kaggle continuation can train.

## Scope and method

Audited the four test files executed against repair commit
`9bda07165cffc4687965ed815837d8bf5b3a0829`. Their bytes were compared with that
commit; [inventory](test_audit_inventory_20261022.json) retains hashes. The
retained run log reports124 passes in303.18seconds. This audit read every test
function, its fixtures and relevant production callers. Collection only
confirmed124 node IDs in3.66seconds; test bodies were not rerun.

[All124 case assessments](test_audit_124_20261022.csv) record the actual
executed behavior, substituted boundary, failure addressed, proof limit,
recommendation and relevant change type. The [39-function table](test_audit_function_table_20261022.md)
groups them without hiding parameter cases.

| File | Functions | Expanded cases | Real coverage | Unproven boundary |
|---|---:|---:|---|---|
| test_kaggle_bootstrap.py |13|86|Entry budget/cost/error handling, filesystem overlay, packaged input routing and orchestration barrier|pip installation, actual GPU discovery and native workers are replaced|
| test_experiment_workflow_resume.py |10|10|Package/receipt/context binding, export/staging/restore/retention|Synthetic trainer and metadata-only checkpoint; no optimizer/RNG/scheduler restore|
| test_shared_resume_transport.py |13|25|Real file copies/hashes/links/restore, deterministic replacement race and safe retention|Tiny checkpoint; owner dispatch is stubbed in its routing case|
| test_machine_paths.py |3|3|Path/profile checks and real immutable byte copying|Junction attribute is simulated; no full-disk or native model admission|
| Total |39|124|Component and selected orchestration coverage|Zero native training cases|

Synthetic inputs are appropriate for isolated hashing, atomic publication,
argument routing and error paths. Replacing the native worker is also reasonable
when the stated question is process ordering. Those substitutions become a
coverage error when interpreted as proof of successful model recovery.

## Specific findings

1. **The native recovery path is absent.** `resume_case` at
   `tests/test_experiment_workflow_resume.py:157` replaces the registered K1 and
   GPTrans adapters through `_install_synthetic_owner` in
   `tests/test_experiment_lifecycle_integration.py:139`. Its checkpoint at
   `tests/test_experiment_workflow_resume.py:101` contains identity, trace hash
   and cursor, without model/AdamW/RNG/scheduler state. `_ImmediateWorker` and
   `_patch_pair_runtime` at lines423/442 return scripted status codes. The
   bootstrap fresh-interpreter case at `tests/test_kaggle_bootstrap.py:422`
   imports the packaged orchestration code but similarly replaces Popen. No
   optimizer tensors can alias in these fixtures because none is restored.

2. **One negative test overstates atomicity.** In
   `tests/test_kaggle_bootstrap.py:49`, original/replacement corruption targets
   `src/molgap/common/io.py`, the first file in the canonical sorted manifest.
   The assertion only checks that `experiment_resume.py` remains unchanged.
   An incorrect implementation that validates and writes each file immediately
   could still pass all these cases: the first corrupt file fails before an
   earlier valid file has been written. Rewrite the existing case to corrupt a
   later file and assert every original byte remains; more cases are unnecessary.

3. **48 budget combinations are excessive.** The case at
   `tests/test_kaggle_bootstrap.py:178` expands8 limits x2 reserves x3 outcomes.
   It tests timeout subtraction, failure accounting and writer selection;
   installation and runtime are stubs. A concrete conservative reduction is11:
   eight successful limit cases with reserve60, one reserve0 success, and pip
   plus runtime failure at the representative14400/60 budget. All eight chosen
   limits, both reserve branches and all three outcomes remain represented.
   The CSV marks the11 retained representatives and37 proposed redundant
   combinations. This is a recommendation from branch/formula inspection,
   not a measured mutation score or runtime saving.

4. **One case checks an alias rather than useful loader behavior.**
   `tests/test_shared_resume_transport.py:100` only asserts that two Python
   names refer to the same function object. Both names could point to a broken
   loader and still pass; a harmless wrapper can fail the assertion. Remove
   this identity-only case from the behavioral suite. Loader behavior should
   be judged through actual load/rejection checks owned by that loader.

5. **Some rejection tests cannot isolate their claimed cause.** The completed
   canonical arm case at `tests/test_experiment_workflow_resume.py:348` changes
   both the decision and its bytes; frozen-byte mismatch already rejects it,
   so the case cannot independently establish the ACTIVE-state check. Merge
   or rename it with the changed-record case at303. The case named
   `test_resume_checkpoint_modified_after_stage_is_rejected_before_post` at397
   exercises local preparation and BLOCKED release, but never calls a submitter
   or spies on POST. Keep the hash guard and describe that narrower scope.

6. **A real component success can coexist with a failed cloud run.** The
   resumed-pair test at `tests/test_experiment_workflow_resume.py:461` reports
   complete after fake workers return zero. It usefully checks argv, restore
   paths, barrier and cost carry-forward. It cannot establish native training.
   The failure case at537 deliberately preserves `training_started=true` for
   inherited checkpoint work. That field must not be read as new training in
   the failed continuation attempt.

7. **Expensive fixture setup is repeated for small predicates.** Every one of
   the10 resume tests builds a temporary Git/source/package/release/receipt and
   exports both arm bundles before testing its predicate. That setup is
   justified for a few integration scenarios, but broad selection repeats it
   even for receipt state, a changed byte or a missing arm. Keep these checks
   as scoped infrastructure regressions; do not run all of them for an
   unrelated checkpoint copy or scratch-probe repair. The48-case matrix is
   mostly lightweight stubs, so its large count must not be blamed for the
   full303seconds without timings. Per-case durations were not recorded;
   this audit does not invent their share or promise a specific speedup.

   There is also overlap between the two nested-metadata cases, two freshness
   cases and four frozen-launcher scenarios in the bootstrap file. Keep their
   different component/integration scopes visible; do not select all layers
   automatically for every resume repair.

## Relation to the two observed failures

- Version1 failed while publishing a resume bundle. The two deterministic
  sidecar-replacement cases at `tests/test_shared_resume_transport.py:142`
  were added after that failure and are included in these124 passes. They
  exercise the relevant real copy/replace interleaving and meaningfully guard
  the snapshot fix. The mutated-checkpoint case at215 retains strict failure
  on real checkpoint byte changes. These are valuable fault-specific tests.
- Version2 stopped during native actual-resume preflight. Two scratch probes
  reused a loaded optimizer state; AdamW CPU step tensors could be mutated by
  the first probe and rejected by the second. The124-case fixtures have no
  populated AdamW restore, so they cannot reveal this. A separate one-case
  regression using real CPU AdamW and the actual probe wrapper subsequently
  passed in5.28seconds. It proves that narrow isolation behavior, not native
  T4 admission or completion of160 passes. It is not part of the124 count.
- Real platform dependencies, installed runtime, mounted executable source,
  native device behavior and the actual checkpoint must still be checked by
  the owning preflight. A fake worker result cannot substitute for it.

The tests themselves repeatedly state CPU-only/synthetic boundaries. The
earlier report treated the aggregate green result as stronger recovery
assurance than it justified. Both test selection and interpretation need
correction; adding more scripted success fixtures would not fill the gap.

## Practical selection and disposition

Do not delete real no-overwrite, digest, source-retention, deterministic race or
safe-pruning guards. For a sidecar snapshot/copy repair, the two race cases and
one mutated-checkpoint case are the direct local regression. For the scratch
state-isolation fix, the one populated-AdamW regression is the direct check.
Then use the existing native continuation preflight and require its durable
certificate plus observed formal progress before reporting resumed training.

Run budget/parser, package/barrier, accounting and storage suites when those
owners change. These remain useful component regressions but are not mandatory
on every model submission or unrelated repair. No new test-selection framework,
validation schema, model implementation or publication adapter is needed.

The37 matrix combinations and one identity case are concrete pruning
recommendations; original test cases were not deleted during this read-only
audit. The negative overlay and proof-scope findings call for rewriting existing
assertions/names. The audit does not claim the remaining86 cases independently
prove86 production scenarios. Version2 is terminal ERROR; version3 has not been
submitted while this audit is being resolved. Scientific acceptance/RML remains
pending.
