# Registered submission trial — 2026-11-02

Starting checkout: molgap-server5ee57ea86. Both attempts used only platforms/kaggle/submit_workflow.py --experiment ... --execute through platforms/local/run.py. No dry-run chain, quota precheck, manual source list, custom packaging, source upload, trainer change or relaxed verification was used.

1. Input atd4ad4ea6c: BLOCKED atlocal_preparation because quickstart describes preset_ref as a path, while the owning parser requires a {path,sha256} pin. The existing experiment_presets._pin generated the input pointer; no hand-copied hash or new validation code was introduced.
2. Input at9550a2bcd: BLOCKED atlocal_preparation because the supplied post-BN preset recipe cannot be parsed by the new shared recipe owner. No override or frozen-recipe mutation was attempted.
3. The machine profile declares E:/MolGap/data, which does not exist. This is an independently observed host prerequisite, not the cause returned by the second submit call. No alternate cache/split was substituted.
4. Read-only review of the same-run input contract showed a reference arm must keep its frozen recipe. Before any freeze, the proposed reference was corrected to the preset's unchanged1e-5 coefficient and candidate1e-3. This revised input has NOT been submitted; do not treat the earlier errors as validation of it.

These local preparation failures produced no Kaggle receipt or GPU training. No Replay/RML completion is claimed. Further normal execution needs an actually compatible preset/reference declaration and the accepted local fixed-data manifest at the real host path. The trial must not turn into a parallel trainer, silent recipe migration or a substitute dataset. The paper supplies a regularization hypothesis only; it does not promise improvement.

The executed failure facts are retained in [submission_trial.json](submission_trial.json). [Luna existing-parser review](luna_review.md) independently confirmed that obsolete `resume_chunks` survives metadata stripping and triggers the recipe rejection; it also confirmed the absent declared data root. No tests or model inference were run by the coordinator.


## Desktop implementation restored — 2026-11-02

Server repair `7fecf23a9` restored Desktop's existing `resume_chunks` recipe field,
serialization and output activation condition. The original preset and recipe
were unchanged. The existing accepted fixed100K manifest was restored byte-for-byte
to the profile's external data root; only 2401 bytes of metadata were staged.
The declared Kaggle dataset supplies graph shards; no local graph copy was made.

The original submission entry, without `--execute` or credentials, passed both
family-recipe checks and both dataset-manifest bindings. It stopped before freeze
or publication because `question.supporting_evidence` is empty. The scientific
owner must supply the actual reviewed RML evidence IDs before re-entering the
same command; no parser bypass, invented evidence or GPU execution is justified.
The compact machine result is retained in `submission_trial.json`.

## Resubmission after infrastructure repair

Two existing RML evidence IDs were bound in source commit 9b5bb985c. The original submit_workflow entry completed local preparation, then returned BLOCKED / PLATFORM_VALUEERROR at platform_operations, reporting a submission journal path that does not exist. No Kaggle receipt is available. The same original request was re-entered once with verbose output and returned the same error. Account quota lookup reported 29.01 GPU hours available; this does not establish the cause. Preserve the prepared request and reconcile before retry; no GPU execution or Replay readiness is claimed.

## Quota-owner reconciliation — 2026-11-02

The real prepared package reproduced the failure before journal creation or an API
call: Server required duplicate `authority.gpu` fields introduced in `9a291e209`.
The frozen launch already declared two T4 devices and a 21600-second wall cap;
the packaged authority declared 12 physical T4 hours. Authenticated read-only owner
listings found neither `kaseichou/molgap-k1-decay-timescale-100k-s42` nor source
dataset `kaseichou/molgap-k1-decay-timescale-100k-src`. No remote retry was made.

The repair copied the quota-budget block from Desktop `04a54a41a` unchanged:
consume the frozen launch wall cap, retain the packaged physical budget and
exact arm/seed authority, and retain Server's SDK transport. The earlier suggestion
to add a duplicate `gpu` declaration and prepare a replacement run was superseded;
the original authority, prepared package and prospective records were preserved.

The repaired `submit_prepared_workflow(execute=False)` returned
`READY_FOR_PLATFORM_OPERATIONS` for the original external prepared directory.
Package, release report, authority, launch and upload allowlist checks were real;
only credentials were replaced with the account name, and API/HTTP access was
forbidden by the diagnostic. No journal, operation directory or owner pointer
was written. This established local platform admission, not cloud execution,
native training completion or Replay qualification.

Existing `tests/test_kaggle_submission.py` quota and local dry-check nodes passed,
as did `test_full_submission_and_repeated_call_never_duplicate` for both K1 and
GPTrans fixtures, including continuation and receipt reuse. These were CPU
journeys with generated data and substituted platform transport/allocation;
they did not execute the actual candidate on Kaggle. Infrastructure budget passed.
`tests/test_documentation_check.py` retained one unrelated failure: three source
line pointers in `luna_review.md` were unresolved. No new test file or scope was added.
