# Local Engineering Acceptance

CPU-only synthetic checks, delegated to Luna Max. No molecular training,
protected-role access, native qualification or quality claim comes from them.

| Invocation | Result |
|---|---|
| Component, substrate, family recipe/kernel/runner, execution/Spec and same-run workflow tests | 339 passed |
| Scientific-input and initial shared-family wiring tests | 22 passed |
| Supplemented optimizer-step/objective-binding wiring tests | 2 passed |
| Kaggle submission and owned-absence mocks | 35 passed |
| Final public-placeholder-shape regression | 8 passed |

The wiring cases also occur in the 22-test invocation; counts are per invocation,
not an assertion of unique tests. An intermediate supplementary fixture lacked
the runtime's deterministic guard and failed two cases; the fixture was fixed
and both passed. No training implementation was changed for that fixture.

Local preparation verified both frozen base state artifacts, dataset/target
manifests, recipes, source imports, same-run acceptance inputs and final launch
bindings. Worker source: `1b8c7b5c8b5e7f51738fb35b5bcce76876506d3f`.
K1 package: `aa2cfd0a5912fc29b4154ff34f9616a6150ca5b9e03edcbb8dab1fff8db4242c`.
GPTrans package: `2b14146067c1c99997c916bbec7ae791fac1295ac92bcc700ebd1c1863665ca6`.

## Preparation Reconciliation

The initial unsubmitted K1 declaration had inconsistent CRLF/raw and LF/source
recipe pins. It was stopped before RML publication or remote mutation. Its
original frozen inputs remain at
`D:/molgap-artifacts/runs/pcqm-k1-post-ln-100k-s42/v1/`.
The dated logical runs use explicit LF bytes and do not overwrite that freeze.

The dated K1 preparation published both prospective trajectories but its global
RML rebuild initially lacked two historical strict-comparison weights in the new
worktree. Both exact files were present in the primary checkout. The explicit
SHA-bound allowlist totaled 44,999,700 logical bytes; verified hardlink aliases
were created without a second payload allocation, remote retrieval, unpickling
or model execution. No inherited catalog was copied. The allowlist is retained
at `D:/molgap-artifacts/reviews/structural-bias-norm-100k/rml_required_allowlist.json`.

`reconcile_local.py` used existing compiler/staging/freeze/binding owners to
finish this exact unsubmitted package, preserving its original report, source
and published actions. A local report serialization error was corrected before
the reconciliation completed. The original and reconciled staging are retained;
the script is a dated recovery tool, not a new normal submission interface.
GPTrans completed the ordinary workflow without that recovery.

Final repository metadata verification reported 125 trajectories, 109 evidence,
250 costs, 313 roles, 37 traces and zero READY packages, with no validation
errors. `check --frozen` confirmed generated RML is current; `git diff --check`
passed. This is not a terminal replay-readiness or native runtime acceptance.
Imported platform JSON receipts preserve exact original CRLF bytes; the local
Git attributes recognize CR-at-EOL instead of rewriting hash-bound evidence.

## Platform Compatibility

Kaggle1's historical owned listing contains SDK privacy placeholders with
`id=0`, empty ref/author/slug and title `[Private Notebook]`. The SDK public
`is_frozen` accessor was unavailable, so qualification uses those observed
public identity fields, not its private backing field. The old transport
rejected these before any source POST. The host-only Common API now completes
the owner listing and asks the authenticated server for a bounded target-slug
search if opaque rows occurred. Matching, malformed, duplicated or opaque
scoped results still block absence qualification. Synthetic tests and actual
read-only queries covered this path. Worker packages and training logic remain
unchanged; source publication continues through the original journaled API.

Remote scheduler state, actual version, native preflight and terminal science
remain separate from this local report. Follow retained submission receipts.

## Version 1 Failure Repair, 2026-10-20 JST

See [failure attribution](failure_v1.md). Luna ran CPU-only tests with the
primary checkout interpreter, this worktree's `src` on `PYTHONPATH`, and
CUDA/HIP/ROCR devices hidden:

```powershell
D:/文档/molgap/.venv/Scripts/python.exe -m pytest -q tests/test_kaggle_bootstrap.py tests/test_experiment_workflow.py tests/test_experiment_workflow_resume.py
D:/文档/molgap/.venv/Scripts/python.exe -m pytest -q tests/test_kaggle_bootstrap.py
```

The grouped run reported 138 passed and two new-test failures in 385.15s.
Both failures were the regression fixture's missing `manifest_sha256`, before
reaching the guard; only that fixture was corrected. The focused repeat then
passed all 77 bootstrap cases in 2.53s. These invocation counts overlap; they
are not additive. The workflow/resume tests were not rerun unnecessarily.

A temporary copy outside the checkout exercised both real-guard cases against
the old entry at `20f5a35c`: the fresh-root case reproduced the same ValueError;
the existing-checkpoint rejection case passed (one expected failure, one pass,
15.98s). No checkout files were changed for this negative control. The old
entry contains the version-1 publication ordering described in the failure
report; no frozen source payload or remote version was modified.

Parent verification matched all four downloaded output SHA256 values to their
version-specific retrieval receipts. `git diff --check` passed. This repair
has no new GPU, native preflight, model-quality or retry acceptance attached.

Before the explicitly authorized retry, one grouped CPU run of
`tests/test_kaggle_bootstrap.py tests/test_experiment_workflow.py -k 'bootstrap or pair_runtime'`
reported **87 passed, 43 deselected** in 7.15s. It covers the repaired entry
and pair-runtime guard/worker boundaries without accessing GPU or remote roles.

## Explicit Retry Preparation

The source repair is committed at `d6058bc5`; the identical shared entry and
test patch is selectively integrated into local `molgap-desktop` at `eea8269f`.
The entry, bootstrap tests and pair-runtime owner match across those tips;
no experimental model adoption or whole-branch merge is implied.

`prepare_retry.py` only declares `attempt_002/` through existing Spec and Common
proposal APIs. Luna's read-only review found both retry arm declarations exactly
equal to their originals, with unchanged recipe, initialization and data paths.
All four prospective source/action/attempt/plan hashes match `d6058bc5` and their
new logical runs. Each workflow's seven source-record refs exist in that commit.

Both attempts completed existing preparation, release checks, prospective
publication and RML rebuild. Full RML validation reported 129 trajectories,
109 evidence, 254 cost records, 313 role records, 37 traces and zero READY;
this is not terminal replay qualification. Frozen packages and API publication
proofs remain under the machine artifact root and each retry's `submission/`.

Source publication initially left status queries unknown. Read-only reconciliation
confirmed each exact source dataset ready and its new kernel absent before
continuing the same journal. No source POST was repeated and no extra notebook
was launched to bypass that uncertainty. Both submissions subsequently bound
actual physical version 1. See [status](STATUS.md), not this dated local report,
for scheduler observations and native-startup limitations.

Post-commit `check --frozen --portable` did not pass in this isolated worktree:
historical V5 records claim locally retained payloads not mounted here, including
old operator-profile/topology outputs and archived checkpoints. Ordinary
`validate` and `check --frozen` passed. No inherited artifact catalog was hydrated,
historical evidence rewritten or full-library portable acceptance claimed.
This storage-availability limitation is separate from the two new submissions'
verified source/input release and their pending native qualification.
