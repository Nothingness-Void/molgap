# Kaggle Adapter

For registered multi-arm training, start with the
[modular workflow](../../docs/operations/EXPERIMENT_WORKFLOW.md). Its local
`prepare-workflow` command builds the source dataset, frozen launch metadata,
kernel entrypoint and release report. `platforms/kaggle/run_experiment.py`
verifies those bytes and calls the generic pair runtime; the runtime preflights
every assigned arm before it starts training. Neither package code nor this
bootstrap submits a kernel.

Use the `kaggle-molgap-workloads` skill and existing Kaggle adapters to publish
the prepared dataset/kernel, verify the active account, pass the release gate,
submit, reconcile the same remote attempt and retrieve its manifest-bound
outputs. For a modular workflow payload, pass its `release_report.json` with
`--release-report`; the adapter rejects a missing or failed report and rechecks
the bound package, launch config, mounts, metadata and entrypoint immediately
before POST. Core preparation and bootstrap do not submit. A prepared directory,
successful release check, local receipt or queue response is not training
acceptance. Keep the platform skill as the owner of account credentials and
authoritative remote state.

## Frozen K1 continuation

For SDK operations with an explicit key, use `credential_api.py`'s
`api_for_credentials(path)`. It binds the requested account even when the SDK
finds a different global OAuth session. Keep credentials outside the repository;
verify the owner before publication and leave global login files intact.

`resume_frozen_k1_arm.py` restores one incomplete arm through the existing
isolated qualification/training workers. Keep the exact original source package,
Spec, recipe, prospective records and producer context. A separate reviewed
platform entry binds the physical continuation, five retained output hashes,
complete-epoch cursor, runtime fingerprint and frozen target-identity plan.
The adapter retains the old completion inspection and separately applies the
explicit encoding binding; it never suppresses another inspection blocker.
Completed peer arms are reused. Incremental recovery allocation is a separate
cost observation; an absent prior ledger stays missing. The owning question
supplies its authorized continuation declaration and source-dataset preparation.

For the new family output protocol, `retrieve_family_outputs.py` streams only
the pinned manifest's required files through an authenticated owning account.
Source/job/version reconciliation and obtaining the small manifest belong to
the platform skill. See [family workflow](../../docs/operations/EXPERIMENT_FAMILY_WORKFLOW.md).

Cross-experiment Kaggle packages are grouped by workload role:
`acquisition/`, `training/`, and `evaluation/`. Experiment-specific kernels
stay with their owning experiment.

Kaggle provides the server-side queue. Submit a durable kernel directly,
verify its remote state, and do not create a local slot-trigger process.
Retrieved outputs and acceptance evidence belong in
`platforms/_records/kaggle/`.

## Paid-run release gate

Before each `kernels push`, including a retry, the owning packager must fail
closed unless the frozen experiment contract agrees with the exact packaged
source on source commit, optimizer and schedule, batch/tail policy, precision,
seed, exposure, data and row identity, and protected-role use. Read numeric
training values from the source snapshot that will execute, not a hand-written
description or the current checkout. Verify the package/source hashes and run
only the existing syntax, clean-import, real-shard and semantic preflights.
Startup success is not contract acceptance. A retry with a new source commit
requires a new prospective authorization before another GPU submission; do not
silently edit the old frozen contract. If any identity is unavailable, stop
before submission and report the exact mismatch.

For new ExperimentSpec source packages, use the shared CLI's `check-release`
and pass its report to `push_kernel_with_accelerator.py --release-report`.
Include the kernel entry script and staged input root when producing the report.
The adapter rechecks the bytes before POST; a stale or failed report blocks the
request. Existing legacy call signatures remain supported. See
`docs/operations/EXPERIMENT_CLI.md` for scoped checks and limitations.

Use `--response-output` to retain the returned source binding and platform
identity. `requested_kernel` is the submitted metadata ID; `kernel` is only the
response's actual ref or URL. Title-generated slugs can differ. `version_number`
and `script_version_id` are separate fields and are never guessed. A response
without sufficient identity sets `reconciliation_required`; reconcile the same
submission before considering another push. A timeout/connection failure is
`submission_unknown`, persists a response observation, and exits nonzero.
Neither the response nor a successful local check is scientific acceptance.

Keep packaging to the existing model-family adapter. Do not create a new
experiment-specific packaging or acceptance framework merely to resubmit a
variant. After completion, follow `research_memory/LIFECYCLE.md` for minimal
retention and terminal acceptance. If routine packaging or acceptance exceeds
five minutes of active local packaging or three minutes of local acceptance
after the required artifacts are present, stop and identify the blocker instead of
performing an unbounded manual rewrite or downloading every raw artifact.

| Account label | Kaggle owner | Accepted fixed PCQM 100K/500K evidence |
| --- | --- | --- |
| Kaggle1 | `nothingnessvoid` | `platforms/_records/kaggle/pcqm_fixed_datasets_v1/` |
| Kaggle2 | `kaseichou` | `platforms/_records/kaggle/pcqm_fixed_datasets_kaggle2_v1/` |
| Kaggle3 | `nvoid912` | `platforms/_records/kaggle/pcqm_fixed_datasets_kaggle3_v1/` |

The three accounts hold accepted, byte-identical private OGB PCQM4Mv2 100K
and SCNet-matched 500K graph datasets. The records above own the exact dataset
references and hashes. Kaggle1 and Kaggle2 are excluded from holding the 1M
and full identities. These mirrors confer no access to protected evaluation
roles. Verify the active credential owner against the intended kernel and
dataset slugs before every push; this table does not describe the current
local login.

Kaggle3 also holds the reusable source layer
`nvoid912/molgap-v5-desktop-runtime`; its evidence is under
`platforms/_records/kaggle/molgap_v5_desktop_runtime_kaggle3_v1/`. The Kaggle3
batch API did not allocate TPU v5e-8 despite correct remote metadata; TPU
remains gated by an interactive hardware and framework preflight.
