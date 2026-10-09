# Family output and acceptance hooks

The shared module connects an owning trainer's outputs to mechanical inspection
and existing terminal/RML closure. Training, submission, runtime qualification
and scientific decisions remain with their existing owners.

## Prospective Same-Run Reference

For an explicitly authorized fresh pair, use acceptance format
`molgap-family-same-run-acceptance-plan-v1`, not a fabricated retained-reference
bundle. Top-level fields are `format`, `spec_identity`, `arms`; each arm has
`arm_id`, `adapter`, `expected`, `contract`, `target_manifest` (pinned local
path/SHA256). Spec v2 must already declare `prospective.same_run_replay` for
exactly those arms. Peer family/base/initialization/data/sampler/transform,
optimization recipe, development identity and terminal exposure must match.
The target manifest supplies byte encoding and target identity only.

The shared checker/stager and `TargetIdentityBinding` support this format.
`ACCEPTANCE_INPUTS_AVAILABLE` means the prospective inputs can be staged;
its `reference_authority` explicitly says the terminal reference is not yet
accepted. Runtime qualification, measured roles/cost, terminal reference
acceptance and strict comparison remain separate downstream checks. Close the
actual reference before its candidate using the existing RML pair protocol.
The retained-reference format and its complete bundle requirements are unchanged.

## Capability matrix

For a retained historical target encoding, use the explicit library binding
`TargetIdentityBinding.from_acceptance_plan(spec, repo_root, plan_path,
plan_sha256=...)` and pass it as `target_identity` to `inspect_output`, or in
each direct closure output item. `accept_workflow` accepts an optional
`target_identities` mapping for library callers. The pinned prelaunch plan must
pass existing reference checks; its target manifest declares exactly one byte
encoding and matches the frozen recipe hash. Default float64 hashing is unchanged.
Float32 raw-byte identity requires original finite float32 tensors, with no
fallback or lossy conversion. Descriptor construction and closure reinspection
retain the same binding. This does not alter a frozen recipe or scientific gate.

| Component | Responsibility |
|---|---|
| `RunContext` | Spec/arm/source/package/recipe identity and observed launch binding |
| `FamilyOutputSession` | Atomic selected/resume/prediction outputs and observed epoch recording |
| `StageRecorder` | Existing canonical RML recorder with distinct phase/run identities |
| `inspect_output` | Frozen requirements, hashes, rows/targets, finite state, exposure and selected endpoint |
| `check_acceptance_plan` | Adapter availability and pinned owning comparison/reference inputs |
| `build_verified_terminal_descriptor` | Deterministic translation into the existing descriptor |
| `close_verified_outputs` | Inspect every arm before existing terminal/RML closure |
| Kaggle `retrieve_family_outputs` | Stream only manifest-bound files through the owning account |

Static profiles: `k1-screen-v1` for `neural_atom_k1/2`, `k1-v1` for
`neural_atom_k1/1`, and `gptrans-v1` for `gptrans_t/1`. `k1-screen-v1` is a
separate output adapter; it does not reinterpret the retained `k1-v1` recipe
or manifest contract. The screen profile adds the registered K1 family trainer
to this protocol. K1 resume state requires model, optimizer, scheduler and RNG
state; GPTrans requires model, EMA, optimizer and RNG state. Both retain the
acknowledged sampler cursor. GPTrans screen output normalizes the existing V4
trainer's selected model, predictions, trace and resume checkpoint into the
shared manifest after checking source, variant, exposure and hashes. It records
an optional `runtime_certificate_id` in runtime metadata and retains
`runtime_certificate.json`. These profiles cover the **new output protocol**;
historical runners are not automatically migrated. Keep their frozen payloads
and acceptance loaders intact. Missing historical counters and wrong account
labels remain missing/wrong, not repaired by relabeling.

## Integrate once in an owning trainer

Keep the existing model, loader, sampler, trainer and scientific recipe.
Read [the addon guide](EXPERIMENT_ADDON_GUIDE.md), then add event hooks:

1. Freeze the owning recipe JSON with `acceptance_requirements`: exactly
   `epochs`, `optimizer_steps`, `sample_presentations`, `development_rows`,
   `source_idx_sha256`, `target_sha256`, `precision`. Pin the whole file in
   `Spec.arms[].training.recipe.sha256`. Requirements must precede training;
   do not infer them from outputs. Missing independent requirements fail closed.
   The recipe also requires `development_role_identity`: the exact frozen
   development-role identity used by its development metrics. It is not inferred
   from a trace label. Missing identity blocks prelaunch and output inspection.
2. Package through the existing core. Call `RunContext.for_training(spec,
   package_dir, expected_package_identity=..., arm_id=..., account=...,
   run_reference=...)`. Account/reference are frozen declarations. Platform
   version is null until the platform returns it; never guess a version.
3. Construct `FamilyOutputSession(output_dir, context, adapter=..., contract=...,
   trajectory_id=..., metric_semantics=...)`. The exact recipe bytes are retained.
4. After a completed epoch, call `epoch_finished(epoch=..., optimizer_step=...,
   sample_presentations=..., **canonical_metrics)` using actual processed work.
   Epochs are one-based. Semantics specify role, target, units and live/EMA weights.
5. At the owning rule's selected endpoint, call `selected(model_state=...,
   epoch=..., optimizer_step=..., weights=..., prediction_eV=..., target_eV=...,
   source_idx=...)`. Aligned 1-D payloads and selected state retain arm context.
6. At a durable checkpoint, call `checkpoint` with actual model, optimizer,
   RNG, scheduler/EMA, step and presentation counts. Cursor fields are `epoch`,
   `next_batch`, `sampler_order_sha256`, taken from acknowledged work, never
   DataLoader prefetch. `tensor_safe_rng_state` translates the existing capture
   helper's NumPy array to primitive data for `weights_only=True` loading.
   The owning resume loader can pass that state to `restore_rng_state`.
7. Call `complete(runtime=..., hardware=..., observed_costs=...)`. Required
   runtime keys are `platform`, `account`, `precision`, `source_commit`, and
   `source_archive_sha256`; optional `runtime_certificate_id` is a SHA256
   identity. Without `observed_costs`, the hook records measured session
   **process wall seconds** and missing allocated-device seconds. A family
   trainer may pass separately observed native costs. After resume, report only
   the measured process segment and retain other attempts' costs separately;
   never label one segment a complete-trajectory GPU cost.

The `output_manifest.json` `costs` array records native observations. Each
entry requires `metric`, `unit`, `value`, `status`, `semantics`, and `hardware`;
optional `scope` and `reason` retain how the measurement applies. Supported
metrics are `wall_seconds` / `seconds` / `process_wall` and
`device_seconds` / `seconds` / `allocated_device` or `device_busy`. Status is
`measured`, `estimated`, or `missing`; a missing value stays null and its reason
is retained. K1's training invocation records measured process wall and
allocated-device seconds with an explicit scope. GPTrans `run_screen_arm`
records current-invocation process wall and allocated-device seconds on a fresh
invocation; its scope excludes bootstrap, queue time and prior history. After a
resume, allocated-device cost remains missing when prior allocation segments
have no retained ledger. Conservative normalization of older observations also
keeps unavailable allocated-device cost missing. These are invocation costs,
not a complete multi-attempt trajectory total or a device-busy measurement.
Canonical RML cost conversion retains its existing
[cost owner](../../src/molgap/research_memory/cost.py).

Pretraining/downstream use separate `StageRecorder` identities and semantics.
This first inspector covers completed downstream epochs. It does not qualify
pretraining, execute a resume, or prove checkpoint predictions by model replay.
Failed, partial, cancelled or NO_TRAIN jobs use the existing terminal route and
truthful blockers rather than this completion hook.

## Output protocol and checks

`output_manifest.json` format is `molgap-family-output-v1`: context, adapter,
progress, runtime, native costs and five distinct artifact roles: `contract`,
`trace`, `predictions`, `selected_model`, `resume`. Each has a relative POSIX path
and exact-byte SHA256. Metadata uses the existing RML `json_bytes` encoder
(UTF-8, sorted keys, finite numbers, LF) and the launch receipt's atomic
no-overwrite publisher. Identical retry is allowed; conflicting metadata fails.
Retain new protocol bundles under a `family_outputs/` subdirectory of the
owning `platforms/_records/` attempt/arm. The scoped `.gitattributes` rule keeps
their hash-bound bytes unchanged across Git checkout line-ending conversion.

The trace requires one `observation` per completed epoch with observed steps and
presentations. Checkpoint, resume and terminal events are retained as metadata;
they neither increment epoch count nor supply completed-epoch exposure counters.
All declared development metrics require MAE, Gap, eV, minimize, the frozen
development-role identity and the matching live/EMA weights. Selected epoch/step
and live/EMA metric must match the trace row
and aligned prediction MAE. Sampler cursor, RNG and resume metadata are checked;
scientific validity and an executable resume are separate qualifications.

`tensor_digest(tensor, role="source_idx" | "target")` hashes little-endian int64
rows or float64 targets. Freeze requirements with these same bytes; a legacy
float32 tensor/file hash is a different identity. File hashes always bind the
exact serialized bytes.

## CLI and RML bridge

Registered preparation retains the checked acceptance plan and all explicit
artifact pins under the source mount's `acceptance/`. Its launch configuration
binds `target_identity.plan_path` and `plan_sha256`; release and worker startup
validate those files before execution. The dispatcher supplies that explicit
binding to the existing output inspector, including retained trainers whose
completion call predates this field. `FamilyOutputSession.complete()` also
accepts `target_identity` directly. Frozen float32 target manifests require
original finite float32 tensors; no encoding is inferred from output or cast
to make an expectation pass. Missing bindings retain the float64 default.

The [local CLI](EXPERIMENT_CLI.md) adds:

```text
check-acceptance --spec SPEC --repo-root ROOT --plan PLAN
inspect-output --spec SPEC --package PACKAGE --expected-package-identity SHA
  --receipt RECEIPT --arm ARM --artifact-root OUTPUT --expectations EXPECTED
accept-terminal --spec SPEC --package PACKAGE --expected-package-identity SHA
  --receipt RECEIPT --repo-root ROOT --descriptor DESCRIPTOR --outputs OUTPUTS
```

Add `--execute` only to run the authorized terminal transaction. Otherwise it
reads and checks inputs. Acceptance context is reconstructed from the existing
verified receipt, requiring complete physical-arm mapping and an observed
version. Unknown producer version remains null in the report, not backfilled.
`MECHANICALLY_VERIFIED` is not scientific acceptance, runtime qualification or
replay readiness; those fields remain `NOT_EVALUATED` here.

Plan format `molgap-family-acceptance-plan-v1`: `spec_identity` and ordered `arms`.
Each arm has `arm_id`, `adapter`, `expected`, `contract`, `comparison_prelaunch`,
`reference_bundle`, `reference_artifacts`. Contract/comparison/bundle pointers
use `{path, sha256}`. Reference artifact names are the existing
`REQUIRED_REFERENCE_ARTIFACTS` plus `predictions`, each with `{path, sha256}`.
Existing comparison/reference validators apply; row/target/prediction identities
must agree. This availability check does not prove trainer execution or causal
qualification. Missing strict reference blocks rather than triggers retraining.

The modular K1 and GPTrans candidate workflows pin their retained family
reference through this plan by default. Add a reference training arm only when
the owning contract explicitly justifies retraining; preparation never silently
recreates a reference.

`outputs` maps every arm to `{output_dir, expected}` under the selected repo.
`build_verified_terminal_descriptor` accepts those inputs and existing per-arm
`locations`: `trajectory_id`, `run_id`, `trajectory`, `terminal`, `trace`.
It translates repetitive observed bindings. Canonical scientific decision,
acceptance, V5, role, cost and terminal records retain their existing owners.
The closure trace must be exactly the inspected trace path and bytes.

Every arm is inspected before a terminal write. Existing closure then uses
independent durable arm transactions; it is not an all-or-nothing multi-arm
transaction. Reconcile partial completion before retrying. Rebuild/check and
two independently qualified replay entries are still required for pair readiness.

## Retrieval and a new family

Executable family registration and the same-family addon boundary are owned by
the [modular workflow](EXPERIMENT_WORKFLOW.md). This section owns the output
profile and event interface.

The platform skill first reconciles the job/source/version and retrieves/pins
the small manifest. Kaggle's adapter receives an authenticated owning account,
receipt context, that manifest/path/hash and a dedicated destination. It lists
metadata with pagination, then streams only the five required files and manifest.
Missing files, wrong accounts and hash conflicts fail closed. A diagnostic log
requires a separate justified action. The latest-session API cannot select a
version; matching a slug alone never proves an attempt.

For a new compatible family, register the output `ArtifactAdapter` with its
family/version and resume requirements, then add the family's event hooks to
its shared trainer. Pair it with the static execution and package-source
registrations described in the modular workflow. Genuine differences in
sampling, selection, roles, data or phase semantics need a reviewed adapter
extension. A constructor, arbitrary import string or synthetic schema/output
test does not make a trainer GPU-qualified or scientifically accepted.
