# Family output and acceptance hooks

The shared module connects an owning trainer's outputs to mechanical inspection
and existing terminal/RML closure. Training, submission, runtime qualification
and scientific decisions remain with their existing owners.

## Capability matrix

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

Static profiles: `k1-v1` for `neural_atom_k1/1`, `gptrans-v1` for `gptrans_t/1`.
K1 requires scheduler state; GPTrans requires EMA state. Both require model,
optimizer, Python/NumPy/Torch/CUDA RNG and the acknowledged sampler cursor.
These profiles cover the **new output protocol**. Existing historical runners
are not automatically migrated. Keep their frozen payloads and acceptance
loaders intact. Missing historical counters and wrong account labels remain
missing/wrong, not repaired by relabeling.

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
7. Call `complete(runtime=..., hardware=...)`. Runtime keys are exactly
   `platform`, `account`, `precision`, `source_commit`, `source_archive_sha256`.
   The hook freezes a manifest and inspects outputs. Cost is measured session
   **process wall seconds**; device allocation/busy seconds remain null. After
   resume it measures only that process segment. Retain other attempts' costs
   separately; never label this value complete-trajectory GPU cost.

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

The platform skill first reconciles the job/source/version and retrieves/pins
the small manifest. Kaggle's adapter receives an authenticated owning account,
receipt context, that manifest/path/hash and a dedicated destination. It lists
metadata with pagination, then streams only the five required files and manifest.
Missing files, wrong accounts and hash conflicts fail closed. A diagnostic log
requires a separate justified action. The latest-session API cannot select a
version; matching a slug alone never proves an attempt.

Add a genuine shared model/trainer, reviewed Spec contract and one static
`ArtifactAdapter` entry for a new compatible family. Hook this session into its
events and test the family-specific state/weight behavior. Genuine differences
in sampling, selection, roles, data or phase semantics need a reviewed adapter
extension. A constructor, arbitrary import string or schema test does not make
an unsupported trainer executable or scientifically qualified.
