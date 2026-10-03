# Family output and acceptance hooks

The shared module connects an owning trainer's outputs to mechanical inspection
and existing terminal/RML closure. Training, submission, runtime qualification
and scientific decisions remain with their existing owners.
For server-side prospective planning, packaging and release preparation, use
the [standard variant workflow](EXPERIMENT_WORKFLOW.md); this guide owns the
producer-output and post-run half, not another submission implementation.

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
| `graph-screen-v1` | Shared pure-2D Gap output profile; the [graph extension contract](GRAPH_SCREEN_EXTENSION.md) owns registration and the small model/addon interface. |

Static profiles: `graph-screen-v1` for a reviewed shared graph registration,
`k1-v1` for `neural_atom_k1/1`, `gptrans-v1` for `gptrans_t/1`, and
`edge-state-v1` for `edge_state_gps/1` (live weights and scheduler state).
The graph profile requires the pure-2D normalized-Gap contract: feature schema
`ogb-atom9-bond3-rwse16-v1`, roles `train` and `development`, recipe
`graph_gap_screen_v1`, sampler `seed42-epoch-global-randperm-v1`, and transform
`train-mean-unbiased-std`. The static registration and its focused tests are
the capability gate; a declaration alone does not enable the route.
K1 requires scheduler state; GPTrans requires EMA state. Both require model,
optimizer, Python/NumPy/Torch/CUDA RNG and the acknowledged sampler cursor.
These profiles cover the **new output protocol**. Existing historical runners
are not automatically migrated. Keep their frozen payloads and acceptance
loaders intact. Missing historical counters and wrong account labels remain
missing/wrong, not repaired by relabeling.

## Integrate once in an owning trainer

For the supported server screens, reuse
`experiment_training_hooks.bind_training_outputs` rather than hand-writing
event translation. `pcqm_gptrans_v4.run_training` and
`pcqm_k1_variants_runner.train_arm` accept its result as the optional
`family_outputs` argument. Default `None` retains the old runner path and old
artifact format. Hook qualification is narrow: GPTrans reference/degree-scale/
chemical-path arms with live/EMA audit; unchanged direct-Gap K1-v4 baseline.
Model-only registry addons and joint-objective/pretraining are not silently
redirected into those trainers. Ordinary EdgeState now has an artifact profile,
but its experiment still owns its epoch loop and must supply actual events.

The new recipe must also pin `trainer_binding` with exactly `name` and
`variant`: `pcqm_gptrans_v4` + the actual supported variant, or
`pcqm_k1_variants_runner` + `neural_atom_k1_v4`. EdgeState primitive integration
uses `edge_state_training_core` + `experiment_owned`; that is not training
qualification. A binding helper never releases compute.

The shared hook checks the frozen source/trajectory/exposure before training,
emits one-based completed epochs alongside unmodified zero-based native rows,
and records the completed permutation hash. The cursor denotes the epoch
boundary (`next_batch=0`); its order SHA describes the **completed** permutation,
not a claim that the next epoch ran. Owning native checkpoints retain the hook
identity. Resume requires that checkpoint and the complete retained family
trace prefix; neither legacy traces nor interrupted missing events are inferred.
The binding helper checks `output == native_output / "family_outputs"` before
writing; the owning runner checks it again against its actual native directory.
Restore both protocols together. Resume validation compares model/EMA,
optimizer, scheduler and tensor-safe RNG state across their two checkpoints;
the owning native checkpoint remains the executable resume authority. These
checks do not prove resumed model execution. Ten-epoch recovery chunks retain
both protocols separately.

```python
hooks = bind_training_outputs(
    spec, package_dir=package_dir, expected_package_identity=package_identity,
    arm_id=arm_id, account=account, run_reference=declared_reference,
    output=output / "family_outputs", native_output=output, contract=recipe_path,
)
# After the existing scientific/cache/runtime/budget gates pass:
run_training(**owning_training_arguments, family_outputs=hooks)
```

Use the prospective trajectory ID in the owning arguments. The submission
receipt independently binds the actually observed platform version during
acceptance. Logical attempt IDs come from the frozen prospective action, not
from a guessed future platform version; multiple attempts require explicit
recovery reconciliation.

Keep the existing model, loader, sampler, trainer and scientific recipe.
Read [the addon guide](EXPERIMENT_ADDON_GUIDE.md), then add event hooks:

1. Freeze the owning recipe JSON with `acceptance_requirements`: exactly
   `epochs`, `optimizer_steps`, `sample_presentations`, `development_rows`,
   `source_idx_sha256`, `target_sha256`, `precision`. Pin the whole file in
   `Spec.arms[].training.recipe.sha256`. Requirements must precede training;
   do not infer them from outputs. Missing independent requirements fail closed.
   Also freeze `metric_semantics` in that recipe using the canonical RML metric
   map (`live_train_metric`, `live_dev_metric`, `ema_dev_metric`). These new
   profiles use `MAE`, `eV`, target `Gap`, direction `minimize` and the field's
   live/EMA weights. Pin actual role identities: train and development must
   differ; live/EMA development must agree. Unused EMA may be explicitly null.
   Session and inspector compare the full definitions to these recipe bytes;
   historical metric labels are not silently normalized or rewritten.
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

The trace requires one observation per completed epoch with observed steps and
presentations. Selected epoch/step and live/EMA metric must match the trace row
and aligned prediction MAE. Sampler cursor, RNG and resume metadata are checked;
scientific validity and an executable resume are separate qualifications.
Checkpoint, resume and terminal events remain in the canonical trace but are
not counted as completed epochs. The completion hook does not synthesize those
events: the owning trainer records actual lifecycle identities through the
existing recorder, without duplicating acknowledged exposure observations.

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

Use `close_family_replay(..., execute=True)` to compose verified closure with
the existing compiler's replay-pool check for the requested trajectories.
It returns `replay_readiness=REPLAY_READY` only if each is present with a
qualified candidate/reference peer under the same comparability key. A complete
terminal can instead return `BLOCKED`; dry runs return `NOT_EVALUATED`.
The standard `accept-terminal` CLI uses this wrapper. Exit zero means the
terminal transaction completed, not scientific/replay qualification; callers
must read the separate `replay_readiness` field before claiming a replay pair.
This wrapper does not manufacture comparison, cost, role, runtime or reference
records, select a winner, or authorize a successor. Those remain owning inputs.

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

## Verification boundary

`tests/test_experiment_lifecycle.py` runs real local prospective planning,
staging, source verification, synthetic receipt reconciliation, producer events,
inspection and terminal/RML closure in one temporary repository. It deliberately
checks that missing scientific qualification stays replay-blocked.
`test_experiment_training_hooks.py` checks saved-state events/resume bindings and
actual owning-runner call sites without executing models. Existing same-run
replay tests own positive paired qualification. These are local synthetic tests,
not proof of a real accelerator run or a five-minute submission benchmark.


## Registered K1 screen output

`k1-screen-v1` covers `neural_atom_k1/2` with model, optimizer, scheduler, RNG
and acknowledged sampler cursor. It is separate from historical `k1-v1` and
server `edge-state-v1`. The static [training registry](REGISTERED_EXPERIMENT_WORKFLOW.md)
adds executable owning trainers for its explicitly listed modes, with observed
runtime certificates and scoped invocation costs. The source recipe pins full
metric semantics and exposure before output production. GPTrans's adapter
retains and translates the owning runner's outputs without replacing that runner.
Incomplete terminal translation uses retained scheduler/process facts; missing
progress after a training-start marker remains unknown. This does not qualify
an executable resume, real hardware, a causal comparison or replay eligibility.
