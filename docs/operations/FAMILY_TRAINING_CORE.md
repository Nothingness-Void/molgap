# Shared Family Training Core

## Pinned input resolution

`family_training_data.resolve_training_manifest` keeps dataset identity separate
from the platform's mount location. An existing declared path must pass its
frozen SHA; a missing path requires exactly one metadata file with that SHA.
The loader resolves shard paths relative to the accepted manifest directory.
Ambiguous, missing or changed identities fail before graph deserialization.
Final shared-family release checks bind the manifest and exercise the packaged
resolver on flat/nested mounts. Kaggle startup retains `input_resolution.json`
and prints `INPUT_RESOLVED` before the native all-arm barrier.

Scope: K1 and GPTrans-T. Platform skills own submission, remote allocation,
reconciliation and retrieval. Data adapters own pinned graph views. Neither
layer may copy or alter the training algorithm to fit a platform or scale.

## Code Owners

| Owner | Responsibility |
|---|---|
| `family_training_core.py` | One outer loop, zero-grad/backward/clip/optimizer/EMA update order and batch-stream execution; tensors stay on-device until an owning reporting boundary. |
| `family_training_data.py` | Fixed-subset topology selection, shard SHA/row/label verification, train-only normalization and parameterized deterministic samplers. |
| `family_training_recipe.py` | Typed data/initialization/budget, schedule, tail and selection policies; no platform-dependent algorithm or slow execution option. |
| `family_training_kernel.py` | Existing K1/GPTrans model/addon factories, accelerated forward/backward, optimizer, bound EMA and existing buffered development evaluation. |
| `family_training_loaders.py` | One persistent train loader and bounded optional CPU development cache; private loader RNG and deterministic pass/resume cursor. |
| `family_training_session.py` | One schedule/train/development/selection/resume lifecycle, using the shared outer loop and caller-owned atomic artifact/trace hooks. |
| `family_training_runner.py` | Registered contract/source/role bindings, native repeat/resume qualification and the single worker entry. |
| `family_training_outputs.py` | Existing atomic checkpoint/trace translation and shared CPU inspection for development-selected or final-only endpoints. |
| `family_execution_profile.py` | [Optional session profiler](FAMILY_EXECUTION_PROFILE.md): phase and operator observation; no alternate training implementation. |
| `packed_graph_dataset.py` | Spawn-pickleable packed PyG payload owner, optionally memory mapped. |
| Existing K1/GPTrans trainer | Family loss/addon, frozen schedule, selection and artifact hooks during migration. |
| [Family outputs](EXPERIMENT_FAMILY_WORKFLOW.md) | Existing checkpoint, trace, artifact inspection and V5/RML closure. |
| [Platform routing](../../platforms/README.md) | Skills/adapters; no model or optimizer implementation. |

## Data Boundary

`DatasetSelection(manifest, manifest_sha256, development)` uses a confined,
relative manifest path. `load_training_data(root, selection)` resolves that
selection under a caller-supplied local mount. The mount path is not scientific
identity; the pinned manifest and its shard bytes are. It accepts the existing
`molgap-pcqm4mv2-fixed-subset-v1` and the accepted legacy
`molgap-pcqm4mv2-kaggle-fixed-subset-v1` formats over OGB atom9/bond3/RWSE16.
The legacy mirror keeps its original manifest/shard identity, requires explicit
false protected-role flags and reuses the existing topology-only view to strip
cached geometry before batching. No new geometry is built or used by the model.
100K, 500K and full differ in their manifest counts, not in the loading code.

Only selected topology roles are opened. Development is optional and confined
to an official-train-derived range disjoint from training. Geometry assets,
official validation and test assets are not traversed. Mean/sample-standard-
deviation come exclusively from selected training labels. This is identity
validation, not role authority or permission to unpickle an untrusted cache.
The computed statistics are preparation observations, not runtime authority.
`FamilyRecipe.target_mean_eV` / `target_std_eV` explicitly freeze train-only
normalization; execution never replaces them with a device/CPU-thread-dependent
reduction. Reuse the owning accepted normalization asset when its contract
requires it (including the historical K1 100K FP32 constants). A new selection
must freeze its train-only statistics before release; development never supplies
normalization.

`PassBatchSampler` receives rows, batch size, seed, algorithm, pass and resume
offset. It supports the existing Python epoch-shuffle and Torch pass-randperm
policies, an explicit drop/retain-tail policy and private RNG. `FamilyLoaders`
retains one loader across passes and reuses the existing bounded development
cache. K1 retains RWSE16; GPTrans retains prepared attention layout. Cache
overflow streams the entire role rather than truncating it. Stochastic dataset
transforms are rejected; this does not promise worker-local random augmentation
replay. Historical runners retain their own frozen RNG behavior.

## Session Boundary

`FamilyTrainingSession(kernel, loaders, checkpoint=..., selected=...,
pass_finished=...)` consumes matching typed recipe/data bindings. Callbacks are
trusted static adapter hooks, not imports selected by JSON. `run(should_pause=...)`
uses exact optimizer steps, not a scale-specific epoch loop. Mid-pass checkpoints
retain model/optimizer/schedule/EMA/RNG, recipe identity, exact sample/cursor and
partial-pass metric order. A different budget/recipe is not a resume. Publication
must complete before a pause or endpoint is acknowledged; failure propagates and
worker/cache cleanup runs. `load_state_dict` validates identity/exposure and
rolls back state if restoration fails.
Resume validation also checks frozen AdamW defaults, parameter/moment schema and
the exact schedule axis/configuration. LR validation permits native cosine's
double-precision recurrence roundoff only; it does not change stored LR.
Transported initialization must match base tensor keys, shape and dtype without
silent conversion. Declared and observed accelerator classes must match before
native qualification; resource matching never changes the training algorithm.

New session checkpoints acknowledge `cursor.optimizer_state_ids`. Missing
acknowledged moments fail before restore; unconditional modes require complete
parameter coverage. SSMA may have untouched conditional-residual parameters,
but missing base moments remain invalid. Historical payloads lacking this field
must use their frozen source, never gain invented participation metadata.

`SessionOutputs` stages selected weights/predictions and the next trace until
checkpoint publication. A hash-bound `session_commit.json` records retained
previous files; publishing `last_checkpoint.pt` is the commit point. The next
owning invocation validates all bindings before rolling back an interrupted
bundle or accepting a committed one. Unknown/corrupt backups fail closed; no
blind trace truncation. Ordinary mid-pass saves keep the single atomic-write
path. One exclusive producer owns the output directory; it is not a live-reader
transaction API, platform backup or power-loss certification.

Development-selected weights must match resume tensor keys, shapes and dtypes,
not only finite values and a file SHA. The runner starts a cost invocation marker
after metadata admission, before input/GPU work. A missing checkpoint binding or
unfinished prior invocation makes cumulative wall cost unknown; it is never
relabelled measured. Queue/device allocation costs remain platform-owned.

Development selection consumes only the explicitly selected internal role at a
complete pass. Final-only selection never opens a development loader or creates
a development score. An unfinished pass is not reported as a finished epoch.
The session does not write a new evidence format: callbacks use the owning
checkpoint/trace/selected-output owners. In particular, the legacy screen output
profile cannot represent a full final-only endpoint; do not manufacture empty
predictions or development metrics to make that inspector pass.

## Checkpoint Checks

`training_reproducibility.nonfinite_tensor_keys` owns tensor finite checking.
The nested session/output checker reuses it: accelerator flags cross to CPU once
per device and CPU scalars are checked in same-dtype groups. Every tensor and
host float still participates; no cached success or skipped optimizer/EMA guard.
That checker alone did not change state format, update math, metric addition
order or checkpoint policy. New packages freeze shared code; old jobs keep source
pins. The later participation field above is an explicit checkpoint extension,
not a silent conversion of historical artifacts.
The [synthetic checkpoint diagnostic](../../experiments/family_checkpoint_efficiency/README.md)
owns measurements and their limits, not native training/quality qualification.

## Execution Cost Boundaries

Partial-pass metrics keep an immutable private CPU float prefix after each
persistence boundary; only newly observed scalars cross devices. Snapshots build
fresh CPU tensors, while epoch reporting adds original scalar values in Python
batch order. This does not cache validation success, alter loss or narrow dtype.
The prepared GPTrans backend omits the unused final reachability expansion;
bounded integer distance codes and all used operations remain unchanged.
K1's private index/mask copies use non-blocking transport on the owning stream;
the synthetic RTX5060 test preserves output, gradient, optimizer and RNG exactly.

The [code-level repair and speed audit](FAMILY_RECOVERY_SPEED_AUDIT.md) owns tests,
operation counts and remaining candidates. These changes are not a measured epoch
speed/MAE claim or native T4/A100/DCU qualification. Sampler changes, model algebra
reordering, async writes and topology-cache changes require separate validation.

## Migration Status

The K1 screen, K1 scale/full and GPTrans screen/full now delegate optimizer
updates and outer-loop execution to the shared core. Packed payload decoding also
uses one top-level owner. Their frozen role/schedule/selection and output
contracts remain separate; sharing a primitive does not authorize new input
or convert a historical recipe into a new execution identity.

Cross-scale code is registered as `neural_atom_k1/3` and `gptrans_t/2`, with
`k1-session-v1` / `gptrans-session-v1` output profiles. These are interface
versions, not model generations, a V3 scientific contract or a change to V5.
Historical fixed100K interfaces remain compatible; historical full-source pins
are unchanged. Old jobs use matching immutable source, never rewritten hashes.

The [test handoff](FAMILY_TRAINING_TEST_HANDOFF.md#external-acceptance) owns CPU
acceptance and remaining native qualification limits. CPU acceptance is not a
native speed/quality claim or an integrated release. Commit source before
packaging. Scientific comparison,
full admission and terminal V5/RML closure remain their owning gates, not
consequences of preparation or mechanical inspection.

## Public Entry

Build `FamilyRecipe`, then use `experiment_execution.build_family_recipe` with
`configuration=recipe`, the applicable interface tuple, exact train/optional
development usage digests and development row/target hashes. Final-only selection
uses null development fields. `family_arm_bindings(recipe)` supplies current
base-source, sampler, normalization and objective references for the Spec. Freeze
this before generating prospective actions. Normalization belongs to the recipe;
the loader's computed statistics are preparation observations, not authority.

The local CLI `prepare-family-training --spec ... --plan ... --repo-root ...
--output ...` calls `experiment_workflow.prepare_family_training`. The plan has
exactly `format="molgap-family-training-preparation-v1"`, `spec_identity`,
`source_files` and `arms`. Each arm has `arm_id`, `device`, `recipe` (relative
JSON) and `initial_state` (existing local file). Use a fresh ignored staging
directory. `prepared_training.json` provides worker jobs, package identity and
prospective hashes. Preparation neither submits nor supplies full admission.

The platform skill supplies account/observed job identity, mounts, isolated
devices, scheduler/Notebook metadata, platform release and durable retrieval.
It invokes the same `experiment_training_worker` preflight/train phases and
`execute_training_phase` on local/Kaggle/IMS/SCNet/Colab. Separate source/data roots
are supported. No new submitter, polling or takeover is introduced. The older
Kaggle-only `prepare-workflow` retains its strict development-comparison plan;
it is not renamed into a supposedly universal submission command.

Launch wall budget and cleanup reserve are configuration, not training policy.
See [Kaggle budget fields](../../platforms/kaggle/README.md#runtime-budget-configuration);
the shared launcher does not impose the old four-hour experiment ceiling.

Native preflight consumes authorized training topology only. It retains two
independent two-step endpoints, an actual serialized one-step resume origin and
its resumed endpoint. Formal training verifies bytes and recomputes complete
model/optimizer/schedule/EMA/RNG fingerprints; `accepted=true` alone is not
authority. Qualification binds exact source/recipe/data/runtime/device/resources,
not a platform name. Scratch work consumes resources but contributes no formal
training exposure. CPU fixtures cannot qualify GPU/DCU work or native speed.

On SIGTERM the worker pauses after the acknowledged batch, retains checkpoint
and pause receipt, and never calls partial work a complete endpoint. A completed
retry only inspects artifacts, without model construction or molecular data
loading. Missing earlier cost accounting stays unknown; platform allocation
seconds are not invented from process wall time.

## Addons And Closure

A new addon adds a small model/loss hook and static mode/Spec/execution/source
registration, not a trainer, optimizer, loader or platform script. JSON cannot
select arbitrary imports. Current adapters allow one reviewed addon mode per
arm; new combinations require explicit registration and a new frozen recipe.

Mechanical verification keeps science/runtime/replay `NOT_EVALUATED`. Final-only
outputs have no predictions or development score; scientific/full-scale closure
uses the owning full-validation workflow. Never pad them with fake predictions
to satisfy the legacy development-comparison terminal adapter.

## Verification

Tests use temporary synthetic CPU graphs and elementary models only:
`test_family_training_core.py` checks optimizer/gradient/RNG equivalence, step
ordering, exact stop boundaries and cleanup. `test_family_training_data.py`
checks mount-independent identity, selected-role confinement, unchanged sampler
orders, hash failures and train-only normalization. Native hardware speed,
quality, role release and replay readiness require their own existing gates.
`test_family_full_lifecycle.py` exercises both real full-run wrappers with
synthetic CPU kernels and explicitly substituted release gates, including exact
checkpoint/model/optimizer/EMA/RNG restoration. It does not qualify native source
or turn the historical full-source pins into current valid declarations.
`test_family_training_loaders.py` covers persistent CPU worker order/resume,
tail policies, role omission, bounded cache overflow and input preservation.
`test_family_training_session.py` covers synthetic dropout/AdamW/EMA replay,
all supported schedule types, partial-pass metrics and failed restore/publication.
These tests do not authorize a job or claim native acceleration at a new scale.
