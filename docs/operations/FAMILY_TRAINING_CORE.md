# Shared Family Training Core

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
| `packed_graph_dataset.py` | Spawn-pickleable packed PyG payload owner, optionally memory mapped. |
| Existing K1/GPTrans trainer | Family loss/addon, frozen schedule, selection and artifact hooks during migration. |
| [Family outputs](EXPERIMENT_FAMILY_WORKFLOW.md) | Existing checkpoint, trace, artifact inspection and V5/RML closure. |
| [Platform routing](../../platforms/README.md) | Skills/adapters; no model or optimizer implementation. |

## Data Boundary

`DatasetSelection(manifest, manifest_sha256, development)` uses a confined,
relative manifest path. `load_training_data(root, selection)` resolves that
selection under a caller-supplied local mount. The mount path is not scientific
identity; the pinned manifest and its shard bytes are. It accepts the existing
`molgap-pcqm4mv2-fixed-subset-v1` format over OGB atom9/bond3/RWSE16 topology.
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

New checkpoints acknowledge `cursor.optimizer_state_ids`: missing acknowledged
AdamW moments fail before restore. Unconditional modes require full parameter
coverage; SSMA may omit untouched conditional-residual moments, never base
moments. Historical checkpoints without this field keep their frozen owner;
do not synthesize metadata to migrate them to a new source package.

Selected weights, predictions and pass traces are staged as a hash-bound bundle.
`last_checkpoint.pt` is published last as the commit point. The owning exclusive
writer validates the complete journal before accepting or rolling back interrupted
publication; unknown/corrupt evidence fails closed. Transfer the whole interrupted
output directory, including `session_pending/` and `session_commit.json`, not just
the checkpoint. This is not a concurrent-reader or power-loss guarantee.

Selected/resume tensor keys, shapes and dtypes must agree. Cost invocation markers
bind closed cumulative wall time to the acknowledged checkpoint; an interrupted,
stale or unbound prior interval stays unknown, not measured. Platform allocation
costs remain separately owned. Only new partial-pass metric scalars are transferred
to the private CPU prefix, with unchanged dtype and sequential reporting order.
See the [server recovery review](SHARED_RECOVERY_SERVER_REVIEW.md) for provenance,
server-specific transaction hooks, executed checks and speed-evidence limits.

Development selection consumes only the explicitly selected internal role at a
complete pass. Final-only selection never opens a development loader or creates
a development score. An unfinished pass is not reported as a finished epoch.
The session does not write a new evidence format: callbacks use the owning
checkpoint/trace/selected-output owners. In particular, the legacy screen output
profile cannot represent a full final-only endpoint; do not manufacture empty
predictions or development metrics to make that inspector pass.

## Migration Status

On server, new K1/GPTrans interfaces use the shared core/session. The legacy
K1 screen delegates its optimizer update only; retained K1 scale/full and
GPTrans screen entrypoints keep their historical loops, runtime gates and
execution-backend choices. The desktop full GPTrans owner is available as
reviewed library code, not server full-training authority. Historical jobs
still require matching immutable source; do not rewrite frozen source pins.
See [server integration](SHARED_TRAINING_SERVER_INTEGRATION.md) for differences.

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

Batch-topology reuse and the selective desktop port are documented in the
[server review](TOPOLOGY_REUSE_SERVER_REVIEW.md). K1 shares a private padding
mask across mixers; prepared GPTrans reuses local destinations and bounds
reachability without altering learned tensors or the loss/update sequence.

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
