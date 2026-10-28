# Configured PPGT / ESA workflow

This extends the ordinary `molgap-scientific-experiment-v1` front door. It uses
PPGT execution interface `ppgt_small/3` and ESA execution interface
`esa_small/4`, the existing models, indexed role loader, FamilyRecipe, Session,
source package, Kaggle journal and saved-output inspector. No replacement
training loop, optimizer, sampler, recovery engine or platform client is added.

## Scientific input

Use the fields and invocation in [the quickstart](EXPERIMENT_QUICKSTART.md).
For each Modern2D arm:

- `model` selects one of the two execution interfaces above. Ordered catalog
  addons are compiled normally; `ppgt_readout_scale/1` and
  `esa_readout_scale/1` are the existing pre-readout examples.
- `recipe_ref` and `data_ref` identify an existing frozen declaration with
  accepted indexed train100K or train500K plus Common-Dev50K. The source recipe
  supplies the model configuration, normalization, initialization and schedule.
- `reference_ref` locates that declaration's pinned input and target metadata.
  Its name is retained for input compatibility; it does **not** imply that the
  model is an accepted scientific reference.
- `training` permits `learning_rate`, `weight_decay`, `horizon_passes`,
  `profiling` and an explicitly frozen `execution_mode`.
  The compiler derives optimizer steps from complete passes and BS128, retains
  the schedule kind, and validates its constraints (the four-stage warmup
  schedule requires more than four passes).
- `execution_mode="batched-v1"` is PPGT-only and requires a new configured
  recipe plus a pinned accepted native qualification in the authority.
  The compiler packages that evidence; historical recipes keep the original
  graphwise default. The exact new model/recipe still passes native admission.
- Scientific addon screens use `purpose: architecture_comparison`. The first
  contract explicitly declares `same_run.reference_arm_id` and
  `same_run.target_manifest_ref`; its authority also names
  `same_run_reference_arm_id`. The reference keeps its frozen recipe unchanged.
  The compiler freezes both identities and the allowed architecture delta.
  `context_only` is reserved for the explicit execution diagnostics below;
  historical context-only training packages cannot release another screen.
- Successful closure registers the baseline once and returns `accepted_reference`.
  `derive-input --from-run ...` then removes the baseline arm and fills
  `reference_ref.bundle_ref` plus the question's frozen reference ID. Review
  the addon delta and supply fresh authority; the baseline is not trained again.
  This path supports architecture addons with an otherwise matching contract.

One slot can contain one or two independent Modern2D arms, including two arms
of the same family. A single arm needs an explicit reason for leaving the other
allocated T4 idle. Each worker retains its independent state and assigned GPU.
Mature fixed-data and Modern2D indexed-data families use separate workflows.

A separately frozen same-run architecture pair may also use the existing
explicit update/evaluation clock. Both arms declare `horizon_steps` and
`evaluation_every_steps`; authority must set
`step_clock_architecture_authorized=true`. `horizon_passes` in the editable
input preserves the schedule reference clock; the compiler derives physical
pass count separately. Both arms must match endpoint, evaluation cadence and
all non-architecture identities. This reuses the Session step clock; it does
not turn a historical transfer study into a causal comparison.

The compiler generates the Spec, recipes, prospective inputs, feature bindings
and acceptance plan. The human does not write those envelopes or compute their
hashes. Feature identity binds topology/role/transform parameters separately
from model/addon identity. Generated metadata stays compact; graph and
checkpoint payloads stay outside Git.

## Explicit resource and role binding

New screens use `molgap-modern2d-configured-authority-v1` with purpose
`CONFIGURED_SCREEN`. The declaration records the actual server run, ordered
arm IDs/interfaces, seed42, BS128, accepted roles, per-arm `recipes` horizons
(`horizon_passes` and `horizon_steps`), and the full two-T4 wall/GPU-hour ceiling.
The `gpu` object names the account owner, `NvidiaTeslaT4`, two allocated devices,
wall ceiling and the same allocated-hour ceiling. Both devices count, even for
a single arm. A 600-second checkpoint reserve and the existing hard-stop guard
apply within that ceiling. The configured wall range is 660–43200 seconds.

Required positive flags are `user_authorized` and
`scientific_training_authorized`; `owner` is `server`. The following remain
false: `protected_role_access_authorized`, `extra_seed_authorized`,
`automatic_successor_authorized`, and `full_training_authorized`.
`scale_500k_authorized` is true only for an explicitly authorized train500K
declaration. Neither a 100K MAE nor an early-stop heuristic decides this flag.
`single_arm_reason` is required when there is one arm.

Historical formal60-pass authorities and canceled attempts are not compatible
with this format. New configuration support itself creates no compute authority.
The existing workload adapter still owns any separately authorized publication,
submission, uncertain-receipt reconciliation and selective retrieval.

## Preparation, runtime and acceptance

The usual platform `--experiment` command prepares locally without SDK calls.
Preparation validates recipes, accepted-role metadata, frozen initial weights,
source closure and the new acceptance plan before publication. Continuing a
completed run keeps its original package and source across HEAD changes.

The runtime extension reads the configured budget. It builds the existing
formal context with an explicit configured flag, checks authority and both role
permits before payload access, and validates decoded feature identities. Fresh
all-arm native repeat/resume qualification remains mandatory. Passing CPU
fixtures does not supply a native certificate.

Session outputs retain selected/last weights, optimizer/RNG state, predictions,
canonical trace, its context companion and the observed `role_audit.json`.
Actual loader, metric and selection callbacks populate the audit; permissions
alone do not establish observed role use. Native certificate/provenance files
are retained as small artifacts when present. Target identity uses the pinned
manifest. The shared inspector verifies saved bytes; pair acceptance consumes
them without inference, closes the reference before the candidate, and checks
actual replay-pool membership. Complete negative outcomes are eligible.
Missing evidence still blocks closure; missing costs remain missing.
Portable recovery uses the existing package/receipt/resume-bundle owners.

Local storage attempt, frozen action attempt and platform version are separate
identities. Terminal evidence uses the frozen action; the launch receipt binds
the physical kernel/version. Recovery never renames the original action.
`accept-run --stage-terminal --outcome ... --decision-ref ... --execute` consumes
the runtime audit automatically. Explicit scientific decisions remain required;
no observed gain grants another run or promotion.

This implementation does not certify native throughput, model quality, a
completed formal training run or Desktop migration. Those require evidence from
their own authorized execution, not a successful declaration or CPU test.

## Explicit scale-transfer studies

A separately authorized `CONFIGURED_TRANSFER_STUDY` keeps the same compiler,
Session, native gates and output protocol. It consumes accepted train500K plus
Common-Dev and uses per-arm pinned original100K contracts in
`authority.transfer_study_ref`. The transfer plan and every source contract are
packaged and checked at release. It cannot declare a same-run causal pair,
retained architecture bundle or STRICT_CAUSAL claim. Such a study retains
context/paired-endpoint evidence; it is not architecture Replay admission.

Its scientific input uses `purpose: transfer_study` and explicitly supplies
`training.horizon_steps` plus `evaluation_every_steps`. `horizon_passes` keeps
the original schedule's reference clock; the profile separately records the
ceiling of actual500K passes and exact terminal cursor. Source LR trajectory,
optimizer, model/addons, target transform and endpoint must match the pinned
100K contract. Evaluations/checkpoints occur at the frozen update interval,
including within a physical pass. The ordinary architecture path retains its
complete-pass semantics and cannot opt into this clock.

Selected CPU metadata accepts either training scale plus Common-Dev. An explicit
`normalization_reference` pins the source100K member/order digest and constants;
both metadata preparation and the training loader recompute them from the actual
nested100K prefix of500K labels. Merely copying recipe constants is insufficient.
Optional typed `model_configs` keeps the actual retained budget model's feature
bindings rather than replacing it with the author's larger default. These
capabilities grant no data publication, training authority or promotion.

## Execution-only PPGT profiling

The same input/recipe/Session/package/platform owners support a separately
authorized `CONFIGURED_EXECUTION_PROFILE` purpose. It requires
`scientific_training_authorized=false`, one PPGT/3 arm, train100K, six scratch
updates and at most1800 seconds/1 allocated T4-hour. The phase extension contains
**preflight only**; the runtime rejects development/formal training access.
Both allocated devices are charged, including the deliberately idle second T4.

`training.profiling` selects the existing profiler. `ppgt_phases=true` adds
temporary PPGT function/module observations without changing model arithmetic.
Disable shape, stack and memory capture for the bounded native profile. The
two-update captured repeat must equal the unprofiled and serialized-resume
endpoints; report timing overhead rather than treating it as native throughput.
The [owning protocol](../../experiments/modern2d_configured_100k/ppgt_profile/protocol.md)
defines outputs, roles, bounds and interpretation. Profiling is not a scientific
screen,500K release or training Replay claim.

## Prepared execution qualification

New explicit configured recipes may select `execution_mode: prepared-v2`
for PPGT/3 or ESA/4. Their authority must select the same mode and set
`fresh_native_execution_gate: true`. Historical native reports cannot release
this implementation. The ordinary all-arm preflight barrier runs repeat and
serialized-resume scratch updates plus an accepted-backend shadow. Saved model
and Adam updates must satisfy the existing tolerance; cursor/schedule/RNG and
optimizer metadata must agree exactly. The training entry rereads hash-bound
proof states before issuing an in-process, recipe-bound execution certificate.
Plain JSON or a changed recipe/certificate cannot activate prepared training.
The existing Session, profiler, output owners and budget guard remain intact.
The [bounded two-family protocol](../../experiments/modern2d_configured_100k/prepared_pair/protocol.md)
defines the authorized run, not this capability description.
