# MolGap common direction v5 — FINAL CONTRACT

Date: 2026-09-17 (Asia/Tokyo)
Document ID: MOLGAP-COMMON-V5-FINAL

This document supersedes all previous common-direction contracts, including V3 and V4. V5 is the stable final architecture contract for future MolGap screening and scale-up unless the user explicitly authorizes a future contract version.

## C0. The correct project topology

MolGap has TWO INDEPENDENT MACHINE WORKFLOWS.

### Server

```text
Server A (high capability)
    -> research analysis
    -> experiment design/freeze
    -> remote submission
    -> Server B (Luna Max, 30-minute heartbeat)
    -> healthy: silent
    -> failure/completion: wake Server A
    -> Server A decides/repairs/submits next justified action
```

Server remains powered on and uses A/B to reduce unattended wall-clock gaps.

### Desktop

```text
Desktop agent while machine is online
    -> research/full/selected 500K work
    -> remote submission when appropriate
    -> machine may power off
    -> NO default monitor
    -> NO server fallback
    -> NO cross-machine wake
    -> when desktop returns, reconcile remote state and continue
```

Desktop shutdown latency is accepted intentionally.

The two sides are connected by Git/evidence integration, not a live conversation control plane.


## C0A. Contract stability and amendment policy

V5 is intentionally designed to stop repeated process redesign.

Agents must distinguish:
- scientific evolution inside the contract; from
- changes to the contract architecture itself.

Scientific evolution is expected. New model families, better diagnostics, calibrated early-stop rules, different explicitly authorized training contracts, new hardware adapters, and improved acceptance code may all be added while V5 remains unchanged.

The following are V5 architectural invariants and should not be casually revisited:

1. `master`, `molgap-server`, `molgap-desktop`, and `archive` retain their current roles.
2. Server and desktop are operationally independent.
3. Only server has the default local A/B loop.
4. Server B is a low-cost 30-minute monitor, not a scientific agent.
5. Server A controls bounded 100K/500K server research within recorded authority.
6. Desktop has no default heartbeat and accepts offline silent time.
7. No default cross-machine live orchestration or takeover exists.
8. Server candidates require qualification before desktop full consideration.
9. Full-scale/evaluation/submission remain desktop-oriented by default.
10. Acceptance, scientific status, transfer status, budget decision, and delivery status remain distinct.
11. Evidence from failed/inconclusive experiments remains discoverable.
12. Resource efficiency is optimized by better decisions and measured engineering, not by weakening scientific contracts or keeping accelerators busy.

An agent may propose changing an invariant, but may not activate the change. The proposal must be explicitly approved by the user as a new contract version.

No agent may reinterpret a local implementation problem as permission to redesign the architecture.

## C1. Why this design is intentional

The two machines optimize different constraints.

Server:
- always available;
- suitable for continuous remote-work supervision;
- server A/B can prevent several hours of avoidable waiting after a server-owned remote run completes.

Desktop:
- may be powered off;
- user accepts silent time;
- does not need continuous operational supervision;
- prioritizes full-scale/evaluation work and selected independent experiments.

Do not force architectural symmetry.

Do not add server monitoring to desktop just because it is technically possible.

## C2. Authority and branch roles

Keep:

| Branch | Role |
|---|---|
| `master` | stable delivery only |
| `molgap-server` | server-owned 100K/500K research, reusable screening infrastructure, server A/B support |
| `molgap-desktop` | full/evaluation/submission plus desktop-owned selected 500K work |
| `archive` | exact inactive/rejected/superseded histories |

Machine, branch, platform, and scientific ownership are related but not identical.

A Kaggle/SCNet/IMS job can be server-owned or desktop-owned according to the experiment's source/protocol/branch decision.

Do not use "the server can access that platform" as permission to monitor a desktop job.

## C3. No live cross-machine control plane

Default architecture explicitly excludes:

- desktop -> server monitor handoff;
- server -> desktop wakeup;
- server takeover of desktop jobs;
- desktop takeover of server A/B chains;
- shared cross-machine live state DB;
- shared live SQLite;
- cross-machine owner leases;
- automatic fallback polling;
- automatic conversation messaging across machines.

Normal exchange mechanisms:
- Git commits;
- experiment decision records;
- reference/evidence indexes;
- user-mediated branch integration;
- durable artifacts.

These can be asynchronous.

## C4. Campaign authorization exists only inside its owning workflow

A bounded campaign still matters.

For SERVER campaigns:
- A may autonomously continue covered 100K/500K work;
- B monitors server-owned jobs;
- no automatic full/protected roles.

For DESKTOP campaigns:
- desktop agent operates only while the desktop is available;
- no requirement for A/B monitoring;
- remote jobs may outlive the local machine session;
- the next decision waits until desktop returns unless the user explicitly delegates that specific task elsewhere.

Do not turn the absence of continuous desktop monitoring into an error state.

## C5. Server heartbeat rules

Every 30 minutes for an active server-owned bound run:

1. read compact binding;
2. verify run identity;
3. query authoritative platform status;
4. healthy queued/running -> silent;
5. terminal/actionable fault -> durable event to server A;
6. API/status unknown -> UNKNOWN, not failure;
7. never submit from B.

B should use minimal context.

Do not reread the whole repository on every tick.

## C6. Server A terminal-event rules

Server A:

1. claims event idempotently;
2. confirms release/run/attempt;
3. collects required evidence;
4. distinguishes infrastructure failure from scientific result;
5. performs mechanical/scientific acceptance;
6. records decision and cumulative cost;
7. chooses zero or one next justified covered action;
8. submits and rebinds B, or pauses/closes;
9. emits `READY_FOR_DESKTOP` evidence when a server candidate reaches the server's scale ceiling.

No desktop notification is required.

## C7. Desktop offline rules

Before desktop shutdown:

- freeze/persist source and config;
- make remote outputs/checkpoints durable;
- record job identity;
- ensure resume logic is valid;
- preserve provenance.

Then the desktop can shut down.

While offline:
- remote desktop-owned job may queue/run/finish;
- nobody is required to inspect it;
- server ignores it;
- no next action occurs.

When desktop returns:
- query actual remote state;
- retrieve artifacts;
- accept/analyze;
- continue.

The delay is accepted wall-clock latency, not a process defect.

## C8. Research evidence is asynchronous

A server experiment can produce a durable evidence package that desktop reads hours or days later.

A desktop experiment can produce a reference index that server consumes later through Git.

No live messaging is necessary.

Evidence packages should use exact identities and pointers rather than copied stale summaries.

## C9. Outcome fields remain separate

Use:

```text
execution_status
artifact_status
comparison_status
scientific_status
transfer_status
budget_decision
full_handoff_status
```

Avoid `accepted=true` as a single overloaded state.

## C10. Strict comparison transaction

For new promotion claims require, as applicable:
- approved source/config;
- data/row/feature/target identity;
- seed/precision;
- optimizer/schedule/loss/exposure/selection;
- reference bundle;
- runtime qualification;
- finite predictions/targets;
- strict shapes;
- source index alignment;
- hashes;
- resume cursor;
- paired comparison/bootstrap;
- role-use history;
- actual native cost.

Missing reference => pending, not automatic retraining.

Row bootstrap != training stochasticity.

Material gate != measured variance unless separately established.

## C11. Reduce wasted training before increasing automation

The main research objective is not to keep GPUs busy.

For each new hypothesis:
- identify target baseline deficiency;
- cite direct evidence;
- name one alternative explanation;
- change one mechanism;
- define cheapest falsifier;
- check related closed routes;
- estimate native cost;
- state what decision the experiment changes.

Use existing evidence first.

A trajectory can terminate without training.

## C12. Trajectory-lite

Trajectory:

```text
state -> evidence -> action -> result -> decision
```

State includes:
- model/checkpoint or scratch identity;
- contract;
- roles already used;
- budget;
- prior evidence.

Generate a few written hypotheses if useful.
Normally execute zero or one new training experiment at a time.

Do not implement:
- default 4-8 parallel training rollouts;
- PPO/GRPO;
- LLM policy training;
- a universal weighted reward mixing Track A and Track B.

Store positive, negative, inconclusive and infrastructure trajectories.

## C13. Calibrate low-cost screens

Current evidence shows 100K is not a universal predictor of target scale.

Before future early stopping:
- backtest same-contract early/late traces;
- identify slow starters;
- identify selection-only missing endpoints;
- separate changes in dataset size, optimizer steps, EMA, role and architecture.

Future cumulative budget rungs may be used only prospectively after calibration.

Example only:

```text
B/16 -> B/4 -> B
```

If adopted:
- resume same run;
- same frozen LR/optimizer trajectory;
- matched candidate/reference prefixes;
- no schedule restart at each rung.

## C14. Target roles and repeated selection

Development repeatedly used for search is selection evidence.

A shadow repeatedly consulted is not an untouched final confirmation.

A full checkpoint cannot obtain a new independent holdout by resplitting rows it already trained on.

Record role membership and usage history.

## C15. Profiling before changing scientific contracts

Measure:
- load/collate;
- H2D;
- forward;
- backward/optimizer;
- validation;
- checkpoint/hash/archive;
- allocation duration.

Low VRAM does not prove low GPU utilization.

Batch/precision/optimizer/schedule/sampler changes are new contracts unless proven to preserve the relevant execution semantics under the applicable policy.

## C16. PairToken guardrails

PairToken:
- creates nonlinear ordered-pair features from node states;
- learns pair selection;
- reduces to a molecule-level token;
- returns the same vector to all nodes at that insertion point;
- does not directly read explicit bond/distance/geometry pair identity.

Do not interpret destructive frozen ablation as net training contribution.

Do not repeat already completed mechanism-activation ablations without a new decision question.

Measure target-scale benefit and same-hardware cost before opening approximate pair variants.

## C17. Cost accounting

Keep hardware-native units separate:
- T4 device-hours;
- A100 hours;
- DCU hours;
- CPU hours;
- wall time;
- queue time.

Do not silently convert them.

Record:
- preflight cost;
- failed infrastructure attempts;
- retries;
- audits;
- training;
- acceptance.

A small MAE gain with large cost can fail a budget decision even when scientifically directional.

## C18. Full-scale direction

Server ceiling is normally 500K.

A server candidate ready for desktop creates a durable `READY_FOR_DESKTOP` package and stops scale-up.

Desktop later decides full-scale when online.

No live handoff.

No server waiting is required before server continues unrelated authorized research.

No desktop wake is required.

## C19. Document architecture for lower-capability agents

Root `AGENTS.md` files should remain short.

Detailed operational rules live in:
- `docs/operations/MOLGAP_COMMON_DIRECTION_V5_FINAL.md` for shared topology and
  evidence boundaries;
- `docs/operations/SERVER_AGENT_HANDOFF_V5_FINAL.md` for the server-only A/B
  loop;
- the desktop branch's short `AGENTS.md`, `CURRENT_STATE.md`, and relevant
  experiment documents for desktop-specific work.

Implementation agents should read only the applicable V5 contract documents
once during bootstrap.

Runtime monitor B should not.

Routine server B reads only:
- compact binding;
- last observation;
- unresolved event/outbox.

Routine desktop work reads only desktop branch state and relevant experiment documents.

## C20. Implementation boundaries

A dedicated implementation Luna Max may safely:
- inspect repo state;
- make narrow docs patches;
- add synthetic control-state tests;
- add strict acceptance helpers;
- add ledger/profiling utilities;
- import reviewed shared commits.

It should not, as part of V5 implementation:
- submit new scientific training;
- consume new protected evaluation roles;
- change scientific thresholds;
- modify production registry;
- invent cross-machine monitoring;
- redesign architecture research.

If scientific meaning is ambiguous, preserve state and report rather than guess.

## C21. Correct rollout order

1. inspect current real state;
2. install the applicable V5 contract documents;
3. patch server/desktop AGENTS narrowly and differently;
4. test that no cross-machine monitoring is introduced;
5. improve server-local A/B durability/idempotency;
6. improve strict acceptance;
7. export reusable reference/evidence indexes;
8. add profiling/ledger improvements;
9. backtest low-cost screening before changing future training ladders.

Do not stop a valid existing run merely to complete this refactor.

## C22. Core regression invariants

A correct implementation must preserve:

- server A/B works only for server-owned chains;
- 30-minute server B heartbeat;
- healthy server jobs are silent;
- server B never submits;
- desktop has no default B heartbeat;
- desktop shutdown does not trigger server takeover;
- desktop/server conversations need not communicate;
- desktop offline silence is accepted;
- server ceiling remains 500K unless explicitly changed later;
- full/official/submission remains desktop-oriented;
- no duplicate training from stale local state;
- negative evidence remains discoverable;
- strict acceptance separates mechanical/scientific/budget decisions;
- Track A and Track B remain separate;
- existing scientific contracts are not rewritten retroactively.


## C23. Permanent V5 screening funnel

The default future research funnel is:

```text
1. Define observed deficiency / research question
2. Retrieve existing evidence and related closed-family history
3. Write one hypothesis card
4. Use the cheapest decision-relevant diagnostic
5. If justified, release one bounded 100K experiment
6. Strict acceptance + scientific interpretation + cost decision
7. Only if qualified and allowed, release 500K confirmation
8. Strict transfer decision
9. If qualified, emit READY_FOR_DESKTOP
10. Desktop independently decides whether full-scale work is worth doing
11. Full/evaluation/submission follow desktop authority
12. Production promotion remains a separate explicit gate
```

The funnel may stop at any stage.

`NO_TRAIN`, `NEGATIVE_UNDER_CONTRACT`, `INCONCLUSIVE`, `STOP_FOR_COST`, and `DUPLICATE_EVIDENCE` are valid outcomes.

Do not require every candidate to consume every rung.

## C24. What may evolve without V6

V5 does NOT freeze scientific progress.

The following may change without a new architecture contract when properly versioned and authorized:
- model architecture candidates;
- data-independent engineering optimizations;
- platform adapters;
- acceptance implementation;
- profiling instrumentation;
- experiment ledger schema;
- statistically justified gate values for future experiments;
- a new future scientific training contract;
- calibrated cumulative-budget rungs;
- multi-seed policy for a final shortlist;
- full-training schedule;
- hardware choice.

These changes must still respect V5 machine responsibilities, evidence boundaries, role discipline, and promotion topology.

## C25. When a new contract version is actually warranted

A future V6 should be considered only if evidence shows that a V5 architectural invariant itself is causing substantial recurring failure or waste that cannot be repaired locally.

Examples:
- the user deliberately wants desktop/server live orchestration;
- the server research ceiling changes permanently beyond 500K;
- machine roles are reorganized;
- a new persistent orchestration system replaces the A/B model;
- branch governance is intentionally redesigned.

A new model family, better early stopping, a different optimizer, or a new GPU platform is NOT by itself a reason for V6.

Until explicit user authorization, V5 remains authoritative.

## C26. Additive comparison-readiness amendment

This section is a backward-compatible V5 amendment. It does not change machine
ownership, the server ceiling, promotion topology, or protected-role authority.

Every future scientific comparison declares one machine-readable class before
training:

- `STRICT_CAUSAL`: all explicit scientific identity fields and complete
  candidate/reference evidence match except the predeclared intervention. Only
  this class permits causal attribution to that intervention.
- `PAIRED_ENDPOINT`: predictions and targets are aligned on the same evaluation
  rows, but one or more training-recipe fields differ. Endpoint prediction,
  paired bootstrap, and ensemble claims are allowed; mechanism causality is not.
- `MATCHED_PREFIX`: terminal evidence is unavailable, but a common optimizer
  step or sample-presentation prefix is strictly aligned. It supports trajectory
  comparison, futility, and `STOP_FOR_COST`, not terminal superiority.
- `CONTEXT_ONLY`: scalar or historical evidence, incomplete identity, or
  unmatched conditions. It supplies context only.
- `NO_COMPARISON`: no comparator is available.

`comparison_readiness.json` is fail-closed. Missing fields are unknown, never
matched. `strict_ready=true` requires `STRICT_CAUSAL`, no blockers, complete
artifacts, aligned predictions, accepted runtime certificates, explicit role
history, and a complete terminal trace. Architecture or mechanism differences
must be listed in `declared_intervention_fields`; all undeclared differences are
confounders.

The explicit identity includes benchmark, dataset, role, row membership/order,
features, target, seed, precision, TF32 and determinism, physical batch,
accumulation, tail policy, optimizer and fused mode, schedule, loss, target
transform asset, presentations, optimizer steps, endpoint selection,
evaluation/selection roles, architecture/config, runtime-certificate scope,
live/EMA weight semantics, EMA update semantics, and evaluation weight source.

A reusable `reference_bundle.json` includes the contract, architecture/config,
source, checkpoint, runtime certificate, aligned prediction/row/target
manifests, trace, role history, target-transform asset, cost, acceptance, and
decision. A scalar MAE is not a reusable bundle. Target transforms use one
portable asset containing mean, standard deviation, variance convention,
source-row hash, target hash, and asset hash; candidate and reference do not
independently recompute it.

Before future server architecture or mechanism training, the prelaunch gate
must locate the comparator and bundle, compare every identity field, declare
the intended difference, and emit `comparison_readiness_prelaunch.json`.
Without `STRICT_CAUSAL`, the work is explicitly relabeled `diagnostic`,
`transfer_study`, `delivery_experiment`, or `contextual_experiment`, or becomes
`NO_TRAIN`. Missing reference evidence remains pending and never authorizes an
automatic baseline rerun.

Future prospective trajectories carry `comparison_class`,
`comparison_readiness_ref`, `comparison_blockers`, and `reference_bundle_id`.
Historical trajectories remain `retrospective_partial`; absent optimizer, EMA,
role, or artifact facts are not reconstructed by inference.

Every future scientific run emits role events for training membership,
prediction input, label reads, metric computation, selection, and external
submission as applicable. Every future training trace explicitly records which
of optimizer step, sample presentations, epoch/pass, learning rate, live train
metric, live dev metric, EMA dev metric, and checkpoint identity are present;
unavailable fields remain unavailable.

Row-bootstrap uncertainty and training stochasticity are distinct evidence.
Only same-contract repeated runs can measure training stochasticity. A paired
row bootstrap, or a policy threshold such as `0.003 eV`, cannot be relabeled as
a measured training-stochasticity floor.
