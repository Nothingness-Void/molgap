# MolGap server agent handoff v5 — FINAL CONTRACT

Date: 2026-09-17 (Asia/Tokyo)
Document ID: MOLGAP-SERVER-V5-FINAL
Owner branch: `molgap-server`

This document supersedes all prior server handoff contracts, including V3 and V4. V5 is the stable final operating contract unless the user explicitly authorizes a future contract version.

## S0. Read this first

The project has two independent machine workflows.

SERVER WORKFLOW:
- Server stays powered on.
- Server Conversation A is the high-capability research coordinator.
- Server Conversation B is the low-cost Luna Max monitor.
- A and B may communicate with each other locally through the already working server-side handoff mechanism.
- A/B are used only for server-owned research chains.
- Server normally owns bounded 100K and 500K screening/research work.
- A analyzes results, diagnoses execution faults, freezes the next permitted experiment, submits it to an authorized remote training platform, then binds B.
- B wakes every 30 minutes, checks only the bound remote job, stays silent when healthy, and wakes A on completion or actionable failure.

DESKTOP WORKFLOW:
- Desktop is independent.
- Desktop does not participate in the server A/B loop.
- Desktop conversations and server conversations are not assumed to communicate.
- Server must not monitor, adopt, continue, or take custody of a desktop-owned job merely because desktop is powered off.
- Desktop shutdown latency is accepted by design.
- Desktop-owned remote jobs may continue on their platform while desktop is off, but no server-side agent is responsible for observing or advancing them.
- Desktop usually owns full-scale work and may also independently run selected 500K work while it is online.

Do not build a cross-machine event bus, shared owner lease, cross-machine heartbeat, automatic desktop fallback, or server takeover mechanism.


## S0A. Final-contract rule

Treat V5 as the default architecture for all future server-side screening work.

Do not redesign the branch topology, server A/B relationship, server 100K/500K ceiling, desktop independence, acceptance state model, or positive-before-full workflow merely because a future agent prefers another orchestration pattern.

Future work may:
- fix bugs;
- improve tests;
- improve profiling and bookkeeping;
- add a new candidate family;
- calibrate an early-stop rule;
- improve acceptance helpers;
- change a scientific training contract when separately authorized.

Future work may NOT silently replace this operating architecture.

Any proposal that changes a V5 architectural invariant must be written as a separate change proposal and must remain inactive until the user explicitly authorizes a new contract version. Without such authorization, implement the requested research inside V5.

V5 architectural invariants:
1. Server and desktop are operationally independent.
2. Only server has the default local A/B loop.
3. Server B uses a 30-minute heartbeat, stays silent when healthy, and never makes scientific decisions or submissions.
4. Server A owns bounded server-side 100K/500K research within recorded authority.
5. Desktop shutdown does not trigger server monitoring or takeover.
6. Full-scale / official evaluation / final submission remain desktop-oriented unless the user explicitly changes that boundary.
7. Positive small-scale evidence is necessary but never self-authorizes full scale.
8. Mechanical acceptance, scientific interpretation, transfer evidence, budget decision, and full handoff remain separate states.
9. Negative and inconclusive evidence remain discoverable so failed families are not repeatedly repackaged.
10. Future efficiency improvements should reduce wasted compute without weakening scientific comparability.

## S1. Preserve the user's actual server A/B loop

The server loop exists to eliminate avoidable idle wall-clock time between a server-owned remote job finishing and the user returning.

Expected sequence:

```text
Server A
  -> analyze current accepted evidence
  -> choose zero or one justified next server-owned action
  -> freeze source/config/data/reference/budget/roles
  -> submit to Kaggle / SCNet / IMS / other authorized platform
  -> bind Server B

Server B
  -> every 30 minutes check the exact bound job
  -> healthy QUEUED/RUNNING: silent
  -> actionable execution fault: persist event and wake Server A
  -> terminal completion: persist event and wake Server A

Server A
  -> claim event idempotently
  -> collect/accept evidence
  -> distinguish infrastructure failure from scientific result
  -> repair/retry if authorized and scientifically unchanged
  -> otherwise decide scientific outcome
  -> choose zero or one next justified action
  -> submit and rebind B, or pause/close
```

B is not a planner. B must never:
- select a model;
- interpret scientific value;
- edit scientific code;
- change batch/optimizer/schedule/data/seed;
- retry a training job;
- cancel a job unless a separately frozen operational rule explicitly delegates that exact action;
- launch a successor;
- read protected evaluation roles;
- create a new A conversation.

## S2. No cross-machine takeover

This section is mandatory because previous handoff specifications misunderstood the user's architecture.

The following behavior is FORBIDDEN by default:

```text
desktop shuts down
    -> server starts monitoring desktop job
    -> server A takes scientific ownership
    -> server submits desktop successor
```

Do not implement that workflow.

Also do not:
- synchronize a live control database between desktop and server;
- create `owner_epoch` fencing between machines for ordinary research;
- create a server fallback monitor for desktop jobs;
- make server inspect desktop scheduler jobs just because they may be running;
- wake server A because a desktop-owned job completed;
- make desktop wait for server authorization for its independent work;
- assume a platform location determines ownership.

Server and desktop exchange reviewed scientific/code evidence through Git and user-directed integration when convenient, not through a live operational control plane.

A future explicit user request may assign a specific desktop task to server, but that is an exceptional new delegation and is not the default architecture.

## S3. Branch responsibility

Keep the four long-lived branches:

- `master`: stable delivery only.
- `molgap-server`: server-owned architecture exploration, 100K/500K screens, reusable runners, acceptance logic, server-side experiment decisions and server A/B operational support.
- `molgap-desktop`: independent desktop integration, full-scale work, official evaluation, final submission, and any desktop-owned 500K work.
- `archive`: exact inactive/rejected/superseded source histories and provenance.

Do not use branch placement as scientific proof:
- merged code != positive result;
- positive result != full-scale authorization;
- archive != scientifically negative in every case;
- two positive modules merged together != validated combination.

Negative/inconclusive experiment conclusions should remain discoverable from the active evidence index even when implementation history moves to `archive`.

## S4. Files to inspect before editing

Read, in order:

1. applicable `AGENTS.md`;
2. branch `CURRENT_STATE.md`;
3. relevant `ROADMAP.md` section;
4. `TRACKS.md` if needed;
5. `ARCHITECTURE.md`;
6. the owning experiment protocol/decision/status;
7. `platforms/REMOTE_HANDOFF.md` before authorized remote access;
8. only the code touched by the task.

Do not reload the literature ledger or all archive history on routine wakes.

Before any edit:
- inspect branch, HEAD, upstream divergence, tracked/untracked changes;
- preserve unrelated user edits;
- do not automatically stash/reset/clean/force-push;
- do not rewrite `AGENTS.md` wholesale;
- do not copy desktop `AGENTS.md` into server.

## S5. Required `AGENTS.md` correction

The server root `AGENTS.md` should preserve a short server-only A/B rule.

It should state, in substance:

```text
## Remote monitor handoff

The always-on server runs a local two-conversation research loop.

Conversation A owns bounded server-side 100K/500K scientific interpretation,
targeted repairs, release validation, and remote submissions within existing
campaign authority.

Conversation B uses the user's Luna Max configuration and a 30-minute
heartbeat. B monitors only server-owned bound jobs. Healthy queued/running work
is silent. Confirmed completion or an actionable execution fault creates one
durable event that wakes the existing server A.

B never performs scientific selection, changes code/contracts, retries or
submits jobs, or opens protected evaluation roles. A consumes the event,
accepts/interprets evidence, then chooses zero or one justified next in-scope
action and rebinds B, or pauses/closes the chain.

Desktop is operationally independent. The server does not automatically
monitor, adopt, or continue desktop-owned jobs when desktop is offline.
Cross-machine live handoff is not part of the default architecture.

Detailed policy:
docs/operations/MOLGAP_COMMON_DIRECTION_V5_FINAL.md
```

Preserve all existing scientific/data/platform safety rules.

## S6. Server runtime control state

The server may maintain a small durable local operational record for SERVER-OWNED chains only.

Minimum conceptual fields:

```text
campaign_id
chain_id
run_id
attempt_id
A_thread_id
B_thread_id
monitor_generation
remote_platform
remote_job_identity
release_identity
reference_identity
event_id
event_type
event_status
budget_spent_native
budget_reserved_native
last_observed_at
decision_ref
```

Do not put credentials in public Git.

Use existing handoff/heartbeat mechanisms first. Add only missing transaction fields.

A small local SQLite store or atomic files are acceptable when needed, but:
- it is server-local;
- it is not synchronized live with desktop;
- Git is not used as a 30-minute polling message queue.

## S7. Reliable A/B event semantics

Use at-least-once delivery plus idempotent consumption.

Recommended state:

```text
OBSERVED
-> EVENT_DURABLE
-> DELIVERED_TO_A
-> A_CLAIMED
-> DECISION_COMMITTED
-> NEXT_RUN_BOUND | PAUSED | CLOSED | READY_FOR_DESKTOP
```

Important:
- send acknowledgement != A processed the event;
- duplicate B ticks must not create duplicate successors;
- if A is busy, keep the same event pending;
- if B restarts, resume from durable state;
- if submission times out after remote acceptance may have occurred, mark `SUBMIT_UNKNOWN` and reconcile before retrying;
- do not claim exactly-once remote operations.

## S8. Server B compact runtime prompt

After implementation, Server B should operate from a compact binding, not the entire repository.

Example:

```text
ROLE: MolGap server monitor B. You are not the research planner.

CHECK EVERY: 30 minutes.

BOUND RUN:
campaign_id=<...>
run_id=<...>
attempt_id=<...>
platform=<...>
remote_job=<...>
release=<...>
A_thread=<...>
monitor_generation=<...>

HEALTHY QUEUED/RUNNING:
- update compact observation
- remain silent

CONFIRMED COMPLETE:
- persist one terminal event
- deliver the event to the existing server A
- do not submit anything

ACTIONABLE EXECUTION FAULT:
- persist one failure event
- deliver it to A
- do not repair, retry, cancel, or change science

STATUS/API UNKNOWN:
- record UNKNOWN
- do not treat as training failure
- do not resubmit

FORBIDDEN:
scientific interpretation
code edits
contract changes
candidate selection
new seed
successor submission
protected evaluation access
desktop job monitoring
creating another A
```

## S9. Acceptance and promotion repair

Server is responsible for making 100K/500K acceptance fail closed.

Separate fields:

```text
execution_status
artifact_status
comparison_status
scientific_status
transfer_status
budget_decision
full_handoff_status
```

Do not overload `accepted=true`.

For a strict comparison require, as applicable:
- approved source commit/archive;
- architecture/config identity;
- data/feature/row-order/target transform;
- seed/precision;
- optimizer/scheduler/loss/selection/exposure;
- exact reference bundle;
- runtime certificate within its actual scope;
- finite predictions and targets;
- strict tensor shapes;
- unique, aligned source indices;
- artifact hashes;
- step/resume cursor;
- paired comparison and required bootstrap;
- role-use history;
- actual native cost and remaining server campaign authority.

Missing reference evidence => `pending`, not automatic baseline retraining.

A row bootstrap does not measure training randomness.

A policy threshold such as 0.003 eV is not automatically a measured stochasticity estimate.

## S10. Scientific workflow to reduce wasted training

Do not generate a literature module list and train everything.

For a new candidate, require one hypothesis card:

```text
hypothesis_id
family_id
current baseline deficiency
supporting evidence
alternative explanation
one changed mechanism
earliest cheap falsifier
closed related routes
expected native cost
what decision this result changes
```

Preferred order:

1. reuse existing accepted evidence/logs;
2. static architecture/information-flow audit;
3. one cheap decision-relevant diagnostic if needed;
4. only then one bounded new training action.

A diagnostic whose possible outcomes all lead to the same training run should usually be skipped.

Default new training concurrency is one unless an explicitly frozen paired experiment says otherwise.

Trajectory means:
`state -> evidence -> action -> result -> decision`.

A trajectory may validly end in `NO_TRAIN`.

Do not implement PPO/GRPO or a 4-8-training-run tournament.

## S11. 100K/500K future efficiency

Keep the user's current principle:
- server uses 100K/500K to qualify;
- only sufficiently positive evidence goes toward desktop full-scale consideration.

But future low-cost ladders should be calibrated using history.

Do not assume:
- 100K endpoint predicts 500K;
- 500K dataset means cheap compute;
- row bootstrap predicts future endpoint.

Backtest same-contract early/late curves first.

A future cumulative schedule such as:

```text
B/16 -> B/4 -> B
```

is only a candidate design after backtesting. If adopted:
- same run resumes;
- same predeclared LR/optimizer trajectory;
- candidate/reference compared at matched prefixes;
- no reset of cosine schedule per rung;
- no silent data-role change.

Existing running contracts remain unchanged.

## S12. Profiling before changing BS128 or precision

Measure first:
- loader/collate;
- H2D;
- forward/loss;
- backward/optimizer;
- validation;
- checkpoint/hash/archive;
- real allocated duration;
- graph-size tail behavior.

Investigate:
- GPU-to-Python scalar synchronization;
- finite-loss control-flow synchronization;
- DataLoader recreation;
- CPU prediction transfers;
- repeated runtime certificate construction;
- incomplete timer scopes.

Low VRAM use is not proof of low compute utilization.

Do not change batch, AMP/TF32, optimizer, schedule, sampler, or relation approximation and call it an infrastructure fix.

## S13. PairToken-specific guardrails

Do not overinterpret its mechanism.

The reviewed PairToken:
- builds normalized nonlinear ordered-pair features from existing node states;
- performs learned all-pair selection;
- reduces to one molecule-level relation token;
- returns the same residual vector to every node at that insertion point;
- does not directly read pair bond identity, distance, or geometry.

Do not repeat already completed disable/uniform/diagonal/no-normalization audits merely to show the branch is active.

If PairToken remains worth work:
1. first establish target-scale benefit;
2. measure same-hardware cost;
3. measure pair materialization/padding;
4. only then consider exact implementation optimizations.

Top-k/random/low-rank variants are new science, not free infrastructure fixes.

## S14. Server-to-desktop interface

There is NO live server-to-desktop A/B handoff.

The only normal interface is a durable scientific package that desktop may inspect later.

For a server candidate ready for full consideration, create a `READY_FOR_DESKTOP` evidence package containing:
- frozen source/config;
- 100K/500K decisions;
- reference/candidate contract identities;
- prediction/artifact hashes/locators where allowed;
- paired analysis;
- training membership and role history;
- recovery state;
- native cost;
- limitations;
- exact question proposed for full-scale work.

Then stop that candidate's server scale-up at the server ceiling.

Do not wake desktop.
Do not monitor desktop.
Do not submit desktop full work.
Do not wait on desktop before continuing unrelated authorized server research.

Desktop will inspect this package when it is next online.

## S15. Regression tests before modifying the live loop

Use fake/synthetic tests first.

Required cases:

- healthy server-owned queued/running job -> silent B;
- server-owned COMPLETE -> one durable event;
- duplicate terminal delivery -> one A decision;
- A busy -> event retained;
- A crash after claim -> recoverable;
- submit timeout -> `SUBMIT_UNKNOWN`, no blind retry;
- stale B generation -> cannot mutate new chain;
- artifacts late -> pending evidence, no duplicate successor;
- NaN/Inf/wrong shape/misaligned source_idx -> reject;
- missing reference/bootstrap -> promotion pending;
- old v3 incompatible reference -> not strict v4 comparator;
- 500K positive -> `READY_FOR_DESKTOP`, no full submission;
- desktop job visible somewhere -> server B ignores it unless the user explicitly reassigns that exact task;
- AGENTS edit accidentally adds desktop takeover -> regression test fails.

## S16. Implementation order for a lower-capability implementation agent

Phase S0:
- inspect and report;
- no code changes.

Phase S1:
- install the applicable V5 contract documents;
- narrow `AGENTS.md` correction;
- no training.

Phase S2:
- inspect/reuse current A/B implementation;
- add only missing durability/idempotency;
- synthetic tests.

Phase S3:
- repair strict acceptance/promotion wrapper;
- synthetic and local tests.

Phase S4:
- add profiling instrumentation and experiment ledger improvements where useful.

Phase S5:
- backtest low-fidelity screening from existing evidence.

Do not launch a new GPU/DCU/A100/T4 experiment merely to prove this implementation.

## S17. Final report format

Report:

1. actual branch/worktree state;
2. existing A/B capabilities reused;
3. files/commits changed;
4. tests actually executed;
5. current server-owned active chains and monitor bindings;
6. acceptance/promotion gaps still unresolved;
7. next one permitted server action, or justified pause;
8. explicit confirmation that no desktop monitoring/takeover mechanism was added.


## S18. Permanent screening lifecycle under V5

For future server-side architecture research, use this lifecycle unless an experiment-specific contract is narrower:

```text
Observed deficiency / research question
-> retrieve existing evidence and closed-family history
-> write one hypothesis card
-> cheapest decision-relevant falsifier
-> if still justified: one frozen 100K action
-> strict acceptance + scientific decision
-> if qualified AND explicitly allowed: 500K confirmation
-> strict transfer/cost decision
-> if qualified: emit READY_FOR_DESKTOP
-> stop server scale-up for that candidate
```

A negative, inconclusive, duplicate, too-expensive, or poorly evidenced candidate may stop before training or before 500K.

Do not restore the old pattern:
`paper idea -> full 100K -> full 500K -> full`
without intermediate evidence and cost decisions.

## S19. V5 change control

When an implementation agent discovers tension with V5:

1. first determine whether it is an implementation bug, missing field, stale documentation, or genuinely new scientific requirement;
2. prefer a backward-compatible implementation fix;
3. do not rewrite the V5 architecture to solve a local inconvenience;
4. if an invariant truly needs changing, create a concise `V5_CHANGE_PROPOSAL` describing:
   - current rule;
   - observed evidence;
   - proposed rule;
   - migration cost;
   - risks;
   - affected experiments;
5. keep the proposal inactive until explicit user authorization.

No agent may self-declare V6.

## S20. Comparability implementation correction

This correction is additive and does not change the server ceiling, Server A/B
roles, desktop independence, protected-role rules, `READY_FOR_DESKTOP`
topology, or any historical scientific decision.

Server release uses two stages:

```text
comparison_readiness_prelaunch.json
  -> PRELAUNCH_STRICT_PLANNED
  -> training may be released

completed run + actual artifacts
  -> comparison_readiness.json
  -> STRICT_CAUSAL only if observed evidence is complete
```

Prelaunch validates the reusable reference bundle; frozen candidate source,
config, source commit/archive and planned identity; one purpose-scoped
intervention group reflected in that planned identity; explicit role
applicability; trace declarations; and runtime qualification plan. It
must not demand future candidate predictions, checkpoint, terminal trace,
completed role events, bootstrap or row-alignment artifact.

The release call receives the actual reusable reference bundle and recomputes
the prelaunch record against it. An asserted bundle ID or asserted
`matched_fields` map cannot release compute, and later repository validation is
not a substitute for this release-time binding.

Role plans declare every event kind as `applicable` or `not_applicable`.
Post-run evidence requires events only for applicable kinds and rejects events
invented for not-applicable kinds. An ordinary architecture screen may and
normally will mark `external_submission` as `not_applicable`.
Every causal training comparison must mark training membership, prediction
input, label read, metric computation and selection as `applicable`. Its
minimum trace includes optimizer step, sample presentations, epoch/pass,
learning rate, live train/dev metrics and checkpoint identity; EMA development
metrics are additionally required when EMA is enabled.

`declared_intervention_fields` is not a general exception list. Architecture
and mechanism comparisons may vary only their declared architecture/config
mechanism; mechanism comparisons name `mechanism_id`. Optimizer, EMA and
schedule comparisons have separate allowed field groups. One strict experiment
has one `intervention_group_id`; unrelated scientific mismatches remain
confounders.

Historical validity and prospective reuse are separate. Preserve claims valid
under an original contract, while independently classifying future reference
reuse as `READY`, `INCOMPLETE_RECOVERABLE`, `NOT_COMPATIBLE`, or `UNAVAILABLE`.
Do not make historical runs claim a portable target-transform asset that did
not exist when they ran.

Post-run strict readiness binds both arms to immutable repository-local
artifact or manifest pointers with SHA-256. Repository validation resolves and
hashes every strict binding. Role and trace completeness are recomputed from
their detailed plans and observations rather than trusted from summary status
strings.

Future reusable references consume a discovered and validated
`experiments/**/target_transform.json`. Its canonical digest, target identity
and bundle identity must agree before prospective strict reuse.
