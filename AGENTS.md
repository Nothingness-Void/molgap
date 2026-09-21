# AGENTS — Reading Protocol

MolGap predicts HOMO/LUMO/Gap (eV) for organic electronic molecules using
molecular ML. This file defines **how an agent navigates and operates in the
repository**. It is not the project history and it is not a substitute for
experiment evidence.

One fact should have one authoritative owner. Follow pointers instead of
copying facts into multiple documents.

## Read in this order

1. **This file** — repository navigation, operating protocol, and hard
   constraints.
2. **`BRANCHES.md`** — branch ownership and checkout routing when the task may
   edit code, submit work, integrate results, or inspect another branch.
3. **`CURRENT_STATE.md`** — authoritative project-level summary for current
   recommendations, active blockers, routing, and next actions.
4. **`ROADMAP.md`** — current task priorities and backlog; read only the
   relevant section.
5. **`TRACKS.md`** — read only when a task is explicitly associated with
   Track A, B, or C.
6. **`ARCHITECTURE.md`** — repository tree and code ownership. Naming rules
   live in **`NAMING.md`**.
7. The `README.md` for the tree being modified:
   - `production/`
   - `experiments/`
   - `platforms/`
   - `research_memory/`
8. The owning experiment's protocol, decision, acceptance, evidence, and
   artifact records.
9. Only the specific code files required by the task.

Do not read every historical document to discover the current project state.
Start from the routing documents above and follow their pointers.

Historical phase 1-7 records, retired production histories, superseded
implementations, and other inactive material are preserved on the `archive`
branch or through explicit archive pointers. Never infer current state or an
open research question from archive history.

## Authority boundaries

`CURRENT_STATE.md` owns the current project-level view:

- recommended model or route;
- active blocker;
- current routing;
- current queue state;
- current next actions.

It does **not** override:

- a frozen experiment contract;
- a dated experiment decision;
- an acceptance record;
- canonical V5 evidence;
- artifact provenance;
- canonical RML records;
- authoritative remote scheduler state.

If one of those authoritative records conflicts with `CURRENT_STATE.md`, do not
rewrite the underlying evidence to match the summary. Reconcile the discrepancy
and update `CURRENT_STATE.md` if the summary is stale.

Use this precedence when resolving factual state:

```text
authoritative remote scheduler / actual durable artifact state
    ->
experiment contract / decision / acceptance
    ->
canonical V5 evidence and canonical RML records
    ->
CURRENT_STATE.md project-level routing summary
    ->
RML derived indexes / generated summaries
    ->
agent interpretation
```

Generated summaries are navigation aids, not replacements for primary evidence.

## Research Memory Layer (RML)

RML is the deterministic research-evidence index for MolGap.

Read:

- `research_memory/README.md`
- targeted files under `research_memory/derived/`

when answering questions about prior experiments, reusable references,
historical cost, role usage, research-family history, duplicate work, or
screening evidence.

Canonical RML inputs remain beside their owning experiments, for example:

```text
trajectory.json
costs/*.json
roles/*.json
trace_manifest.json
handoff/READY_FOR_DESKTOP.json
```

Everything under:

```text
research_memory/derived/
```

is compiler-generated state.

Never edit generated RML derived files manually.

RML derived outputs are indexes and summaries, not primary scientific evidence.
Before making a scientific decision from an RML result, follow its pointers back
to the owning experiment's canonical contract, decision, acceptance, V5
evidence, role, cost, trace, and artifact records.

Missing historical information must remain missing. Do not infer unknown
hardware, cost, role use, rationale, training traces, or scientific conclusions
merely to make RML look complete.

## New research question protocol

Before proposing, modifying, or submitting an experiment for a new scientific
question, query RML first.

Use RML to answer:

1. Has this question or mechanism already been tested?
2. Has the same or a related model family already failed under a comparable
   contract?
3. Which strict references are available?
4. Which development, validation, shadow, test, or protected roles have already
   been used?
5. What native cost evidence exists?
6. Is there already a closed, negative, inconclusive, or duplicate trajectory
   that answers the question?
7. Which evidence gaps remain genuinely unresolved?

Then read the authoritative records linked by RML.

A new experiment is justified only after that evidence review.

For a genuinely new research route, create a canonical:

```text
record_mode = prospective
```

`trajectory.json` **before** running a new diagnostic or training experiment.

The prospective record should capture the decision-relevant state required by
the active contract, including as applicable:

- research question;
- observed baseline deficiency;
- supporting evidence;
- alternative explanation;
- changed mechanism;
- cheapest decision-relevant falsifier;
- related closed families;
- reference identity;
- role-use state;
- budget/cost expectation;
- the decision the experiment is intended to change.

A trajectory may validly terminate without training.

Valid outcomes include, when supported by the active contract:

```text
NO_TRAIN
NEGATIVE_UNDER_CONTRACT
INCONCLUSIVE
STOP_FOR_COST
DUPLICATE_EVIDENCE
```

Do not launch training merely because a trajectory exists.

Advance through diagnostic, 100K, 500K, full-scale, or other gates only when
required and authorized by the active scientific contract.

After each accepted canonical research milestone, update the relevant canonical
records and rebuild RML.

Examples of milestones:

- diagnostic decision;
- terminal 100K acceptance;
- terminal 500K acceptance;
- final scientific decision;
- new native-cost record;
- new exact role-use record;
- accepted reusable trace;
- READY_FOR_DESKTOP generation.

Do not rebuild RML merely for transient runtime progress such as every epoch or
heartbeat.

Before handoff or commit, verify that RML derived state is current.

Historical evidence may be imported as:

```text
record_mode = retrospective_partial
```

only when directly supported by pre-existing authoritative records.

Never invent retrospective hypotheses, rationale, costs, role use, traces, or
evidence to justify work that was already launched. Unknown fields remain
unknown.

## Hard constraints

- **Python**: on this Windows checkout, use
  `.venv\Scripts\python.exe`. System Python does not provide the required
  torch/PyG environment.

- **Targets**: model targets are `homo`, `lumo`, and `gap` in eV under the
  applicable PubChemQC / B3LYP Kohn-Sham contract. Do not silently substitute
  experimental values or another target definition.

- **Geometry consistency**: for any route that consumes generated 3D
  coordinates, training and inference must use compatible geometry-generation
  contracts. Current generated-3D production inference uses ETKDG. Never pair
  PM6-trained geometry with ETKDG inference unless a new scientific contract
  explicitly validates that mismatch. Pure-2D routes must not be forced to
  construct 3D coordinates.

- **Reuse, don't fork**: reusable logic belongs in `src/molgap/`. Stage- or
  experiment-local `scripts/` should be thin CLI wrappers. Public inference is
  implemented in `src/molgap/inference.py` and lazily exported from
  `src/molgap/__init__.py`. Do not redefine shared model classes inside
  experiment scripts. See `ARCHITECTURE.md`.

- **Do not repeat completed experiments** without a new decision-relevant
  question. Retrieve the experiment's existing decision and RML trajectory
  first.

- **Test locally before delivery**. Do not claim tests were executed unless
  they actually ran.

- **Remote durability**: every remote training or inference job must preserve
  sufficient immutable source/config identity, checkpoints or resumable state,
  durable artifacts, and provenance for later reconciliation. Never rely on a
  transient worker filesystem or one uninterrupted process as the only copy of
  the result.

- **Remote resource discipline**: follow the owning platform and experiment
  resource contract. Under the current default workflow, parsing,
  ETKDG/graph construction, and graph acceptance should use appropriate CPU
  resources before GPU/DCU training is released. GPU/DCU allocation is for
  work that actually requires accelerator compute. Do not change scientific
  batch/precision/optimizer semantics merely to improve utilization.

- **Native cost accounting**: keep hardware-native units separate. Do not
  silently convert or combine A100, T4, DCU, CPU, wall-time, or queue-time
  measurements. Unknown is not zero; estimated is not measured; not-applicable
  is not missing.

- **IMS access boundary**: before any command against the molecular-research
  server / IMS environment, read and obey `platforms/REMOTE_HANDOFF.md`.
  Its path and safety restrictions apply even to read-only discovery and
  metadata probes.

- **Protected roles**: never consume a protected evaluation role merely to
  complete implementation, RML indexing, debugging, or infrastructure testing.

- **No evidence invention**: missing reference, cost, role identity, artifact,
  or comparison evidence remains missing or pending. Do not silently retrain a
  reference or synthesize evidence to close a bookkeeping gap.

## Branch ownership

Long-lived branch roles are defined in `BRANCHES.md`.

At a high level:

- `molgap-desktop` owns desktop integration, full-scale training, official
  evaluation, final submission, and explicitly desktop-owned selected 500K
  work.
- `molgap-server` owns its independent bounded server-side 100K/500K
  screening/research loop and server A/B operational support.
- `master` is stable delivery only.
- `archive` preserves inactive, rejected, superseded, or otherwise historical
  source/evidence histories.

Compute host alone does not determine ownership.

A Kaggle, SCNet, IMS, Colab, or other remote job is desktop-owned or
server-owned according to its recorded experiment/source/protocol/branch
decision.

Work on the owning long-lived branch when safe.

Use a temporary `codex/` branch when concurrency, isolation, or review requires
it. Do not create a branch merely for an individual seed, retry, or
infrastructure attempt.

Integrate reviewed reusable work into the owning long-lived branch with explicit
provenance. Do not wholesale merge an unrelated branch merely to obtain one
helper.

Keep temporary experiment branches only while they serve an active question.

Delete a completed temporary experiment branch only after its commits are
durably reachable from:

- its owning long-lived branch (`molgap-desktop` or `molgap-server`); or
- `archive`.

Promotion from an owner branch to `master` is a separate explicit reviewed
action.

## Desktop offline behavior

Desktop work is operationally independent from the server A/B research loop.

Desktop has:

- no default heartbeat monitor;
- no default Conversation B;
- no automatic server fallback;
- no automatic server takeover.

If the desktop is powered off, no server agent automatically monitors, adopts,
repairs, or advances desktop-owned jobs.

Silent time after a desktop-owned remote job finishes is accepted by design.

Therefore desktop-owned remote work must be durable and recoverable:

- freeze source/configuration before submission;
- persist input/cache identity;
- preserve checkpoints or resume state;
- preserve durable remote artifacts;
- preserve job identity and provenance;
- reconcile authoritative scheduler/artifact state when the desktop returns.

Desktop normally owns:

- full-scale training;
- official evaluation;
- final submission;
- explicitly desktop-owned selected 500K work.

The server independently owns its bounded 100K/500K research loop.

Cross-machine live handoff is not part of the default architecture.

Do not add:

- desktop-to-server monitor handoff;
- server-to-desktop wakeup;
- automatic server takeover of desktop jobs;
- automatic desktop takeover of server chains;
- shared cross-machine live SQLite/control DB;
- cross-machine owner leases;
- automatic fallback polling;
- automatic conversation bridges.

Normal exchange is asynchronous through:

- Git commits;
- canonical experiment records;
- RML/V5 evidence indexes;
- durable artifacts;
- user-directed integration.

Read:

- `docs/operations/MOLGAP_COMMON_DIRECTION_V5_FINAL.md`
- `docs/operations/DESKTOP_AGENT_HANDOFF_V5_FINAL.md`

for the detailed V5 operating contract.

## Remote reconciliation protocol

Before acting on a remote job after local downtime or stale context:

1. identify the owning experiment and branch;
2. identify the exact run/job/attempt;
3. query authoritative remote state under the applicable platform boundary;
4. verify source/config/release identity;
5. retrieve or verify durable artifacts;
6. distinguish infrastructure failure from scientific result;
7. perform the applicable acceptance transaction;
8. update canonical evidence/decision/cost/role records;
9. rebuild RML after the accepted milestone;
10. only then decide whether another action is justified.

Never submit a successor merely because local status appears stale.

Unknown remote state is `UNKNOWN`, not failure.

## Comparison and promotion discipline

A new scientific promotion claim requires the evidence required by its active
contract, which may include:

- approved source/config identity;
- data identity;
- row/split identity;
- feature identity;
- target/transform identity;
- seed;
- precision;
- optimizer;
- scheduler;
- loss;
- exposure;
- selection rule;
- strict reference bundle;
- runtime qualification;
- finite predictions/targets;
- shape checks;
- source-index alignment;
- artifact hashes;
- resume/step cursor;
- paired comparison;
- bootstrap or other required uncertainty analysis;
- role-use history;
- actual native cost.

Missing strict reference evidence means pending, not automatic baseline
retraining.

Row bootstrap does not measure training stochasticity.

A material promotion threshold is not automatically a measured variance
estimate.

Mechanical acceptance, scientific interpretation, transfer qualification,
budget decision, and full-scale handoff are separate states.

Do not collapse them into one `accepted=true`.

## READY_FOR_DESKTOP

`READY_FOR_DESKTOP` is a strict evidence package, not a generic positive label.

Generation is fail-closed under the active RML/V5 contract.

A server-side candidate does not become READY merely because a 100K or 500K
metric is positive.

The package must satisfy the required prospective trajectory, qualification,
reference, role, cost, artifact, comparison, and provenance requirements.

A READY package:

- does not wake the desktop;
- does not submit full-scale work;
- does not modify production;
- does not consume protected roles by itself.

Desktop independently decides what to do with READY evidence when it is online.

## Training-trace and early-stop discipline

Do not enable a new early-stop or screening policy merely because partial
training curves exist.

RML backtesting requires strictly comparable trace identities under the active
analysis contract.

Relevant comparability may include:

- scientific contract;
- dataset identity;
- split/row identity;
- architecture identity;
- optimizer;
- LR schedule;
- precision;
- EMA semantics;
- target transform;
- evaluation/selection role;
- terminal endpoint;
- x-axis semantics.

Epoch numbers alone are not generally comparable across different dataset
sizes or training contracts.

Prefer decision-relevant coordinates such as:

- optimizer steps;
- sample presentations;
- matched frozen prefixes.

If the available historical traces are insufficient, the correct result is:

```text
insufficient_evidence
```

Do not replace missing rates or cost savings with zero.

A future screening or early-stop policy must be calibrated prospectively before
activation.

## Conventions

- Repository documentation is written in English for compact machine reading.
- One file should answer one question.
- Do not double-write a fact. If the same fact must be discoverable elsewhere,
  link to its authoritative owner.
- Dated experiment decisions describe historical facts. Do not use them as live
  status documents.
- Avoid live words such as `current`, `default`, `running`, or `next` inside
  dated decision records. Point instead to `CURRENT_STATE.md` or `ROADMAP.md`.
- Directories are named for a role or research question, not for a calendar
  phase.
- `production/` is the delivery line.
- `experiments/` is one research question per directory.
- `platforms/` contains compute-environment adapters and remote operating
  boundaries.
- `research_memory/` contains RML schemas, generated indexes, and RML
  documentation; canonical experiment records remain with the owning
  experiment.
- Comments explain *why*, not *what*.
- Install the project in editable mode with `pip install -e .` using the
  project virtual environment.

## Doc map — single sources of truth

| Question                                                          | Authoritative entry point                                             |
| ----------------------------------------------------------------- | --------------------------------------------------------------------- |
| What is the current project-level recommendation/blocker/routing? | `CURRENT_STATE.md`                                                    |
| What should be worked on next?                                    | `ROADMAP.md`                                                          |
| Which branch owns this task?                                      | `BRANCHES.md`                                                         |
| What do Track A/B/C mean?                                         | `TRACKS.md`                                                           |
| Has this research question/family already been tested?            | `research_memory/README.md` -> RML indexes -> owning experiment       |
| What does the evidence for an experiment actually say?            | the experiment's contract / acceptance / `decision.md` / V5 evidence |
| What ships, and in what order?                                    | `production/README.md`                                                |
| How do I run or reconcile remote work?                            | `platforms/README.md` and the applicable platform handoff             |
| What are the IMS safety boundaries?                               | `platforms/REMOTE_HANDOFF.md`                                         |
| Where should code be edited?                                      | `ARCHITECTURE.md`                                                     |
| How should paths and artifacts be named?                          | `NAMING.md`                                                           |
| How do I install or run basic inference?                          | `README.md`                                                           |
| What are the V5 machine/workflow invariants?                      | `docs/operations/MOLGAP_COMMON_DIRECTION_V5_FINAL.md`                 |
| What are the desktop-specific V5 rules?                           | `docs/operations/DESKTOP_AGENT_HANDOFF_V5_FINAL.md`                   |

## Default scientific workflow

For a new research question, the default reasoning path is:

```text
define observed deficiency / question
    ->
query RML for prior evidence and related closed families
    ->
follow RML pointers to authoritative records
    ->
decide whether existing evidence already answers the question
    ->
NO_TRAIN if no new experiment is decision-relevant
    ->
otherwise create prospective trajectory
    ->
define hypothesis and cheapest falsifier
    ->
run the cheapest justified diagnostic
    ->
if justified, release the next bounded training action
    ->
strict acceptance
    ->
scientific interpretation
    ->
cost / role / trace recording
    ->
RML rebuild
    ->
decide zero or one next justified action
```

For the current V5 server screening funnel, when applicable:

```text
research question
    ->
evidence retrieval
    ->
prospective hypothesis card
    ->
cheap diagnostic
    ->
100K
    ->
strict acceptance + decision
    ->
500K only if qualified and authorized
    ->
strict transfer/cost decision
    ->
READY_FOR_DESKTOP only if fully qualified
    ->
stop server scale-up
```

Desktop full-scale, official evaluation, final submission, and production
promotion remain separate decisions under their owning contracts.

The objective is not to keep accelerators busy.

The objective is to obtain the minimum new evidence required to make the next
scientifically justified decision.
