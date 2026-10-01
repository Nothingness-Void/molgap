# AGENTS - Reading Protocol

MolGap predicts HOMO/LUMO/Gap (eV) for organic electronic molecules.
This file owns navigation and operating boundaries, not history or evidence.
One fact has one authoritative owner: follow pointers rather than copy facts.

## Reuse First

Start with the [modular experiment workflow](docs/operations/EXPERIMENT_WORKFLOW.md),
then use its [reuse map](docs/operations/EXPERIMENT_ADDON_GUIDE.md) and
[local CLI](docs/operations/EXPERIMENT_CLI.md) before adding plumbing. Read them
before a desktop multi-arm runner, source package or prospective record. The
workflow and CLI stage local identity/evidence and registered training; the
platform skill owns remote submission and reconciliation.
Reuse family trainers, checkpoint/inference owners, saved-prediction analysis,
acceptance and V5/RML helpers; inspect the callable and its nearest caller/test.
Platform operations stay in `kaggle-molgap-workloads`,
`ims-molgap-workloads`, and `scnet-bw-dcu-molgap` skills and existing adapters.
Reuse navigation is not scientific, resource, role or submission authorization.

## Delegation Preference

For simple, repetitive or bulk subtasks, use `gpt-6-luna` with `max` reasoning.
For other execution subtasks, use `gpt-6.1-sol` with `medium` reasoning unless
the user specifies otherwise. Do not silently substitute the older `gpt-6-sol`;
report unavailable model/effort combinations instead. The parent owns integration
and acceptance; delegation does not release training or protected-role access.

## Read in This Order

1. This file: navigation and hard boundaries.
2. [BRANCHES.md](BRANCHES.md) before edits, submission, integration or another branch.
3. [CURRENT_STATE.md](CURRENT_STATE.md): recommendation, blockers, routing, queue state.
4. [ROADMAP.md](ROADMAP.md): only the relevant priorities and triggers.
5. [TRACKS.md](TRACKS.md) only for a Track A/B/C task.
6. [ARCHITECTURE.md](ARCHITECTURE.md): code ownership; [NAMING.md](NAMING.md): naming.
7. The modified tree's README: production, experiments, platforms or research_memory.
8. Owning experiment protocol, decision, acceptance, evidence and artifact records.
9. Only code required by the task.

Read [Research Protocol](docs/operations/RESEARCH_PROTOCOL.md) when designing
research, accepting/interpreting terminal results, comparing/promoting models,
or backtesting screening/early stopping. Do not load every historical document.
Inactive phase 1-7, retired production and superseded histories live in archive
or explicit archive pointers; never infer live state or an open question there.

## Authority

CURRENT_STATE owns only the project-level recommendation, blocker, routing,
queue state and next actions. It cannot override frozen contracts, dated
decisions, acceptance, canonical V5/RML evidence, provenance or scheduler truth.
Reconcile conflicts and correct a stale summary; never rewrite evidence to fit it.

Precedence: authoritative remote scheduler / durable artifacts ->
experiment contract / decision / acceptance -> canonical V5 / RML records ->
CURRENT_STATE -> generated RML indexes / summaries -> agent interpretation.
Generated summaries are navigation, not primary scientific evidence.

## RML and Research Entry

- Query [RML](research_memory/README.md) and targeted derived indexes before
  proposing, modifying or submitting a new scientific question. Follow their
  pointers to canonical evidence before deciding; do not repeat a completed
  experiment without a new decision-relevant question.
- Canonical trajectory, cost, role, trace and handoff records stay beside the
  owning experiment. Never manually edit `research_memory/derived/`.
- For a genuinely new route, publish canonical `trajectory.json` with
  `record_mode = prospective` **before any new diagnostic or training**.
  The required evidence review, hypothesis and cheapest falsifier are detailed
  in [Research Protocol](docs/operations/RESEARCH_PROTOCOL.md#new-research-question-protocol).
- A trajectory does not authorize training. NO_TRAIN, NEGATIVE_UNDER_CONTRACT,
  INCONCLUSIVE, STOP_FOR_COST and DUPLICATE_EVIDENCE are valid contract outcomes.
  Advance diagnostic/100K/500K/full gates only when required and authorized.
- Update canonical records and rebuild RML after accepted research milestones,
  not each epoch/heartbeat. Verify derived state before handoff or commit.
- Import `retrospective_partial` only from pre-existing authoritative records.
  Missing hardware, cost, roles, rationale, traces and conclusions stay unknown;
  never invent retrospective hypotheses or evidence to justify launched work.

## Terminal Attribution

After accepting a module terminal result, write concise failure-mode attribution
beside its decision **before another unrelated module**. Query that disposition
with RML before selecting another module in the same family. Use retained
contract/artifact/trace/role/cost evidence and existing analysis helpers.
Infrastructure-only or NO_TRAIN closure needs the reason and missing discriminator.
Follow the [detailed attribution checklist](docs/operations/RESEARCH_PROTOCOL.md#terminal-attribution-before-another-module).
Do not infer underfitting, overfitting, exposure shortage or module harm from one
endpoint or mismatched train/development metrics. Missing comparator/cohort/trace/
checkpoint means `insufficient_evidence`. Preserve the accepted decision;
do not invent a reference or repeat training merely to complete attribution.
Attribution is interpretation, not a replacement RML schema or promotion gate.

## Hard Constraints

- **Python:** use `.venv\Scripts\python.exe` on this Windows checkout, not
  system Python. Install editable with that environment's `pip install -e .`.
- **Targets:** `homo`, `lumo`, `gap` in eV under applicable PubChemQC /
  B3LYP Kohn-Sham contracts; no experimental or alternate target substitution.
- **Geometry:** generated-3D training/inference contracts must be compatible.
  Production generated-3D inference uses ETKDG. No PM6-trained/ETKDG-inference
  pairing without a new validating contract. Pure-2D must not construct 3D.
- **Reuse:** shared logic belongs in `src/molgap/`; experiment scripts are thin
  wrappers, not duplicate model classes. Public inference is
  `src/molgap/inference.py`, lazily exported by `src/molgap/__init__.py`.
- **Tests:** test locally before delivery; claim execution only if it occurred.
- **Durability:** remote training/inference must retain immutable source/config,
  input/cache identity, atomic checkpoints or resumable state, independently
  retrievable durable artifacts, exact job identity and provenance. A transient
  worker filesystem or uninterrupted process cannot be the only result copy.
- **Resources:** obey platform/experiment contracts. Parsing, ETKDG/graph
  construction and graph acceptance normally use CPU before GPU/DCU release.
  Accelerators serve accelerator work; do not change batch/precision/optimizer
  semantics merely for utilization.
- **Cost:** keep A100, T4, DCU, CPU, wall and queue units separate. Unknown is not
  zero; estimated is not measured; not-applicable is not missing.
- **IMS:** before **any** command, including read-only discovery/metadata, read
  and obey [REMOTE_HANDOFF](platforms/REMOTE_HANDOFF.md). Stay within its resolved
  path/safety boundary, below `/lustre/home/users/sm2/chou/`; never bypass it.
- **Protected roles:** no consumption merely for implementation, RML indexing,
  debugging or infrastructure tests. Explicit applicable role authority is required.
- **No fabrication:** missing reference, cost, role, artifact or comparison stays
  missing/pending. No silent reference retraining or synthetic bookkeeping evidence.

## Branch Ownership

[BRANCHES.md](BRANCHES.md) owns checkout mapping and terminal Git routing.
Compute host does not determine ownership: source/protocol/branch decisions do.

- `molgap-desktop`: integration, full training, official evaluation, submission,
  and explicitly desktop-owned selected 500K work.
- `molgap-server`: independent bounded 100K/500K research and server A/B support.
- `master`: stable delivery only; owner-to-master promotion is separate/reviewed.
- `archive`: exact inactive, rejected, superseded or historical source/evidence.

Every new desktop question starts a dedicated `codex/exp/` branch in a separate
worktree from the verified current molgap-desktop tip. Keep implementation,
submission, reconciliation, acceptance, decision and terminal RML there until
closure; seeds/retries/infrastructure attempts for that question stay there.
Positive results merge to desktop only when adopted. Accepted diagnostics with
reusable implementation follow BRANCHES' explicit non-promotion route.
Negative complete histories go to archive; keep accepted canonical evidence
discoverable in desktop RML without rejected implementation, then rebuild/check.
A finished remote attempt does not close or route an unresolved experiment.
Use an owning branch when safe, otherwise a temporary `codex/` isolation branch.
Integrate only reviewed reusable work with provenance; never wholesale merge an
unrelated branch for a helper. Delete a completed temporary ref only after its
commits are durably reachable from its owning long-lived branch or archive.

## Desktop Offline / V5

Desktop is independent of server A/B: no default heartbeat, Conversation B,
server fallback or takeover. Offline silent time is accepted; no server agent
automatically monitors, adopts, repairs or advances desktop jobs.
Freeze source/config/input identity and preserve resumable checkpoints, durable
outputs and job provenance before submission; reconcile when desktop returns.
Server independently owns its bounded loop; never duplicate/take over its jobs.

Do not add desktop-to-server monitor handoff, server-to-desktop wakeup,
automatic takeover in either direction, shared cross-machine live DB/SQLite,
owner leases, fallback polling or automatic conversation bridges.
Exchange asynchronously through reviewed Git, canonical records, RML/V5
indexes, durable artifacts and user-directed integration.
Read the [V5 common contract](docs/operations/MOLGAP_COMMON_DIRECTION_V5_FINAL.md)
and [desktop handoff](docs/operations/DESKTOP_AGENT_HANDOFF_V5_FINAL.md) at bootstrap.
Changing V5 architectural invariants requires explicit user approval of a future
contract version; implementation difficulty never grants redesign authority.

## Remote Reconciliation

After downtime or stale context, before acting:
1. Identify owning experiment/branch and exact run/job/attempt.
2. Query authoritative remote state under the platform boundary.
3. Verify source/config/release identity and retrieve/verify durable artifacts.
4. Separate infrastructure failure from science; perform applicable acceptance.
5. Update canonical evidence/decision/cost/roles; rebuild RML for accepted milestones.
6. Only then choose zero or one next justified action.

Never submit a successor just because local status is stale.
Unknown remote state is `UNKNOWN`, not failure.

## Comparison, READY and Early Stop

Follow [Research Protocol](docs/operations/RESEARCH_PROTOCOL.md) and the frozen
contract for identity, strict reference/runtime, aligned finite artifacts, hashes,
exposure/resume, paired uncertainty, role history and actual native-cost checks.
Missing strict reference means pending, not baseline retraining.
Row bootstrap is not training stochasticity; material gates are not measured variance.
Mechanical acceptance, science, transfer, budget and full handoff remain separate,
never one overloaded `accepted=true`. Track B never silently changes Track A.

`READY_FOR_DESKTOP` is fail-closed evidence under RML/V5, requiring prospective
trajectory, qualification, reference, roles, cost, artifacts, comparison and
provenance, not merely a positive 100K/500K score. Server stops scale-up there.
READY does not wake desktop, launch full training, modify production or consume
protected roles; desktop decides independently when online.
No new early-stop/screen policy from partial curves alone. Require comparable
trace identities and prospective calibration; epoch numbers alone are not
comparable across sizes/contracts. Missing evidence/rates/savings stay
`insufficient_evidence`, not zero. See the conditional detailed protocol.

## Conventions and Navigation

English compact docs; one file/question and one authoritative fact, linked elsewhere.
Dated decisions are historical, not live status; avoid current/default/running/next
there and point to CURRENT_STATE or ROADMAP. Name directories by role/question.
production owns delivery; experiments one question/directory; platforms adapters
and remote boundaries; research_memory schemas/docs/generated indexes, not canonical
experiment records. Comments explain why, not what.
Basic install/inference: [README](README.md); shipping: [production](production/README.md).
Remote routing: [platforms](platforms/README.md); closed evidence: [experiments](experiments/README.md).
Detailed research checklists: [Research Protocol](docs/operations/RESEARCH_PROTOCOL.md).
Historical branch audits: [Branch History](docs/operations/BRANCH_HISTORY.md).
Obtain the minimum new evidence needed for the next justified decision, not busy GPUs.
