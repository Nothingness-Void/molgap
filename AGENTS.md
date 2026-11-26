# AGENTS - Reading Protocol

MolGap predicts HOMO/LUMO/Gap (eV) for organic electronic molecules.
This file owns navigation and operating boundaries, not history or evidence.
One fact has one authoritative owner: follow pointers rather than copy facts.

## Read Routes by Role

Start from the row for the task. Each route goes to the
[lifecycle map](docs/operations/LIFECYCLE_MAP.md), then
[families.md](research_memory/derived/families.md) (closed and open items per model
family), then the family page and the last page named.

| Role | Then read |
|---|---|
| Propose / submit a new experiment | [quickstart](docs/operations/EXPERIMENT_QUICKSTART.md), then platform skill in `skills/` |
| Retrieve, accept, close a run | platform skill in `skills/`, then [RML interfaces](docs/operations/RML_INTERFACES.md) (closure format) |
| Plan research | [Research Protocol](docs/operations/RESEARCH_PROTOCOL.md), then RML interfaces (planning section) |
| Change infrastructure | [quickstart](docs/operations/EXPERIMENT_QUICKSTART.md), then the [reuse map](docs/operations/EXPERIMENT_ADDON_GUIDE.md) |
| Look up past notes | [vault](docs/operations/VAULT.md) (a view, never an authority) |

Read only what the task requires; do not load the full bootstrap or archive set.
Consult [BRANCHES.md](BRANCHES.md) for custody/branch routing, [CURRENT_STATE.md](CURRENT_STATE.md)
for live queue/blockers, [ROADMAP.md](ROADMAP.md) and [TRACKS.md](TRACKS.md) for priorities,
[ARCHITECTURE.md](ARCHITECTURE.md) for code ownership, and [NAMING.md](NAMING.md) for naming changes.
Run local CLIs through `platforms/local/run.py` (`-- --help` or `--module molgap.research_memory -- --help`)
for live flags and valid schema choices; changing a command requires updating its row in the lifecycle map.

## Authority

`CURRENT_STATE.md` owns only project-level recommendations, blockers, queue state and next actions.
It never overrides frozen contracts, dated decisions, acceptance, canonical V5/RML evidence, or remote scheduler truth.
Reconcile conflicts and fix stale summaries; never rewrite evidence or fabricate missing references, costs, roles or comparisons.

Precedence: authoritative remote scheduler / durable artifacts ->
experiment contract / decision / acceptance -> canonical V5 / RML records ->
`CURRENT_STATE.md` -> generated RML indexes / summaries -> agent interpretation.

## Hard Constraints

- **Python & Windows checkout:** use `.venv\Scripts\python.exe`, not system Python.
  Reuse the primary checkout `.venv` across worktrees (never `pip install -e .` from a temporary worktree).
  Run CLIs via `platforms/local/run.py` so a worktree runs its own source.
  `.venv\Lib\site-packages\__editable__.molgap-0.1.0.pth` must hold only the ASCII junction `C:\molgap-src`
  (non-ASCII paths crash under GBK/`-X utf8`); if rewritten, rebuild with
  `cmd /c mklink /J C:\molgap-src <primary checkout>\src` and restore that line.
- **Targets & geometry:** `homo`, `lumo`, `gap` in eV under PubChemQC / B3LYP Kohn-Sham contracts only.
  Production generated-3D inference uses ETKDG; no PM6-trained/ETKDG-inference pairing without a new validating contract.
  Pure-2D must not construct 3D.
- **Tail holdout (`tail_holdout_v1`) & protected roles:** the last 128,606 training rows
  (rows 3,250,000..3,378,605, shards 0065-0067) are permanently reserved; never train, tune or select on them,
  never upload them to Kaggle, and score each final checkpoint offline once via `platforms/local/pcqm_tail_holdout_eval.py`
  (scores live in `D:/molgap-artifacts/tail_holdout_master_leaderboard.json`).
  Tail MAE (~7-8 meV higher than 50K internal dev MAE) is a different scale: never compare across them.
  Official valid is report-only; protected roles require explicit authority and are never consumed for debugging or tests.
- **IMS safety:** before **any** IMS command (including read-only discovery), read and obey
  [REMOTE_HANDOFF](platforms/REMOTE_HANDOFF.md) and stay under `/lustre/home/users/sm2/chou/`.
- **Simplicity & reuse:** prefer the simplest implementation that satisfies the current need; avoid speculative abstractions.
  Shared logic belongs in `src/molgap/`; experiment scripts stay thin wrappers; public inference is `src/molgap/inference.py`.
  Reuse platform skills (`kaggle-molgap-workloads`, `ims-molgap-workloads`, `scnet-bw-dcu-molgap`),
  [standard training logging](docs/operations/TRAINING_LOGGING.md), and existing family trainers/verifiers.
  Test locally before delivery ([verification conventions](docs/operations/INFRASTRUCTURE_VERIFICATION.md)):
  reproduce failures in the nearest existing real workflow, run affected checks once, and rework if test code growth exceeds feature code growth (no bloated test scaffolding).
- **Durability, resources & cost:** remote runs must retain immutable source/config, resumable checkpoints,
  independently retrievable durable artifacts, and provenance. Parsing/ETKDG/graph checks belong on CPU before GPU/DCU release.
  Keep A100, T4, DCU, CPU, wall and queue units separate (unknown != zero; estimated != measured).
- **Infrastructure budget & new experiments:** before an infrastructure commit, run
  `.venv\Scripts\python.exe platforms/local/infra_budget.py` (limits in [INFRA_BUDGET.json](docs/operations/INFRA_BUDGET.json))
  and `tests/test_documentation_navigation.py`. Raise budget limits only with user approval (bug fixes <= 20 net lines preapproved).
  For agent-facing docs, skills or CLI infrastructure changes, also run `platforms/local/agent_infra_bench.py`
  (`--base HEAD --candidate WORKTREE --suite full`) and require an `IMPROVED` gate verdict (exit 0: zero regression + measurable quality or efficiency gain).
  A new experiment adds files under `experiments/<q>/`, one recipe/spec, and at most one self-contained addon module + catalog entry;
  it does not edit shared trainers, verifiers, `.gitattributes` or tests.

## Research, RML, Branches and Reconciliation

- **RML & attribution:** query [RML](research_memory/README.md) (`lookup`/`levers`) before proposing new work;
  publish prospective `trajectory.json` before new diagnostic/training ([Research Protocol](docs/operations/RESEARCH_PROTOCOL.md#new-research-question-protocol));
  never manually edit `research_memory/derived/`. Shared terminal closure updates RML once (`terminal-rebuild` is recovery-only).
  After accepting a module terminal result, write concise failure-mode attribution before another unrelated module
  ([attribution checklist](docs/operations/RESEARCH_PROTOCOL.md#terminal-attribution-before-another-module)); missing evidence stays `insufficient_evidence`.
- **Comparisons & gates:** mechanical acceptance, science, transfer, budget and `READY_FOR_DESKTOP` remain separate;
  Track B never silently changes Track A. No new early-stop/screen policy from partial curves alone.
- **Remote reconciliation:** after downtime or stale context, identify run/branch -> query authoritative remote state ->
  verify/retrieve durable artifacts -> separate infra failure from science -> update canonical records/RML -> choose zero or one next action.
  `UNKNOWN` is not failure; never submit a successor on stale local status. Reconcile `SUBMIT_UNKNOWN` journals (`PHASE_NEXT` in `kaggle_submission.py`) before retry.
  Lowercase persisted `complete` and uppercase run statuses are separate vocabularies.
- **Branches & worktrees:** [BRANCHES.md](BRANCHES.md) owns `molgap-desktop` / `molgap-server` / `master` / `archive` routing.
  Start each new desktop question on a `codex/exp/` worktree from verified `molgap-desktop`;
  merge adopted positives to desktop, route complete negatives to `archive` (keeping only accepted canonical evidence in desktop RML),
  and remove completed worktrees promptly after durable preservation ([local storage lifecycle](docs/operations/LOCAL_STORAGE_RETIREMENT.md)).

## Desktop Offline / V5

Desktop is independent of server A/B: no default heartbeat, Conversation B,
server fallback or takeover. Offline silent time is accepted; no server agent
automatically monitors, adopts, repairs or advances desktop jobs.
Server independently owns its bounded loop; never duplicate/take over its jobs.

Do not add desktop-to-server monitor handoff, server-to-desktop wakeup,
automatic takeover in either direction, shared cross-machine live DB/SQLite,
owner leases, fallback polling or automatic conversation bridges.
Exchange asynchronously through reviewed Git, canonical records, RML/V5
indexes, durable artifacts and user-directed integration.
Read the [V5 common contract](docs/operations/MOLGAP_COMMON_DIRECTION_V5_FINAL.md)
and [desktop handoff](docs/operations/DESKTOP_AGENT_HANDOFF_V5_FINAL.md) only when
changing ownership/topology; V5 invariants require explicit user approval to change.

## Navigation

Basic install/inference: [README](README.md) | Shipping: [production](production/README.md) |
Remote routing: [platforms](platforms/README.md) | Closed evidence: [experiments](experiments/README.md) |
Docs map: [docs/README.md](docs/README.md) | Project skills (authority): [skills/](skills/README.md).
Obtain the minimum new evidence needed for the next justified decision, not busy GPUs.
