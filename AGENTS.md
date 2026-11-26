# AGENTS - Reading Protocol

MolGap predicts HOMO/LUMO/Gap (eV) for organic electronic molecules.
This file owns navigation and hard boundaries, not history or evidence; follow pointers rather than copying facts.

## Read Routes by Role

Start from your task row; read only what the task requires (never load the full archive, and never treat `experiments/_closed/` as active behavior).

| Role | First read |
|---|---|
| Find code owner, verifier, dashboard, or SCNet/IMS/Kaggle script | [ARCHITECTURE.md](ARCHITECTURE.md) before grepping code |
| Propose / submit a new experiment | [LIFECYCLE_MAP.md](docs/operations/LIFECYCLE_MAP.md), [quickstart](docs/operations/EXPERIMENT_QUICKSTART.md), then platform skill in `skills/` |
| Retrieve, accept, close a run | platform skill in `skills/`, [LIFECYCLE_MAP.md](docs/operations/LIFECYCLE_MAP.md), then [RML interfaces](docs/operations/RML_INTERFACES.md) |
| Plan research | [families.md](research_memory/derived/families.md), family page, [Research Protocol](docs/operations/RESEARCH_PROTOCOL.md), then RML interfaces |
| Change infrastructure | [ARCHITECTURE.md](ARCHITECTURE.md), [quickstart](docs/operations/EXPERIMENT_QUICKSTART.md), then [reuse map](docs/operations/EXPERIMENT_ADDON_GUIDE.md) |
| Look up status, branches, priorities, naming, or notes | [CURRENT_STATE.md](CURRENT_STATE.md), [BRANCHES.md](BRANCHES.md), [ROADMAP.md](ROADMAP.md), [TRACKS.md](TRACKS.md), [NAMING.md](NAMING.md), [vault](docs/operations/VAULT.md) |

Run local CLIs via `platforms/local/run.py` (`-- --help` or `--module molgap.research_memory -- --help`) for live flags and valid schema choices; changing a command requires updating its row in [LIFECYCLE_MAP.md](docs/operations/LIFECYCLE_MAP.md).

## Authority

`CURRENT_STATE.md` owns only project-level recommendations, blockers, queue state and next actions; never rewrite evidence or fabricate missing references, costs, roles or comparisons.
Precedence: authoritative remote scheduler / durable artifacts -> experiment contract / decision / acceptance -> canonical V5 / RML records -> `CURRENT_STATE.md` -> generated RML indexes / summaries -> agent interpretation.

## Hard Constraints

- **Python & Windows checkout:** use `.venv\Scripts\python.exe`, never system Python. Reuse the primary checkout `.venv` across worktrees (never `pip install -e .` from a temporary worktree) and run CLIs via `platforms/local/run.py` so a worktree runs its own source.
  `.venv\Lib\site-packages\__editable__.molgap-0.1.0.pth` must hold only the ASCII junction `C:\molgap-src` (non-ASCII paths crash under GBK/`-X utf8`); if overwritten, rebuild with `cmd /c mklink /J C:\molgap-src <primary checkout>\src` and restore that line.
- **Targets & geometry:** `homo`, `lumo`, `gap` in eV under PubChemQC / B3LYP Kohn-Sham contracts only (not experimental solid-state IP/EA). Generated-3D production inference uses ETKDG (no PM6-trained/ETKDG-inference pairing without a validating contract); pure-2D must not construct 3D.
- **Tail holdout (`tail_holdout_v1`) & protected roles:** the last 128,606 training rows (rows 3,250,000..3,378,605, shards 0065-0067) are permanently reserved — never train, tune, select, or upload to Kaggle; score each final checkpoint offline once via `platforms/local/pcqm_tail_holdout_eval.py` (`D:/molgap-artifacts/tail_holdout_master_leaderboard.json`). Tail MAE (~7-8 meV above 50K dev MAE) is not comparable across scales. Official valid is report-only; protected roles require explicit authority and are never consumed for debugging/tests.
- **IMS safety:** before **any** IMS command (including read-only discovery), read and obey [REMOTE_HANDOFF](platforms/REMOTE_HANDOFF.md) and stay under `/lustre/home/users/sm2/chou/`.
- **Simplicity, reuse & testing:** shared logic belongs in `src/molgap/`; wrappers stay thin; public inference is `src/molgap/inference.py`. Reuse platform skills (`kaggle-molgap-workloads`, `ims-molgap-workloads`, `scnet-bw-dcu-molgap`), [training logging](docs/operations/TRAINING_LOGGING.md), and existing family trainers/verifiers. Follow [verification conventions](docs/operations/INFRASTRUCTURE_VERIFICATION.md): reproduce in the nearest real workflow, run affected checks once, and rework if test code growth exceeds feature code growth.
- **Durability & resources:** remote runs retain immutable source/config, resumable checkpoints, independently retrievable durable artifacts, and provenance. Parsing/ETKDG/graph checks run on CPU first; keep A100/T4/DCU/CPU/wall/queue units separate (`unknown != 0`, `estimated != measured`).
- **Infra budget, benchmark gate & new experiments:** before an infra commit, run `.venv\Scripts\python.exe platforms/local/infra_budget.py` ([INFRA_BUDGET.json](docs/operations/INFRA_BUDGET.json); raise caps only with user approval, bug fixes <= 20 net lines preapproved) and `tests/test_documentation_navigation.py`. For agent-facing docs/skills/CLI infra changes, also run `platforms/local/agent_infra_bench.py` (`--base HEAD --candidate WORKTREE --suite full`) and require `IMPROVED` (exit 0). A new experiment adds only `experiments/<q>/`, one recipe/spec, and <= 1 self-contained addon module + catalog entry — never edit shared trainers, verifiers, `.gitattributes`, or tests directly.

## Research, RML, Reconciliation & Branches

- **RML & attribution:** query [RML](research_memory/README.md) (`lookup`/`levers`) before proposing work; publish prospective `trajectory.json` before training ([Research Protocol](docs/operations/RESEARCH_PROTOCOL.md#new-research-question-protocol)); never hand-edit `research_memory/derived/`. Shared terminal closure updates RML once (`terminal-rebuild` is recovery-only). Write concise failure-mode attribution after accepting a module terminal result ([attribution checklist](docs/operations/RESEARCH_PROTOCOL.md#terminal-attribution-before-another-module)); missing evidence stays `insufficient_evidence`.
- **Gates & reconciliation:** keep mechanical acceptance, science, transfer, budget, and `READY_FOR_DESKTOP` separate; Track B never silently changes Track A; no early-stop policy from partial curves alone. After downtime/stale context: identify run/branch -> query authoritative remote state -> verify/retrieve durable artifacts -> separate infra from science -> update records/RML -> take zero or one next action. `UNKNOWN` is not failure; reconcile `SUBMIT_UNKNOWN` journals (`PHASE_NEXT` in `kaggle_submission.py`) before retry; lowercase `complete` and uppercase run statuses are separate vocabularies.
- **Branches & worktrees:** [BRANCHES.md](BRANCHES.md) owns `molgap-desktop` / `molgap-server` / `master` / `archive` routing. Start desktop questions on `codex/exp/` worktrees from verified `molgap-desktop`, merge positives to desktop, route complete negatives to `archive`, and remove completed worktrees promptly ([local storage lifecycle](docs/operations/LOCAL_STORAGE_RETIREMENT.md)).

## Desktop Offline / V5

Desktop and server A/B are independent: offline silent time is accepted, with no default heartbeat, Conversation B, server fallback, or takeover.
Do not add desktop-to-server monitor handoff, server-to-desktop wakeup, automatic takeover in either direction, shared cross-machine live DB/SQLite, owner leases, fallback polling, or automatic conversation bridges.
Exchange asynchronously via reviewed Git, canonical records, RML/V5 indexes, durable artifacts, and user-directed integration; read [V5 common contract](docs/operations/MOLGAP_COMMON_DIRECTION_V5_FINAL.md) and [desktop handoff](docs/operations/DESKTOP_AGENT_HANDOFF_V5_FINAL.md) only when changing topology (requires user approval).

---
Basic install/inference: [README](README.md) | Shipping: [production](production/README.md) | Remote routing: [platforms](platforms/README.md) | Closed evidence: [experiments](experiments/README.md) | Docs map: [docs/README.md](docs/README.md) | Skills: [skills/](skills/README.md).
Obtain the minimum new evidence needed for the next justified decision, not busy GPUs.
