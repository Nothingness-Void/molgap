# AGENTS — Reading Protocol

MolGap: ML prediction of HOMO/LUMO/Gap (eV) for organic electronic molecules,
trained on PubChemQC B3LYP/6-31G* data. This file is **how to navigate the repo**,
not a project description. One fact lives in one place — follow the links.

## Read in this order
1. **This file** — protocol + hard constraints (below).
2. **`CURRENT_STATE.md`** — the only source of "what's true now": recommended model,
   conclusions, blocker, next actions. If anything conflicts, this wins.
3. **`ROADMAP.md`** — task priorities / backlog (read the relevant section only).
4. **`TRACKS.md`** — read only when a task is labeled A, B, or C.
5. **`ARCHITECTURE.md`** — tree and code map; tells you which file owns a change.
   Naming rules are in **`NAMING.md`**.
6. The one `README.md` for the tree you are working in: `production/`,
   `experiments/`, or `platforms/`. Then the specific decision record it links.
7. The specific code files your task touches.

`production/history/` retains the frozen phase 1-7 records. Never infer the
current model or an open question from that historical tree.

Do not read all docs to find "the current truth" — it's in `CURRENT_STATE.md`.

## Hard constraints (do not break)
- **Python**: always `.venv\Scripts\python.exe` — system Python lacks torch/pyg.
- **Train-inference consistency**: training and inference MUST use the same conformer
  method (ETKDG). Never mix PM6 training coords with ETKDG inference.
- **Targets**: `homo`/`lumo`/`gap` (eV, B3LYP Kohn-Sham), NOT experimental values.
- **Reuse, don't fork**: reusable logic lives in `src/molgap/` only; each stage's
  or experiment's `scripts/` are thin CLI wrappers. Public inference is implemented in `src/molgap/inference.py`
  and lazily exported by `src/molgap/__init__.py`. Don't redefine model classes
  in scripts. See `ARCHITECTURE.md`.
- **Don't re-run completed experiments** — cite the experiment's own decision
  record under `experiments/`.
- **Test scripts locally before delivering.**
- **Remote durability**: every cloud job MUST checkpoint progress atomically and
  produce independently retrievable output chunks. Never rely on a transient
  worker filesystem or a single long-running task as the only copy of results.
- **Remote resource separation**: use high-memory CPU jobs for parsing,
  ETKDG/graph construction, and graph acceptance. Submit GPU/DCU jobs only
  after an immutable graph cache passes acceptance; GPU/DCU time is reserved
  for encoder training, embedding extraction, and fusion.
- **IMS access boundary**: before any molecular-research-server command, read
  and obey the safety boundary in `platforms/REMOTE_HANDOFF.md`. It restricts
  all path access, including read-only discovery and metadata probes.

## Branch governance

The repository has four long-lived branches with non-overlapping roles:

- **`master`** is the minimal, stable delivery branch. It receives only
  validated, optimized, reviewable production content; exploratory work and
  remote-run history do not accumulate there.
- **`molgap-server`** is the integration branch for this server-side agent. New
  architecture protocols, Kaggle screens, runners, acceptance logic, compact
  evidence, and server-side decisions land here.
- **`molgap-desktop`** is the independent desktop integration branch. It owns
  full training, official evaluation, and final submission work rather than
  server-side architecture discovery.
- **`archive`** retains rejected or inactive experiment implementations,
  process records, and provenance that must remain reproducible but should not
  stay in an active integration branch.

Do not create one long-lived branch per experiment. Work directly on the
owning integration branch when safe. Create a short-lived `codex/<topic>` branch
only when parallel work, worktree isolation, or a risky change requires it.
Afterward, merge accepted work into `molgap-server`, preserve rejected compact
evidence in `archive`, verify the commits are reachable remotely, and delete
the inactive temporary branch.

Before branching or integrating, fetch the remote, inspect the worktree, and
preserve unrelated user changes. Never overwrite another machine's branch.
`molgap-server` and `molgap-desktop` exchange only reviewed, explicit commits;
neither branch is a scratch copy of the other.

Keep commits classified and independently traceable:

- `docs(...)` freezes protocols, decisions, and authority boundaries;
- `feat(...)` adds an architecture, runner, or acceptance capability;
- `fix(...)` repairs implementation or infrastructure without disguising a
  scientific-contract change;
- `ops(...)` records remote packaging, submission, retrieval, or handoff.

Large caches, checkpoints, models, logs, and prediction payloads remain in
ignored local/platform record storage. Git receives compact metrics, hashes,
manifests, acceptance records, and decisions. A server experiment must pass its
declared confirmation gate before desktop full training, and only a validated
delivery candidate may be promoted from the integration branches to `master`.

## Conventions
- Docs in English (LLM efficiency). One file answers one question.
- Don't double-write a fact; if it must appear twice, the second is a link.
- Decision records use dated historical language. Never write `current`,
  `default`, `running`, or `next` there; point to `CURRENT_STATE.md` or
  `ROADMAP.md`.
- Directories are named for a **role or question**, never a calendar phase.
  `production/` is the delivery line, `experiments/` is one question per
  directory, `platforms/` is compute-environment adapters.
- Comments explain *why*, not *what*.
- Install: `pip install -e .` (editable, via pyproject.toml).

## Doc map (single sources of truth)
| Question | File |
|----------|------|
| What's true now? | `CURRENT_STATE.md` |
| What to do next? | `ROADMAP.md` |
| What do Track A/B/C mean? | `TRACKS.md` |
| How was it done? | the experiment's `decision.md` under `experiments/` |
| What ships, and in what order? | `production/README.md` |
| How do I run this remotely? | `platforms/README.md` |
| Where to edit code? | `ARCHITECTURE.md` |
| How should paths and artifacts be named? | `NAMING.md` |
| How to install / basic inference? | `README.md` |
