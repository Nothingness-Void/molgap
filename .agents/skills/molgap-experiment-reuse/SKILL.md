---
name: molgap-experiment-reuse
description: Reuse existing MolGap owners when changing trainers, packaging, submission preparation, acceptance or RML closure. Not needed for routine command execution or healthy monitoring.
---

# MolGap experiment reuse

Use the selected checkout and its project Python. Follow its `AGENTS.md` for
ownership and authority. Reuse is not permission to train, retry, consume a
protected role or promote a model.

## Choose the existing entry

| Task | Existing entry |
| --- | --- |
| Prepare/submit a registered Kaggle experiment | Platform skill's `platforms/kaggle/submit_workflow.py --experiment ...` |
| Recover interrupted preparation | Same submitter with `--prepared ... --recover-prepared`; retain original identity |
| Retrieve known Kaggle outputs | `platforms/kaggle/retrieve_family_outputs.py --all-arms`; it pins the receipt version's manifests itself |
| Diagnose/accept a retrieved run | `platforms/local/accept_run.py`; run identity, reviewed closure input and `--strict` where applicable |
| Local single-model exercise or closure | `platforms/local/train.py`; its `--accept-only` route handles local prepared runs |
| IMS / SCNet workload | Its platform skill and existing adapter, not Kaggle's front door |
| Change a model or Addon | Existing family trainer, catalog and output owners; read the applicable section of `docs/operations/EXPERIMENT_ADDON_GUIDE.md` |

Routine use does not require inspecting Python signatures or reading tests.
Read `docs/operations/EXPERIMENT_QUICKSTART.md` only for a missing input shape;
read the matching CLI section for advanced options. Do not chain low-level
validation commands before a front door that already performs them.

Prior art for a new idea (innovation layer): besides RML replay, grep
`experiments/**/*.md` audit tables (e.g. `experiments/pcqm_gap_architecture/architecture_route_audit.md`),
since RML has no by-mechanism query; if WebSearch fails, WebFetch the arXiv abstract.

## When implementation is required

Name the existing owner and the specific unsupported behaviour in the work
update. Inspect that callable and its nearest real caller/test, then make the
smallest change there. Use [entrypoints](references/entrypoints.md) only when the
owner is unclear. This is navigation, not another mandatory report or checklist.

Shared code belongs in `src/molgap/`; experiment wrappers stay thin. Reuse
identity, atomic IO, packaging, family training, saved-prediction analysis and
terminal helpers. Do not copy a whole experiment or create a parallel framework.
Keep active frozen payloads unchanged. Tests follow the repository's affected
fixed nodes; strengthen an existing flow instead of adding a default test gate.

## Important capability limits

- `run-diagnostic` only probes metadata or constructs a random model. For saved
  predictions use `diagnose`; for frozen-checkpoint inference use the accepted
  loader and matching factory. Neither command grants role access.
- A prepared directory is not a submission; a receipt or scheduler completion
  is not scientific acceptance. Resolve unknown remote effects in the original
  platform journal before a successor.
- Acceptance generates objective evidence. Outcomes, decisions and observed
  protected-role use must be supplied from actual facts; missing evidence stays
  pending. RML entry alone does not establish strict replay qualification.

Report the executed path, any substitutes and remaining blockers. Do not claim
completeness from test counts. Follow the repository's terminal-attribution rule
before selecting another unrelated scientific module.
