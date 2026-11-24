# Lifecycle map

Verified by: tests/test_documentation_navigation.py::test_lifecycle_map_commands_exist

Five stages: research, plan, propose, accept, archive. The archive feeds the
next round of research. This page owns the command order. Flags live in the
[CLI reference](EXPERIMENT_CLI.md); gates and attribution live in the
[research protocol](RESEARCH_PROTOCOL.md). A command change must change its row
here (the documentation navigation test above checks it; it runs in the `accept` scope of `molgap.local_testing`).

## Where to start

- **Retrieve, accept, close:** the platform skill ([Kaggle](../../skills/kaggle-molgap-workloads/SKILL.md),
  [IMS](../../skills/ims-molgap-workloads/SKILL.md)), this page, [families.md](../../research_memory/derived/families.md),
  that family's page under `research_memory/derived/families/`, then the
  [closure format](RML_INTERFACES.md#decisionfacts-molgap-closure-facts-v1).
- **Plan research:** [families.md](../../research_memory/derived/families.md), the family page
  (for example [gptrans](../../research_memory/derived/families/gptrans.md)), this page,
  the [research protocol](RESEARCH_PROTOCOL.md), then [planning with RML](RML_INTERFACES.md#planning-with-the-evidence).
- **Change infrastructure:** this page, [families.md](../../research_memory/derived/families.md),
  [execution_infra](../../research_memory/derived/families/execution_infra.md),
  the [quickstart](EXPERIMENT_QUICKSTART.md), then the [reuse map](EXPERIMENT_ADDON_GUIDE.md#pick-the-operation).

## Commands
`rml` means `platforms/local/run.py --module molgap.research_memory -- --repo-root .`;
`cli` means `platforms/local/run.py -- ` (runs `molgap.experiment_cli`).
Who: M = machine does it, H = a person (or the reviewing agent) decides.

| Stage | Command | Who | Reads | Writes | Next reader |
|---|---|---|---|---|---|
| Research | `rml lookup` | M | derived indexes, cards | stdout | the planner |
| Research | `rml cards` | M | derived indexes | stdout | the planner |
| Research | `rml levers --family F` | M | family cards | stdout (tested levers, noise) | the planner |
| Research | `rml gaps` | M | derived indexes | stdout (untested ground) | the planner |
| Research | `cli catalog` | M | addon catalog | stdout (usable addons) | the planner |
| Plan | `cli freeze-input` | M | committed `experiment.json`, `protocol.md`, `decision.md` | frozen inputs under the artifact root, bound to HEAD | the submitter |
| Plan | `cli check-workflow` | M | frozen spec, RML policy | stdout (blockers) | the planner |
| Propose | `platforms/kaggle/check_quota.py` | M | Kaggle accounts | stdout (quota) | the submitter |
| Propose | `platforms/kaggle/submit_workflow.py` | M | frozen inputs, `pre_submit_review.json` | package, journal, prospective record | `retrieve_family_outputs.py` |
| Propose | `platforms/ims/v4_submit/submit_from_windows.py` | M | run spec, payload | IMS job, payload store | the same script with `--retrieve-to` |
| Observe | `platforms/ims/check_queue.py --collect` / `--web --snapshot` | M | scheduler and authorized V4 records / sanitized snapshot | atomic snapshot / read-only HTTP | [dashboard](../../platforms/ims/dashboard/README.md) |
| Accept | `platforms/kaggle/check_logs.py` | M | live kernel log | stdout (progress) | the watcher |
| Accept | `platforms/kaggle/retrieve_family_outputs.py` | M | receipt, remote outputs | retrieved arm files, `execution_root` | `accept_run.py` |
| Accept | `rml analyze` | M | saved predictions | analysis files | the decision |
| Accept | `platforms/local/accept_run.py` (`--diagnose`) | M | retrieved files | diagnosis, staged draft | the decision |
| Accept | `cli draft-closure` | M | retrieved files, role records | `closure_input.draft.json`, facts, role audit | the person who fills it |
| Accept | `platforms/local/accept_run.py` (`--closure-input`) | M | filled closure input | terminal records, derived RML | `rml check` |
| Archive | `rml check --frozen` | M | all RML records | stdout (pass or fail) | the committer |
| Archive | `rml doctor` | M | derived indexes | `completeness_report.json` | the maintainer |
| Archive | `cli import-closure` | M | experiment branch | evidence paths on the parent branch | the next planner |
| Archive | `rml terminal-rebuild` | M | trajectory | derived RML (recovery only) | `rml check` |

Stage 5 feeds stage 1: closure regenerates
[families.md](../../research_memory/derived/families.md), so the next round starts
from it. Closure also runs the vault refresh hook (see [VAULT](VAULT.md)).

## What a person decides

The commands never decide these. Each leaves a file that a later command checks
for shape, not for truth.

1. **Question and gate.** Write `experiment.json`, `protocol.md` and `decision.md`
   before `freeze-input`. Read the noise from `rml levers` and the family page;
   do not copy a gate from an older run.
2. **Pre-submit review.** A reviewer writes `pre_submit_review.json` (reviewer,
   verdict, reviewed commit, frozen source hash) after reading the frozen
   source. The hash covers the frozen `.py` files and `experiment.json`. The Kaggle
   submitter accepts the verdicts `APPROVE` and `APPROVE_WITH_CHANGES` only and checks
   the hash and commit, not the review's quality. This gate covers Kaggle only; the
   IMS submit script builds from a run template and does not read the record.
3. **Outcome and reason.** Choose each arm's outcome from the diagnosis and the
   gate, and write the decision file before `accept_run.py --closure-input`.
   Do not edit the decision after closure.
4. **Role states.** Give `--protected-role-use` only from observed facts.

## Not in this table

`rml rebuild` (rerun only after a standalone canonical edit; closure already
rebuilds), `cli prepare-resume` and the paused-run path (see the
[quickstart](EXPERIMENT_QUICKSTART.md#paused-stop_for_cost-runs)), and the SCNet
route ([skill](../../skills/scnet-bw-dcu-molgap/SKILL.md)).

Other `rml` commands (rare or called by the commands above): `validate`, `status`, `package-ready`, `backtest-screening`, `plan`, `finalize`, `terminal-pipeline`, `recover-trace`, `replay-pool`, `backtest-policy`.
Other `cli` commands (called by the workflow scripts; see the [CLI reference](EXPERIMENT_CLI.md)): `locate-run`, `draft-role-audit`, `diagnose`, `validate-spec`, `package`, `preflight`, `run-diagnostic`, `launch-receipt`, `terminal`, `plan-prospective`, `check-release`, `check-acceptance`, `inspect-output`, `accept-terminal`, `prepare-workflow`, `prepare-family-training`, `accept-workflow`, `build-terminal`, `prepare-release`, `check-preparation`, `workflow-info`, `build-resume`, `source-dependencies`, `sandbox`.
