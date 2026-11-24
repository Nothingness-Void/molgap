# Docs

Operational runbooks and small explanatory pointers. Phase 1-7 narrative and
superseded records live on the `archive` branch, so this directory stays small.
Nothing here is live status or the code map:

| Question | Read instead |
|---|---|
| What is true now? | `CURRENT_STATE.md` |
| What should be done next? | `ROADMAP.md` |
| Which file owns a behavior? | `ARCHITECTURE.md` |
| What did one experiment conclude? | that experiment's `decision.md`, indexed by `experiments/README.md` |
| What ships, in what order? | `production/README.md` |
| Which command comes next, and who reads its output? | `docs/operations/LIFECYCLE_MAP.md` |
| What does the Obsidian vault hold? | `docs/operations/VAULT.md` (a view, not an authority) |
| Which skill runs a platform? | `skills/README.md` |
| How was a phase 1-7 result obtained? | `archive` branch: `production/history/phaseN.md` |

Method documents live beside the work they describe: the Delta-learning and
calibration contracts are `production/05_delta_gw/METHOD.md` and
`production/06_uq/METHOD.md`; the phase 8 timeline is
`experiments/_closed/PHASE8_TIMELINE.md`.

## One owner per fact

Each fact below lives on one page only; a test fails if another page repeats it.

| Fact | Owner |
|---|---|
| Infrastructure budget command and limits | `AGENTS.md` |
| Tail holdout size and rule | `AGENTS.md` |
| Kaggle credential directory | `skills/kaggle-molgap-workloads/references/account-and-cli.md` |

## Archived pages

Old audit, handoff and acceptance pages (dated, or about one finished
migration) were removed from the tree. They are in git tag
`docs-archive-20261124`. Read one with
`git show docs-archive-20261124:docs/operations/<NAME>.md`. The old pages are also
in git history (commits before 982aaac61), so a clone without the tag still has them. Paths quoted inside
experiment records may point to them; those records stay unchanged.
