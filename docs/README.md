# Docs

Dated narrative history and operational runbooks. Nothing here is live status or
the code map:

| Question | Read instead |
|---|---|
| What is true now? | `CURRENT_STATE.md` |
| What should be done next? | `ROADMAP.md` |
| Which file owns a behavior? | `ARCHITECTURE.md` |
| What did one experiment conclude? | that experiment's `decision.md`, indexed by `experiments/README.md` |
| What ships, in what order? | `production/README.md` |
| How was a phase-1-7 result obtained? | `production/history/phaseN.md` |

Contents:

| Path | Role |
|---|---|
| `operations/` | Platform runbooks (e.g. SCNet DCU environment) |
| `archive/` | Superseded narrative documents, kept for provenance |

Method documents now live beside the work they describe: the Delta-learning and
calibration contracts are `production/05_delta_gw/METHOD.md` and
`production/06_uq/METHOD.md`; the phase-8 decision timeline is
`experiments/_closed/PHASE8_TIMELINE.md`. Paths quoted inside archived documents
are historical — follow the current index instead.
