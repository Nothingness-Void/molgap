# Kaggle Result Map

This directory is the local durable handoff for Kaggle work. Its top level is
organized by lifecycle so a new reader does not need to inspect run names.

| Path | Contents |
|---|---|
| `acquisition/completed/` | Reconciled durable chunks and accepted inventories |
| `acquisition/launches/` | Exact payload/config snapshots for submissions |
| `acquisition/datasets/` | Downloaded broad, hard, and complementary candidate datasets |
| `evaluation/runs/` | Completed 1M external, PCQM, and replay-fusion evaluations |
| `evaluation/datasets/` | Model/embedding datasets used by fixed evaluation runs |
| `evaluation/reference/` | Immutable PCQM reference samples |
| `archive/audits/` | Asset audits and status probes |
| `archive/checks/` | Preflight and candidate-fetch checks |
| `archive/acquisition_attempts/` | Failed, partial, or superseded residual-fetch runs |
| `archive/organization/` | Account lifecycle and naming records |
| `archive/runtime/` | Downloaded runtime-source snapshot |

Large CSV, ZIP, log, model, and runtime-source payloads are ignored by Git.
Compact manifests, validation summaries, and decision Markdown are the review
surface.

The reusable source payload is
`scripts/phase8/remote/kaggle/acquisition/molgap_2m_candidate_fetch/`. Do not
edit a historical launch snapshot to start a new run; create a new bounded
launch directory and preserve its resume inputs.

See `archive/organization/organize_20260721/README.md` for the detailed
ACTIVE/EVAL/ARCHIVE account lifecycle.
