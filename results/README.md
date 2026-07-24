# Result Asset Map

`results/` stores immutable metrics, predictions, manifests, and decision
records. It is evidence, not the source of current status; start with
`CURRENT_STATE.md`.

| Location | Role |
|---|---|
| `phase1/` to `phase10/` | Per-phase experiment evidence |
| `phase8/README.md` | Phase 8 evidence map and archive boundary |
| `kaggle/README.md` | Durable remote acquisition/evaluation handoffs by lifecycle |
| `scnet/` | SCNet outputs and logs kept local |
| `ab3d/` | Closed 3D encoder comparison |

Large predictions, caches, logs, model payloads, and remote source snapshots are
ignored by Git. Keep compact JSON/Markdown summaries beside them so humans and
agents can review a decision without loading the raw artifacts.
