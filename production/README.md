# Production Line

The delivery path, in data-flow order. A stage owns the artifacts it produces and
the thin CLIs that produce them.

Only work that feeds the shipped predictor or database belongs here. A question
being investigated belongs in `experiments/`. Which model is recommended right
now is in `CURRENT_STATE.md`, never here.

This is the delivery line for **Track A - Production**. Track definitions live
only in `TRACKS.md`.

| Stage | Question it answers | Contents |
|---|---|---|
| `01_acquire/` | Which molecules do we train on? | Sampling specs, fetch and feature-selection CLIs |
| `02_graphs/` | How are they represented? | 2D bond and 3D ETKDG caches, fixed splits, build CLIs |
| `03_train/` | What are the encoder and fusion weights? | The registered routed-v4 compatibility record and training CLIs |
| `04_evaluate/` | Which candidate ships? | Current acceptance, freeze, latency, model inventory, and API checks |
| `05_delta_gw/` | Can B3LYP be lifted toward GW? | Reusable Delta interfaces; historical outputs are on the archive branch |
| `06_uq/` | How much do we trust a row? | Reusable calibration/OOD interfaces; historical outputs are on the archive branch |
| `07_database/` | The deliverable. | Predicted-property database build |

Phase 1-7 history, retired v2/v3 production records, and historical Delta/UQ
outputs are deliberately absent from this tree. They are preserved on the
`archive` branch. Large ignored legacy runtime assets are under
`models/archive/`, `data/raw/archive/`, and `data/cache/archive/`.

Stage directories are numbered because the order is a dependency, not a calendar.
Renaming a stage means editing one constant in `src/molgap/constants.py`.

Directory and artifact names follow `NAMING.md`.

Reusable logic lives in `src/molgap/`; the `scripts/` folder inside each stage
only parses arguments and persists outputs. See `ARCHITECTURE.md`.
