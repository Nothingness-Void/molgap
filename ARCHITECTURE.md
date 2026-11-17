# Architecture

Repository and artifact names follow `NAMING.md`. This file owns placement and
code boundaries; it does not duplicate naming rules.

This file answers one question: **to change behavior X, which code owns it?**
Model recommendations and job status belong in `CURRENT_STATE.md`.
Track A/B/C ownership belongs in `TRACKS.md`.

## Boundary

- Reusable behavior lives in `src/molgap/`.
- `scripts/` parse arguments, call package code, and persist outputs.
- `results/` contains evidence and never supplies runtime logic.
- `models/` contains assets; registration is explicit in `constants.py`.
- Archived code is reproducibility evidence and is not imported by supported
  paths.

## Package Map

Master holds only the modules that the public inference API imports. Every
other module moved to the `archive` branch (see `PROMOTION_20261117.md`).

| Module | Owns | Edit when |
|---|---|---|
| `constants.py` | Repository paths, hyperparameters, model registry | Adding or retargeting an explicit registry entry |
| `compat/legacy_constants.py` | Old phase constants kept for compatibility | Keeping an old import working |
| `graphs.py` | SMILES-to-2D/3D PyG graphs and ETKDG construction | Changing graph or conformer representation |
| `gps.py` | `GPSWrapper` and 2D encoding | Changing the 2D encoder |
| `schnet.py` | `SchNetWrapper` and 3D encoding | Changing the PyG SchNet encoder |
| `tensornet.py` | `TensorNetWrapper` 3D encoder | Changing the TensorNet encoder |
| `fusion.py` | `FusionHead` | Changing embedding-level fusion |
| `multi2d_router_fusion.py` | Frozen-GPS dense gates | Changing GPS7/GPS9/GPS11 prediction routing |
| `inference.py` | Model loading and batch prediction | Changing prediction behavior |
| `utils.py` | Shared splits, metrics, SMILES, fingerprints, and IO | Changing cross-cutting utilities |
| `__init__.py` | Lazy package-level public exports | Changing the public import surface |

## Loading Structure

- `load_repaired_2m_2d(key=...)` loads a pure-2D multi-expert registry entry:
  several direct-output GPS predictors plus either a dense soft gate
  (`kind: multi2d_dense`) or a fixed equal average (`kind: multi2d_equal`). It
  has no 3D or fusion component.
- `load_hybrid(key=...)` loads one registry-defined 2D + 3D + fusion trio.
- `load_routed_dual_gps_hybrid(key=...)` loads a routed hybrid registry entry.
- `predict_smiles_batch_repaired_2m_2d()`, `predict_smiles_batch_hybrid()`, and
  `predict_smiles_batch_routed_dual_gps()` are the corresponding batch paths.
- The registry key recommended for use is intentionally not repeated here; read
  `CURRENT_STATE.md`. A loader default is a component/compatibility choice and
  does not imply the recommended predictor.
- Registry entries flagged `artifact_retained: False` are provenance only; their
  checkpoints are absent and loading them fails.
- Registry structure and exact asset paths are authoritative in
  `src/molgap/constants.py`.

## Tree Map

Three top-level trees, split by role rather than by calendar phase. A phase
number ages; a role does not.

| Tree | Answers | Entry point |
|---|---|---|
| `production/` | What ships, in data-flow order | `production/README.md` |
| `experiments/` | One directory per open or closed question | `experiments/README.md` |
| `platforms/` | How a run reaches a given compute environment | `platforms/README.md` |

Each production stage keeps its own `scripts/` for argument parsing and output
persistence; the reusable behavior stays in `src/molgap/`. Stage roots are
constants (`ACQUIRE_DIR` through `DATABASE_DIR`) so renaming a stage is a
one-line change in `constants.py`.

Every active CLI must resolve paths from those constants rather than from
`Path(__file__).parents[n]`, because a depth-derived root breaks the moment a
file moves.

`production/history/` holds the frozen phase 1-7 line. It is reproducibility
evidence and is not extended.

## Asset Map

| Path | Role |
|---|---|
| `data/raw/` | Source tables and downloaded raw inputs |
| `data/cache/` | Regenerable local graph/embedding caches |
| `models/README.md` | Checkpoint asset map |

Experiment method and conclusions live in each experiment's own decision record;
follow `experiments/README.md` rather than restating them here.

## Removed Code

The 2026-11-17 slim removed the older model modules (`hybrid.py`,
`late_router.py`, `multi2d_data.py`, `schnetpack.py`, `visnet.py`, `archive/`),
the K1 and GPT-T research library, the experiment tooling, and the scripts and
tests that need them. `PROMOTION_20261117.md` lists where to get them back.
