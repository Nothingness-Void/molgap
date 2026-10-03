# Architecture

Repository and artifact names follow `NAMING.md`. This file answers one
question: **to change behavior X, which owner do I edit?** Model recommendations
and job status belong in `CURRENT_STATE.md`; Track A/B/C ownership belongs in
`TRACKS.md`. The detailed module table is the conditional-read
[module index](docs/operations/ARCHITECTURE_MODULE_INDEX.md).

## Boundary

- Reusable behavior lives in `src/molgap/`.
- `scripts/` parse arguments, call package code, and persist outputs.
- `results/` contains evidence and never supplies runtime logic.
- `models/` contains assets; registration is explicit in `constants.py`.
- Archived code is reproducibility evidence and is not imported by supported paths.

## Owner map

| Change | Owner | Route |
|---|---|---|
| Model or graph representation | The owning family model and graph primitive | Read the family section of the [module index](docs/operations/ARCHITECTURE_MODULE_INDEX.md) |
| Existing-family addon | `experiment_spec.py` + `experiment_execution.py` + the owning family hook | [Addon guide](docs/operations/EXPERIMENT_ADDON_GUIDE.md) |
| New pure-2D graph family | `experiment_execution.py: graph_training_adapter`, `shared_model_adapter.py`, `graph_screen_training.py`, `pcqm_graph_inputs.py` | [registered workflow](docs/operations/REGISTERED_EXPERIMENT_WORKFLOW.md) and [quickstart](docs/operations/EXPERIMENT_QUICKSTART.md) |
| Package, preflight, launch, recovery, or retention | The shared `experiment_*` lifecycle modules | [lifecycle module index](docs/operations/ARCHITECTURE_MODULE_INDEX.md#lifecycle-evidence-and-platform-owners) |
| Platform submission or retrieval | The selected platform adapter and workload skill | [platforms/README.md](platforms/README.md) |
| Scientific evidence or RML closure | Family acceptance and `research_memory/` | [experiment evidence contract](experiments/README.md#evidence-contract) |
| Public inference or registry | `inference.py`, `__init__.py`, and `constants.py` | [production/README.md](production/README.md) |
| Documentation pointers or entry size | `documentation_check.py` | [verification commands](docs/operations/INFRASTRUCTURE_VERIFICATION.md) |

## Registered family lifecycle

`ExperimentSpec` declares immutable family/addon identity and scientific fields.
`TrainingAdapter` selects the owning trainer and typed addon mode. The generic
graph path uses `graph_training_adapter(family, model_factory=..., addons=...,
source_files=...)`; a family model factory returns a `torch.nn.Module` whose
`forward(batch)` produces a normalized Gap vector, and an addon hook applies a
small model delta. `graph_screen_training.py` owns the reusable pure-2D trainer;
`pcqm_graph_inputs.py` owns its input ABI. Legacy family trainers retain their
declared input semantics.

The lifecycle then reuses prospective planning, explicit source packaging,
release/preflight checks, strict launch binding, isolated workers, all-arm
preflight, immutable inspection, checkpoint transport, allocation retention,
terminal translation, and RML closure. Platform adapters own submission,
reconciliation, and retrieval. The shared CLI never submits a job.

For a same-family addon, add only its model delta, typed `AddonContract`, one
`TrainingAddon` descriptor, reviewed source dependency, and focused tests. For
a new graph family, add the family contract, `make_model(arm)`, static adapter
entry, and focused factory/training/output checks; reuse the shared PCQM graph
input owner when its contract applies. Do not copy lifecycle or platform
code. A declaration or synthetic check does not qualify real shards, GPU
runtime, scientific acceptance, or a non-PCQM/non-pure-2D route.

## Loading structure

- `load_repaired_2m_2d(key=...)` loads a registry-defined pure-2D multi-expert.
- `load_hybrid(key=...)` loads a registry-defined 2D + 3D + fusion trio.
- `load_routed_dual_gps_hybrid(key=...)` loads a routed registry entry.
- Corresponding batch paths are `predict_smiles_batch_repaired_2m_2d`,
  `predict_smiles_batch_hybrid`, and `predict_smiles_batch_routed_dual_gps`.
- Read `CURRENT_STATE.md` for the recommended registry key. Loader defaults are
  compatibility choices and do not imply recommendation.
- `artifact_retained: False` entries are provenance only and cannot load.

## Tree map

Three top-level trees are split by role, not calendar phase:

| Tree | Answers | Entry point |
|---|---|---|
| `production/` | What ships, in data-flow order | `production/README.md` |
| `experiments/` | One directory per open or closed question | `experiments/README.md` |
| `platforms/` | How a run reaches a compute environment | `platforms/README.md` |
| `research_memory/` | Evidence-linked trajectories, reuse, cost, and completeness | `research_memory/README.md` |

Production stage scripts resolve roots from `constants.py`; they do not derive
roots from `Path(__file__).parents[n]`. `tests/test_repository_layout.py` checks
this, CLI aliases, help paths, active experiment pointers, and bounded control
documents. `production/history/` is frozen reproducibility evidence.

## Asset map

| Path | Role |
|---|---|
| `data/raw/` | Source tables and downloaded raw inputs |
| `data/cache/` | Regenerable local graph/embedding caches |
| `models/README.md` | Checkpoint asset map |

Experiment method and conclusions live in each experiment decision record; use
the [evidence index](experiments/EVIDENCE_INDEX.md) to locate a question without
copying its metrics into this document.

## Authority pointers

The [registered workflow](docs/operations/REGISTERED_EXPERIMENT_WORKFLOW.md)
owns the static family/addon registry, preparation, acceptance, and recovery
contracts. The [CLI reference](docs/operations/EXPERIMENT_CLI.md) owns command
arguments and result meanings. The [quickstart](docs/operations/EXPERIMENT_QUICKSTART.md)
is the new-agent route; this file only maps code ownership.
