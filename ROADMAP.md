# Roadmap - Priorities and Backlog

This file answers one question: **what should be done, and in what order?**
Live model/job state is in `CURRENT_STATE.md`; methods are in `docs/phaseN.md`;
metrics and decisions are under `results/`.

## Goal

Deliver a versioned property database of commercially available organic
molecules with near-GW gas-phase HOMO/LUMO/Gap predictions and trust signals.
The predictor is an implementation dependency; the database is the deliverable.

## Priority Queue

| Priority | ID | Task | Exit or trigger | Detail |
|---|---|---|---|---|
| P0 | P8.20-S | Complete retention-D seeds 43 and 44 and compare all three seeds | Seed 42 passed; require all seeds to improve in the same direction | `results/phase8/repaired_2m/retention_d_seed42_decision.md` |
| P1 | P8.20-G9 | Recompute GPS7/GPS9 complementarity on repaired-2M | Trigger only after the three-seed GPS7 gate | `results/phase8/repaired_2m/one_week_plan_20260723.md` |
| P1 | P9.2 | Recompute Delta labels against the selected B3LYP base | Start after the Phase 8 compression decision freezes the base | `docs/phase9.md` |
| P1 | P10-UQ | Refit calibration and OOD assets | Trigger after P9.2 selects the Delta path | `docs/phase10.md` |
| P2 | P9-AB | Benchmark descriptor LightGBM against encoder LoRA | Compare accuracy, calibration, throughput, and deployment size on one split | `results/phase9/v3_delta_decision.md` |

Do not tune against any sealed set. Do not relaunch a remote task merely because
its local output has not arrived.

## Phase Order

| Phase | Scope | Exit artifact |
|---|---|---|
| Phase 8 | Freeze the B3LYP base and bounded specialists | Selected deployable B3LYP path |
| Phase 9 | Delta learning toward GW | Validated Delta model against the frozen base |
| Phase 10 | Batch inference, calibration, OOD, and database build | Versioned near-GW prediction pipeline and database |
| Phase 11 | Delivery | Queryable release, reproducible build, and data card |

## Delivery Backlog

| ID | Task | Trigger |
|---|---|---|
| P10.2 | Batch SMILES -> B3LYP + Delta + UQ CSV | P9/P10 model bundle frozen |
| P10.3 | Element, MW, and topology applicability gates | Before database generation |
| P10.4 | Embedding/fingerprint OOD score | Before database generation |
| P10.5 | Layered real-capability sounding | Before public accuracy claims |
| P10.6 | Curate commercial-molecule universe | Inference contract frozen |
| P10.7 | Build versioned property database | P10.2-P10.6 complete |
| P11.1 | Package predictor and reproducible database build | P10 exit gate |
| P11.2 | Add queryable access | Versioned assets available |
| P11.3 | Publish provenance, schema, limitations, and data card | Release candidate ready |

## Conditional Backlog

| Task | Trigger |
|---|---|
| Experimental solid-state Delta head | A specific OLED experimental target is requested |
| Extend elements beyond CHONSFCl | Rejected-use analysis justifies refetch and retraining |
| Conformer ensemble for flagged rows | Residual analysis shows geometry/flexibility dominance |
| NNP geometry or conformer selection | The same geometry gate is met |
| SchNet denoising pretraining | Coverage and Delta work no longer dominate expected gain |
| Paper figures and write-up | An academic delivery is requested |

## Completed Work

- P8.20-D repaired-2M retention-D seed 42 passed the general-model gate:
  `results/phase8/repaired_2m/retention_d_seed42_decision.md`.
- Phase 1-7 history: `docs/phase1.md` through `docs/phase7.md`.
- Phase 8 decision timeline: `docs/phase8.md`.
- Closed Phase 8 code and results: `scripts/phase8/archive/README.md` and
  `results/phase8/archive/README.md`.
- Closed 3D encoder comparison: `results/ab3d/comparison.md`.

Use these records; do not recreate completed experiments to rediscover their
conclusions.
