# Roadmap - Priorities and Backlog

This file answers one question: **what should be done, and in what order?**
Live model/job state is in `CURRENT_STATE.md`; methods live in each experiment's
metrics and decisions are under `results/`.
Track ownership is defined only in `TRACKS.md`.

## Goal

Deliver a versioned property database of commercially available organic
molecules with near-GW gas-phase HOMO/LUMO/Gap predictions and trust signals.
The predictor is an implementation dependency; the database is the deliverable.

## Priority Queue

| Priority | ID | Task | Exit or trigger | Detail |
|---|---|---|---|---|
| P0 | FREEZE | Freeze the project by 2026-08-04 12:00 JST | Stop unfinished compute, archive atomic partials, and permit only packaging or correctness fixes | `production/04_evaluate/project_freeze/README.md` |
| P0 | PACKAGE-A | Package the frozen Track A pure-2D models | Retrieve and hash all selected checkpoints; add a tested inference loader, latency record, and valid/invalid/OOD smoke test | `production/04_evaluate/project_freeze/track_a_final_decision.md` |
| P0 | PACKAGE-B | Package the frozen Track B PCQM specialist | Preserve the seven accepted checkpoints; add reproducible inference and cost records | `production/04_evaluate/project_freeze/track_b_final_decision.md` |
| P0 | REPORT | Build the final comparison and presentation package | Normalize common/OOD/P8-hard/PCQM metrics, cost, limitations, and architecture figures | `production/04_evaluate/project_freeze/README.md` |

Do not tune against any sealed set. Do not relaunch a remote task merely because
its local output has not arrived.

The scientific decisions are frozen. No new acquisition,
architecture family, Router, MoE, OOF, compression, or scale-up experiment may
start before the presentation.

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
| OGB-compliant PCQM4Mv2 submission retrain | Reopen after the project freeze and presentation work; train every encoder from scratch on official PCQM4Mv2 data only, then evaluate full validation and test-dev under the four-hour inference rule |
| GPS7/GPS9 OOF gain labels | A molecule-level Router is explicitly reopened; this is not required by fixed-identity bounded fusion |
| Experimental solid-state Delta head | A specific OLED experimental target is requested |
| Extend elements beyond CHONSFCl | Rejected-use analysis justifies refetch and retraining |
| Conformer ensemble for flagged rows | Residual analysis shows geometry/flexibility dominance |
| NNP geometry or conformer selection | The same geometry gate is met |
| SchNet denoising pretraining | Coverage and Delta work no longer dominate expected gain |
| Paper figures and write-up | An academic delivery is requested |

## Completed Work

- Track A froze the repaired-2M three-GPS dense pure-2D model as its accuracy
  identity and the GPS7/GPS9 equal model as its lower-cost preset. The
  dual-SchNet residual was rejected:
  `production/04_evaluate/project_freeze/track_a_final_decision.md`.
- Track B froze the four-encoder, three-seed bounded PCQM Gap Fusion at
  `0.112011 eV` on the fixed official-validation subset:
  `production/04_evaluate/project_freeze/track_b_final_decision.md`.

- The local PCQM GINE expert was scaled to a nested 1M sample and accepted as a
  task-only leaderboard candidate:
  `experiments/pcqm_gine_expert/results/local_scaleup_1m_v7_decision.md`.
- P8.20-D repaired-2M retention-D passed its three-seed general-model gate:
  `experiments/repaired_2m_scaling/results/retention_d_multiseed_decision.md`.
- P8-QM9 eliminated weak architectures and promoted three candidates to the
  PubChemQC 100K transfer gate:
  `experiments/qm9_architecture/README.md`.
- The Track C PubChemQC 100K transfer gate accepted GPS11-160 identity plus a
  `+-0.10 eV` bounded correction from GPS9 and two lightweight SchNet branches:
  `experiments/pubchemqc100k_architecture/results/experiment_manifest.json`.
- Phase 1-7 history: `production/history/` (`phase1.md` through `phase7.md`).
- Per-question decision records: `experiments/README.md`.
- Closed code and evidence: `experiments/_closed/README.md` and
  `experiments/_closed/SCRIPTS_ARCHIVE.md`.
- Closed 3D encoder comparison: `experiments/_closed/ab3d/comparison.md`.

Use these records; do not recreate completed experiments to rediscover their
conclusions.
