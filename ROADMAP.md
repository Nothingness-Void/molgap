# Roadmap - Priorities and Backlog

This file answers one question: **what should be done, and in what order?**
Live model/job state is in `CURRENT_STATE.md`; methods live in each experiment's
metrics and decisions are under `results/`.
Track ownership is defined only in `TRACKS.md`.

## Goal

Deliver a versioned property database of commercially available organic
molecules with auditable B3LYP gas-phase HOMO/LUMO/Gap predictions and trust
signals. A near-GW Delta extension is optional follow-up work, not a database
MVP dependency. The predictor is an implementation dependency; the database is
the deliverable.

## Short-Term Plan (2026-08-22 to 2026-09-05)

The presentation is complete. The next deliverable is a small, reproducible
B3LYP-only database pilot. It must use the accepted Track A model and must not
wait for Delta/UQ refits or Track B leaderboard work.

| Priority | ID | Task | Exit condition | Owner path |
|---|---|---|---|---|
| P0 | DOCS-POST-PRESENTATION | Replace freeze-era live status with the delivery plan | `CURRENT_STATE.md`, this file, and the database README agree | `CURRENT_STATE.md`, `ROADMAP.md`, `production/07_database/README.md` |
| P0 | A-CONTRACT | Define the B3LYP-only row contract and provenance fields | Schema, model key, split/version, applicability fields, and SHA256 manifest are written before inference | `production/07_database/README.md` |
| P0 | A-BATCH-MVP | Implement reusable batch inference with a thin production CLI | 1K dry run produces atomic output, rejected-row reasons, manifest, and finite-value checks | `src/molgap/`, `production/07_database/` |
| P1 | A-APPLICABILITY | Report structural applicability without silent filtering | Every row records valid parse, allowed elements, MW range, graph success, and `in_domain` | `production/04_evaluate/`, `production/07_database/` |
| P1 | A-OOD-SIGNAL | Add a clearly labeled screening signal for model disagreement | Per-expert disagreement is reproducible and validated on existing common/OOD/P8-hard evidence; do not call it calibrated UQ | `src/molgap/inference.py`, `production/07_database/` |
| P1 | A-PILOT-CATALOG | Run and accept a 10K pilot before scaling | Counts, deduplication, finite predictions, coverage, manifest, and checksum all reconcile | `production/07_database/` |
| P1 | B-PACKAGE | Preserve a reproducible Track B Gap-specialist package if cheap | Checkpoint custody, inference smoke test, and cost record; never blocks A | `experiments/pcqm_route_b/`, `production/04_evaluate/` |
| P2 | DELTA-REFIT | Refit Delta only against the repaired-2M base | A concrete downstream use case and time budget exist after the database pilot | `production/05_delta_gw/` |

### Operating Rules

- Do not start new architecture, Router, MoE, OOF, compression, conformer, or
  full-scale 3D training experiments. Track C is closed.
- Do not use the historical v3 Delta/UQ outputs as calibrated fields for the
  repaired-2M model. They remain historical evidence until refit and revalidated.
- Do not silently discard invalid, unsupported-element, out-of-range, or graph-
  failed molecules. Retain a reason code in the build output.
- Do not launch a full commercial-universe inference until the 1K dry run and
  10K pilot pass their manifests and acceptance checks.
- Do not tune against sealed data or promote Track B into the production
  registry without a separate deployment decision.

## Phase Order

| Phase | Scope | Exit artifact |
|---|---|---|
| Phase 8 | Freeze the B3LYP base and bounded specialists | Selected deployable B3LYP path |
| Phase 9 | Optional Delta learning toward GW | Validated Delta model against the repaired-2M base; deferred, not a delivery gate |
| Phase 10 | B3LYP batch inference, applicability, OOD screening, and database build | Versioned B3LYP prediction pipeline and database |
| Phase 11 | Delivery | Queryable release, reproducible build, and data card |

## Delivery Backlog

| ID | Task | Trigger |
|---|---|---|
| P10.2 | Batch SMILES -> B3LYP CSV with provenance and rejection reasons | Track A model bundle frozen |
| P10.3 | Element, MW, and topology applicability gates | Before database generation |
| P10.4 | Reproducible OOD screening signal | Before database generation; disagreement is a screening signal, not calibrated UQ |
| P10.5 | Layered real-capability sounding | Before public accuracy claims |
| P10.6 | Curate commercial-molecule universe | 10K pilot accepted and inference contract frozen |
| P10.7 | Build versioned B3LYP property database | P10.2-P10.6 complete |
| P11.1 | Package predictor and reproducible database build | P10 exit gate |
| P11.2 | Add queryable access | Versioned assets available |
| P11.3 | Publish provenance, schema, limitations, and data card | Release candidate ready |

## Conditional Backlog

| Task | Trigger |
|---|---|
| OGB-compliant PCQM4Mv2 submission retrain | A concrete leaderboard objective and remote budget are approved; train every encoder from scratch on official PCQM4Mv2 data only |
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
- PACKAGE-A closed on 2026-07-31. Both presets are registered, loadable through
  `molgap.inference`, reproduce their accepted external metrics within
  `1e-4 eV`, and have latency, encoder-pass, and valid/invalid/OOD smoke-test
  records under `production/04_evaluate/project_freeze/`. Delta/UQ was not
  refitted and stays calibrated to its v3 base.
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
