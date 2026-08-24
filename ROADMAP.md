# Roadmap - Priorities and Backlog

This file answers one question: **what should be done, and in what order?**
Live model/job state is in `CURRENT_STATE.md`; methods live in each experiment's
metrics and decisions are under `results/`.
Track ownership is defined only in `TRACKS.md`.

## Goal

Run an iterative architecture tournament against the frozen repaired-2M
production comparator without destabilizing the shipped Track A contract.
Structural GPS9 is the first positive baseline, not the end of the search. The
B3LYP general-model objective and official PCQM Gap objective remain separate;
an experimental encoder cannot enter production before fixed external
promotion.

## Short-Term Plan (from 2026-08-25)

The active research line is Track C. Persistent EdgeState Structural GPS passed
the controlled three-seed 100K screen and replaced plain RWSE16 Structural GPS9
as the sole repaired-2M scale-up candidate. Accept its complete immutable 2M
input, measure one epoch, train it once, and open the fixed external blocks once.
Conservative 2D+3D and PCQM transfer are conditional follow-ups. Track A
database work remains isolated. Router and MoE are not reopened by this
architecture search.

| Priority | ID | Task | Exit condition | Owner path |
|---|---|---|---|---|
| Complete | C-EDGE-STATE-GPS | Accept the independent seed-43/44 confirmations after seed 42 passed its GPS++-lite gate | All three seeds improved and the mean validation gain exceeded `0.001 eV`; EdgeState won | `experiments/resource_bounded_architecture/results/edge_state_100k_multiseed/decision.md` |
| P0 | C-FULL-2M | Package the accepted immutable repaired-2M EdgeState input, measure one epoch, then scale exactly this winner once | Do not submit before complete input acceptance; projected run stays below 10 hours; common/OOD/P8-hard opens once after training | `experiments/resource_bounded_architecture/` |
| P1 | C-CONSERVATIVE-3D | After the 2D candidate is frozen, test an exact-identity, low-gate, `0.03 eV` bounded 3D correction | Internal validation must select a non-identity head; then common/OOD/P8-hard must improve without a `>0.0005 eV` common regression | `src/molgap/hierarchical_fusion.py`, `experiments/resource_bounded_architecture/` |
| P2 | B-PCQM-STRUCTURAL | Train the winning lightweight architecture on official PCQM4Mv2 as a Gap-only model | Official train/validation protocol only; no PubChemQC labels or production claims | `experiments/resource_bounded_architecture/` |
| Closed | C-GAP-ONLY | Gap-only regressed against the accepted three-output Structural GPS9 Gap head in all three seeds | Decision and exact artifacts accepted; no scale-up | `experiments/resource_bounded_architecture/results/gap_rwse_100k_screen/decision.md` |
| Closed | C-NORMALIZED-RWSE | Normalized/gated RWSE beat Gap-only locally but remained worse than the accepted three-output model | Retain as component evidence; no standalone scale-up | `experiments/resource_bounded_architecture/results/gap_rwse_100k_screen/decision.md` |
| Closed | C-GATED-RWSE | Edge-aware residual GatedGCN improved seeds 42/44 and the mean but regressed on seed 43 | Strict direction-consistency gate failed; retain as diversity evidence, no 2M scale-up | `experiments/resource_bounded_architecture/results/gated_structural_100k_multiseed/decision.md` |

### Execution Schedule

| Window | Platform | Work | Hard stop |
|---|---|---|---|
| Complete | Kaggle P100 | EdgeState seeds 42/43/44 passed strict acceptance | Evidence: `experiments/resource_bounded_architecture/results/edge_state_100k_multiseed/decision.md` |
| Next | Local plus target accelerator | Package and accept the immutable repaired-2M input, then run one measured epoch | Do not train if the projected total exceeds 10 hours |
| After timing gate | SCNet BW-1 when allocated; Colab A100 fallback | Train the repaired-2M winner from random initialization with atomic per-epoch resume | One full run only; no ensemble or second architecture |
| After full training | Local evaluation | Run the one-time common/OOD/P8-hard acceptance against `repaired_2m_dense_2d` | Require common improvement of at least `0.001 eV`; OOD and P8-hard may not regress by more than `0.0005 eV` |
| Conditional | Colab A100 | Run the conservative 2D+3D internal gate after the 2D identity is frozen | Stop at the identity fallback unless internal validation selects a non-identity correction |
| Conditional | Kaggle or Colab | Transfer the lightweight winner to official PCQM Gap-only training | Keep Track B data, metrics, and registry isolated from Track A |

### Operating Rules

- Do not modify the production registry or public default while Track C is in
  screening. Promotion still requires the fixed Track A external gate.
- Train candidates from random initialization. Do not use pretrained weights,
  warm starts, distillation, or fine-tuning to claim an architecture gain.
- Do not rerun the rejected `0.10 eV` frozen-2D plus dual-SchNet residual. The
  P1 repair is a distinct exact-identity model with robust normalization,
  correction regularization, and an explicit pure-2D fallback. Any geometry
  input still preserves ETKDGv3+MMFF train-inference consistency.
- Use the accepted 100K scaffold split for selection. Common/OOD/P8-hard are a
  one-time full-candidate acceptance gate, not tuning data; sealed data stays locked.
- A remote run requires a local import/forward/backward check, a measured timing
  projection, an atomic resume contract, and independently retrievable outputs.
- Do not use the historical v3 Delta/UQ outputs as calibrated fields for the
  repaired-2M model. They remain historical evidence until refit and revalidated.
- Do not silently discard invalid, unsupported-element, out-of-range, or graph-
  failed molecules. Retain a reason code in the build output.
- Do not launch a full commercial-universe inference until the 1K dry run and
  10K pilot pass their manifests and acceptance checks.
- Do not tune against sealed data or promote Track B into the production
  registry without a separate deployment decision.

## Workstream Order

| Stage | Scope | Exit artifact |
|---|---|---|
| Frozen production | Keep the accepted B3LYP base and specialists reproducible | `repaired_2m_dense_2d` remains loadable and externally evidenced |
| Architecture refresh | 100K controlled screen, then at most one repaired-2M winner | Standalone candidate accepted or question closed with a decision record |
| Database delivery | B3LYP batch inference, applicability, OOD screening, and database build | Versioned B3LYP prediction pipeline and database |
| Optional extensions | Delta/UQ refit or leaderboard submission | Separate approved objective and validation contract |

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

- The paired PubChemQC 100K screen accepted RWSE16 Structural GPS9 over the
  same-run GPS9-192 control for all three seeds and retained exact checkpoints,
  predictions, hashes, and timing evidence:
  `experiments/resource_bounded_architecture/decision.md`.
- The 2026-08-24 stage transition separated the frozen Track A production
  identity from the reopened, resource-bounded Track C architecture question.
  The live state, task order, track ownership, experiment index, and dated
  feasibility decision now point to one another without reopening closed work:
  `experiments/resource_bounded_architecture/decision.md`.
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
