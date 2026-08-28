# Roadmap - Priorities and Backlog

This file owns only task order, triggers, and exit conditions. Live model and
job state is in `CURRENT_STATE.md`; completed methods, metrics, and conclusions
belong to each experiment's decision record. Track ownership is defined in
`TRACKS.md`.

## Goal

Repair the failed strict official-only PCQM4Mv2 EdgeState input contract
without changing the frozen repaired-2M production model. The leaderboard
specialist remains an isolated Track B asset.

## Active Queue

| Priority | ID | Task | Exit condition | Owner |
|---|---|---|---|---|
| P0 | B-PCQM-RICH-FEATURE-SCREEN | On official training rows only, compare the failed v1 atom/bond features against OGB's complete categorical atom/bond contract with EdgeState and Gap supervision fixed | Passed on seeds 42/43/44; evidence is owned by `experiments/pcqm_edge_state_full/results/feature_screen/` | `experiments/pcqm_edge_state_full/` |
| P1 | B-PCQM-SCHEDULE-SCREEN | On the same accepted OGB graphs and seeds 42/43/44, compare ten-epoch cosine collapse with twenty epochs using two-epoch warmup plus cosine decay | Passed on all three seeds; evidence is owned by `experiments/pcqm_edge_state_full/results/schedule_screen/decision.md` | `experiments/pcqm_edge_state_full/` |
| P1 | B-PCQM-PAIRBIAS-SCREEN | Test one fixed OGB-rich PairGPS8-192 candidate with pair-biased attention on the accepted official-train-only 100K/10K split | Seed 42 improves overall by at least `0.002 eV` without more than `0.002 eV` subset regression; only then confirm seeds 43/44 and compare against the same-cost three-control ensemble | `experiments/pcqm_edge_state_full/pairbias_screen/` |
| P0 | B-PCQM-OFFICIAL-SUBMIT | Public code, report, checkpoint Release, both OGB NPZ files, and the OGB-LSC form submission are complete | Await OGB code/report validity review; do not resubmit or claim a score before acceptance | `experiments/pcqm_edge_state_full/results/rich_full/submission_status.md` |
| P0 | C-FULL-2M-INPUT | Materialize and accept the immutable repaired-2M EdgeState input; run one measured epoch | Identity, counts, finite values, hashes, and resume paths pass; projected training is at most 10 hours | `experiments/resource_bounded_architecture/` |
| P0 | C-FULL-2M-TRAIN | Train exactly one EdgeState candidate from random initialization | P0 input gate passes; one complete resumable checkpoint and aligned predictions are accepted | `experiments/resource_bounded_architecture/` |
| P1 | C-FULL-2M-EVAL | Compare the frozen candidate once on common/OOD/P8-hard | Common improves by at least `0.001 eV`; OOD and P8-hard do not regress by more than `0.0005 eV` | `experiments/resource_bounded_architecture/` |
| P2 | C-CONSERVATIVE-3D | Test the exact-identity, low-gate, `0.03 eV` bounded 3D correction after the 2D identity freezes | Internal validation selects a non-identity head before any external block opens | `experiments/resource_bounded_architecture/`, `platforms/colab/conservative_2d3d_fusion/` |
| P3 | B-PCQM-STRUCTURAL | Retained pointer to the strict-official repair tasks above | The isolated Track B decision is recorded without changing Track A | `experiments/pcqm_edge_state_full/` |

The accepted 100K EdgeState screen is evidence, not an active task. Its exact
decision is
`experiments/resource_bounded_architecture/results/edge_state_100k_multiseed/decision.md`.

The completed full rich-feature validation, clean-clone audit, and official form
submission are recorded under
`experiments/pcqm_edge_state_full/results/rich_full/`. OGB review is pending;
no leaderboard score exists.

## Mandatory Gates

1. **Before remote submission:** local import, forward/backward, immutable input
   acceptance, measured timing projection, atomic checkpointing, and durable
   output chunks must pass.
2. **Before external evaluation:** the standalone full-scale candidate and its
   aligned predictions must be complete and frozen.
3. **Before production promotion:** the fixed Track A external gate must pass;
   then public loader, registry, hashes, latency, and smoke tests are updated in
   one production decision.
4. **On failure:** write a dated decision beside the experiment evidence and
   close the branch. Do not move the failure into `CURRENT_STATE.md`.

## Operating Rules

- Do not modify the production registry while Track C is screening.
- Architecture claims use random initialization; no pretraining, warm start,
  fine-tuning, or distillation may be credited as an architecture gain.
- Do not tune on common/OOD/P8-hard or sealed data.
- New pure-2D architecture questions use the fixed sequence QM9, matched
  PubChemQC-100K validation, one frozen intermediate test, then at most one
  authorized full-data run.
- Do not rerun the rejected `0.10 eV` frozen-2D plus dual-SchNet residual.
- Geometry paths must preserve ETKDGv3+MMFF train-inference consistency.
- Router, MoE, OOF gain labels, and dataset replacement remain closed unless a
  new question and stop rule are added here first.
- Invalid molecules remain visible with reason codes; do not silently filter.
- Historical v3 Delta/UQ outputs must not be described as calibrated for the
  repaired-2M production model.

## Delivery Queue

These Track A delivery tasks remain valid but do not override the active Track C
experiment unless the project objective changes.

| ID | Task | Trigger |
|---|---|---|
| P10.2 | Batch SMILES to B3LYP CSV with provenance and rejection reasons | Track A model bundle remains frozen |
| P10.3 | Element, molecular-weight, and topology applicability gates | Before database generation |
| P10.4 | Reproducible disagreement-based OOD screening signal | Before database generation; never label it calibrated UQ |
| P10.5 | Layered real-capability sounding | Before public accuracy claims |
| P10.6 | Curate the commercial-molecule universe | 10K pilot and inference contract pass |
| P10.7 | Build the versioned B3LYP property database | P10.2-P10.6 complete |
| P11.1-P11.3 | Package, expose, and document the database | P10 exit gate passes |

## Conditional Queue

| Task | Trigger |
|---|---|
| PairGPS2D sealed-test disposition | Explicit authorization to reopen the independent branch after its validation-only decision; arithmetic equivalence must be established before using benchmark-selected TF32 for an accuracy claim |
| OGB-compliant PCQM4Mv2 submission retrain | A separate leaderboard objective and compute budget are approved |
| Experimental solid-state Delta head | A specific experimental target is requested |
| Extend the supported element set | Rejected-use analysis justifies refetch and retraining |
| Conformer ensemble or NNP geometry | Residual evidence identifies geometry as the limiting factor |
| Paper figures and write-up | An academic delivery is requested |

Completed work is indexed, without duplicated metrics, in
`experiments/README.md`, `experiments/_closed/README.md`, and
`production/README.md`.
