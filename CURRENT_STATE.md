# Current State

> This file owns only live project truth: production identity, active
> candidates, blockers, and immediate handoff. Metrics and history belong to
> experiment decisions; ordering belongs to `ROADMAP.md`.

## Production Identity

- **Recommended model:** repaired-2M three-GPS dense pure 2D.
- **Registry key:** `repaired_2m_dense_2d`.
- **Lower-cost preset:** `repaired_2m_equal_2d`.
- **Public loader:** `load_repaired_2m_2d` and
  `predict_smiles_batch_repaired_2m_2d` in `src/molgap/inference.py`.
- **Decision owner:**
  `production/04_evaluate/project_freeze/track_a_final_decision.md`.

The routed-v4 compatibility model remains registered. Historical Delta and UQ
bundles remain calibrated to their historical v3 base. Asset ownership is in
`models/README.md` and the production decision above.

## Active Objectives

### Track B: official PCQM4Mv2

The OGB-rich EdgeState Structural GPS completed strict official-only training
from random initialization. Epoch 19 reached official-validation Gap MAE
`0.102063 eV`; the frozen checkpoint then produced both official test NPZ files
from raw SMILES in `147.046 s`. The files passed OGB shape, dtype, order,
finiteness, timing, and hash acceptance.

The public reproduction repository and checkpoint Release are
`https://github.com/Nothingness-Void/pcqm4mv2-edgestate`. Final public head
`ae00b44` passed a fresh-clone audit. The OGB-LSC form was submitted on
2026-08-28 and is awaiting code/report validity review. No test score or rank
exists yet. Evidence is under
`experiments/pcqm_edge_state_full/results/rich_full/`.

This specialist cannot change Track A production.

### Track C: repaired-2M EdgeState

Persistent EdgeState Structural GPS passed the controlled three-seed 100K
screen and is the sole repaired-2M scale-up candidate. No complete 2M model has
been trained. Submission remains blocked until the immutable input passes
acceptance and a measured epoch projects the bounded run below 10 hours.

The decision is
`experiments/resource_bounded_architecture/results/edge_state_100k_multiseed/decision.md`;
the execution contract is
`experiments/resource_bounded_architecture/STATUS.md`.

## Parallel Workstreams

- The official-only Kaggle architecture search is closed. Capacity, generic
  field concatenation, and radical-context variants did not beat the matched
  same-cost control ensemble. See
  `experiments/pcqm_edge_state_full/architecture_search/`.
- The corrected FP32 PairBias seed-42 screen is the only bounded PCQM
  architecture question still permitted. Seeds 43/44 require its predetermined
  gate. See `experiments/pcqm_edge_state_full/pairbias_screen/`.
- The independent PairGPS2D validation-only screen passed, but its test role
  remains sealed and no full-data or production run is authorized. See
  `experiments/pubchemqc100k_architecture/results/pair_gps_2d_fair_screen/decision.md`.
- The exact-identity conservative 2D+3D head is implemented and locally tested;
  model training has not started. See
  `experiments/resource_bounded_architecture/README.md` and
  `platforms/colab/conservative_2d3d_fusion/README.md`.
- No repaired-2M EdgeState job has been submitted. Every new remote run must
  first appear in `ROADMAP.md` with its input, timing, checkpoint, and durable
  output contracts.

## Boundaries

- Common, OOD, and P8-hard are one-time acceptance scopes, not tuning data.
- Future sealed data remains locked.
- Router, MoE, dataset replacement, and closed late-fusion branches remain
  closed unless `ROADMAP.md` records a materially new question.
- Track B remains isolated from Track A production. Its frozen disposition is
  `production/04_evaluate/project_freeze/track_b_final_decision.md`.
- Train and inference geometry must remain ETKDG-consistent.

## Evidence Map

| Question | Authoritative pointer |
|---|---|
| What ships now? | `production/README.md` |
| Why is repaired-2M pure 2D recommended? | `production/04_evaluate/project_freeze/track_a_final_decision.md` |
| What did the architecture tournament decide? | `experiments/resource_bounded_architecture/README.md` |
| What passed for EdgeState? | `experiments/resource_bounded_architecture/results/edge_state_100k_multiseed/decision.md` |
| What happened in official PCQM? | `experiments/pcqm_edge_state_full/README.md` |
| What did PairGPS2D establish? | `experiments/pubchemqc100k_architecture/results/pair_gps_2d_fair_screen/decision.md` |
| Where are remote records? | `platforms/README.md` |
| Where are experiments indexed? | `experiments/README.md` |
| Where are rejected branches? | `experiments/_closed/README.md` |
| Which model assets exist? | `models/README.md` |

The immediate execution order is defined only in `ROADMAP.md`.
