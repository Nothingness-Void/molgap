# Roadmap - Priorities and Backlog

This file owns only task order, triggers, and exit conditions. Live model and
job state is in `CURRENT_STATE.md`; completed methods, metrics, and conclusions
belong to each experiment's decision record. Track ownership is defined in
`TRACKS.md`.

## Goal

Select one pure-2D, Gap-only architecture for the official PCQM4Mv2 leaderboard
under a hard 12-hour A100 full-training budget. Kaggle 100K selection must
finish before any molecular-research-server use.

## Active Queue

| Priority | ID | Task | Exit condition | Owner |
|---|---|---|---|---|
| P0 | B-PCQM100K-GLOBAL-STATE | Run one recurrent graph-state EdgeState candidate under the frozen seed-42 contract | Downloaded artifacts pass no-inference acceptance and the candidate strictly beats the EdgeState comparator, or the mechanism closes | `experiments/pcqm_gap_architecture/` |
| P0 | B-PCQM100K-MULTISEED | Confirm the seed-42 challenger at seeds 43/44 without changing data or training parameters | Challenger improves every seed and the three-seed mean, or closes | `experiments/pcqm_gap_architecture/` |
| P1 | B-PCQM-A100-GATE | Benchmark only the frozen Kaggle winner on official-train graphs | At least 1,800 graphs/s, no epoch above 32 minutes, projected run at most 10.5 hours, and at least 15% memory reserve | `experiments/pcqm_gap_architecture/` |
| P1 | B-PCQM-FULL-TRAIN | Train exactly one frozen Gap-only winner on official PCQM train | Timing gate passes; one resumable run completes inside the 12-hour budget | `experiments/pcqm_gap_architecture/` |
| P2 | B-PCQM-OFFICIAL-VALID | Evaluate the frozen full-data model once on official validation | Artifacts and inference timing pass the official protocol; no architecture tuning reopens | `experiments/pcqm_gap_architecture/` |
| P3 | B-PCQM-TESTDEV | Produce the final official test-dev submission | Explicit user authorization after official-validation acceptance | `experiments/pcqm_gap_architecture/` |

The accepted 100K EdgeState screen is evidence, not an active task. Its exact
decision is
`experiments/resource_bounded_architecture/results/edge_state_100k_multiseed/decision.md`.

The official-PCQM cache and first matched seed-42 comparator are accepted
evidence. Their exact decision is
`experiments/pcqm_gap_architecture/results/seed42_structural_vs_edge_state/decision.md`.

The learned-query and local-operator seed-42 screens are closed evidence. Their
exact decisions are `experiments/pcqm_gap_architecture/results/query_pool_seed42/decision.md`
and
`experiments/pcqm_gap_architecture/results/local_operator_search_seed42/decision.md`.

The accepted R3 validation gate, failed R5/R6/R7/R8/R9 branches, and untriggered R4
fallback are evidence. Their exact dispositions are
`experiments/top20_architecture_qm9/pair_gps_2d_r3_decision.md`,
`experiments/top20_architecture_qm9/edge_state_jk_readout_r5_decision.md`, and
`experiments/top20_architecture_qm9/edge_conditioned_r6_decision.md`, with R7
owned by `experiments/top20_architecture_qm9/graph_token_r7_decision.md` and R8
owned by `experiments/top20_architecture_qm9/multihop_edge_state_r8_decision.md`.
R9 is owned by
`experiments/top20_architecture_qm9/sparse_path_attention_r9_decision.md`.

## Mandatory Gates

1. **Before remote GPU submission:** local syntax/AST/manifest checks and
   immutable CPU-cache acceptance must pass. Forward/backward and memory checks
   run only in the remote GPU preflight; models are not executed locally.
2. **Before external evaluation:** the standalone full-scale candidate and its
   aligned predictions must be complete and frozen.
3. **Before production promotion:** the fixed Track A external gate must pass;
   then public loader, registry, hashes, latency, and smoke tests are updated in
   one production decision.
4. **On failure:** write a dated decision beside the experiment evidence and
   close the branch. Do not move the failure into `CURRENT_STATE.md`.

## Operating Rules

- Do not modify the production registry while Track B is screening.
- Architecture claims use random initialization; no pretraining, warm start,
  fine-tuning, or distillation may be credited as an architecture gain.
- Do not tune on common/OOD/P8-hard or sealed data.
- PCQM leaderboard architecture questions use the frozen official-train-derived
  100K/10K Kaggle split. QM9 and PubChemQC results are historical inspiration,
  not advancement gates or reusable weights.
- Do not access the molecular-research server until one candidate passes the
  three-seed Kaggle gate. Later access is restricted to
  `/lustre/home/users/sm2/chou/`.
- Predict Gap directly. HOMO/LUMO auxiliary targets, 3D inputs, residual
  targets, pretrained checkpoints, and prediction fusion are outside this
  screen.
- Continued pure-2D discovery tests one materially new architecture at a time.
  Every failure gets a decision record and cannot be retried as a seed or
  schedule variation; the next attempt must change the information flow.
- Do not rerun the rejected `0.10 eV` frozen-2D plus dual-SchNet residual.
- Geometry paths must preserve ETKDGv3+MMFF train-inference consistency.
- Router, MoE, OOF gain labels, and dataset replacement remain closed unless a
  new question and stop rule are added here first.
- Invalid molecules remain visible with reason codes; do not silently filter.
- Historical v3 Delta/UQ outputs must not be described as calibrated for the
  repaired-2M production model.

## Delivery Queue

These Track A delivery tasks remain valid but do not override the active Track B
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
| Experimental solid-state Delta head | A specific experimental target is requested |
| Extend the supported element set | Rejected-use analysis justifies refetch and retraining |
| Conformer ensemble or NNP geometry | Residual evidence identifies geometry as the limiting factor |
| Paper figures and write-up | An academic delivery is requested |

Completed work is indexed, without duplicated metrics, in
`experiments/README.md`, `experiments/_closed/README.md`, and
`production/README.md`.
