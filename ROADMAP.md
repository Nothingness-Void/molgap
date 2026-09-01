# Roadmap - Priorities and Backlog

This file owns only task order, triggers, and exit conditions. Live model and
job state is in `CURRENT_STATE.md`; completed methods, metrics, and conclusions
belong to each experiment's decision record. Track ownership is defined in
`TRACKS.md`.

## Goal

Select one Gap-only architecture for the official PCQM4Mv2 leaderboard under a
hard 12-hour A100 full-training budget. Kaggle 100K selection must finish
before any molecular-research-server use. Geometry candidates must remain
ETKDG-consistent and earn advancement against the accepted pure-2D comparator.

## Active Queue

| Priority | ID | Task | Exit condition | Owner |
|---|---|---|---|---|
| P0 | B-PCQM100K-RING-HIERARCHY-SEED42 | Build and accept the frozen deterministic smallest-ring hierarchy cache, then run one seed-42 candidate | CPU cache acceptance passes first; one hash-pinned GPU run either strictly beats the frozen distance-plus-angle comparator or closes the mechanism | `experiments/pcqm_gap_architecture/` |
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

The sparse topology-wedge cache was accepted. R2 failed during preflight on an
OGB `AtomEncoder` API mismatch before any epoch, and the one implementation-
only R3 repair then completed the unchanged seed-42 contract. R3 strictly
passed by a small margin and is frozen in
`experiments/pcqm_gap_architecture/results/sparse_triangle_edge_state_r3_seed42/decision.md`.
The paired multiseed contract completed and selected Sparse Triangle as the
accepted pure-2D comparator. Its decision is
`experiments/pcqm_gap_architecture/results/sparse_triangle_edge_state_multiseed/decision.md`.
The accepted geometry cache and three-candidate seed-42 screen selected only
distance-plus-angle bottom fusion for paired confirmation. Its paired seeds
43/44 task completed the strict gate: all three paired deltas are negative and
the mean improved, but seed 44 is marginal. The exact decision is
`experiments/pcqm_gap_architecture/results/geometry_bottom_fusion_multiseed/decision.md`.
The resulting distance-plus-angle model is the frozen 100K comparator. The
exact seed-42 geometry decision is
`experiments/pcqm_gap_architecture/results/geometry_bottom_fusion_seed42/decision.md`.

The contracted torsion question added only a persistent 16-wide state on
non-backtracking bonded paths `i-j-k-l`, fixed periodic features
`[sin(phi), cos(phi), sin(2phi), cos(2phi)]`, one shared gated update cell, and
sparse exchange with the three bonds and two adjacent wedges. It reuses the
accepted ETKDGv3+MMFF94s single-conformer geometry, the frozen 100K/10K roles,
and the unchanged 192/64/16 GPS9, direct Gap, FP32, batch-48, AdamW, and
40-epoch/patience-8 contract. The CPU torsion cache must pass no-model
acceptance before one six-and-a-half-hour Kaggle1 paired GPU task. The fresh
distance-plus-angle comparator is trained first under the same seed-42 data
order. No seed 43/44, full-data, official-role, or server work is included.
The CPU cache passed no-model acceptance. GPU versions 1 and 2 stopped in an
invalid numerical identity preflight before training. A user-directed manual
audit established that CUDA sparse reductions, not nonzero torsion injection,
caused the mismatch; version 3 uses exact parameter-level checks under the
unchanged scientific contract. The audit and launch identity are in
`experiments/pcqm_gap_architecture/results/sparse_torsion_edge_state_seed42/gpu_manual_audit.md`.
Version 3 completed the comparator and candidate epochs 0--38 before its
planned search budget expired. Its hash-pinned resume preserved all RNG,
optimizer, scheduler, data-order, checkpoint, and trace state. Version 4
verified the bundle but stopped before training on an RNG-device mismatch;
version 5 restored that state, ran only candidate epoch 39, and completed
artifact assembly. Independent no-inference acceptance passed, while sparse
torsion failed the scientific gate. The route is closed by
`experiments/pcqm_gap_architecture/results/sparse_torsion_edge_state_seed42/gpu_decision.md`.

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
5. **Infrastructure retry authority:** after preserving evidence, automatically
   diagnose, repair, test, and resubmit infrastructure-only failures without a
   new user confirmation. The retry must preserve architecture, data roles,
   split, target, seed, optimizer, schedule, precision, and sealed-role flags;
   it must use a new remote version and remain idempotent. Scientific gate
   failures are not retryable under this authority.

## Operating Rules

- Do not modify the production registry while Track B is screening.
- Architecture claims use random initialization; no pretraining, warm start,
  fine-tuning, or distillation may be credited as an architecture gain.
- Do not tune on common/OOD/P8-hard or sealed data.
- PCQM leaderboard architecture questions use the one frozen
  official-train-derived 100K/10K Kaggle screening split. The sparse-wedge
  cache is a deterministic derived representation of that same split. QM9 and
  PubChemQC results are historical inspiration, not advancement gates or
  reusable weights.
- Do not access the molecular-research server until one candidate passes the
  three-seed Kaggle gate. Later access is restricted to
  `/lustre/home/users/sm2/chou/`.
- Predict Gap directly. HOMO/LUMO auxiliary targets, residual targets,
  pretrained checkpoints, and prediction fusion are outside this screen.
- The active geometry screen is deterministic single-conformer
  ETKDGv3/MMFF94s distance/angle injection inside Sparse Triangle blocks. The
  one authorized torsion question reuses that conformer and adds only its
  sparse torsion state; an independent SchNet, late fusion, residual
  correction, and conformer ensembles remain closed.
- Continued discovery tests one materially new mechanism at a time. The three
  geometry candidates may share one task because they form one predeclared
  factorization of distance and angle information.
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

| Task | Trigger | Bounded action |
|---|---|---|
| Recent-architecture sparse torsion screen | Completed and scientifically failed at seed 42 | Closed by its accepted decision; no additional torsion seeds are authorized |
| Sparse atom--bond dual-stream attention | Completed, accepted, and scientifically failed at seed 42 | Closed by its decision; no extra seeds or parameter variants are authorized |
| Ring/conjugation hierarchy | Satisfied: torsion and sparse bond-attention both scientifically failed | Execute only the frozen smallest-ring hierarchy in `ring_hierarchy_seed42_protocol.md`; no ring-definition or training grid is authorized |
| Compact Cartesian invariant body-order basis | Torsion, bond-stream, and ring mechanisms all fail and a parameter/throughput preflight fits the same ceiling | Isolate one CACE-like invariant basis; do not add equivariant tensors, pretraining, or a new optimizer in the same screen |
| PairGPS2D sealed-test disposition | Explicit authorization reopens the independent branch after its validation-only decision | Establish arithmetic equivalence before using benchmark-selected TF32 for an accuracy claim |
| Experimental solid-state Delta head | A specific experimental target is requested | Open a separate target contract; do not alter Track B |
| Extend the supported element set | Rejected-use analysis justifies refetch and retraining | Version the new data and registry separately |
| Conformer ensemble or NNP geometry | Residual evidence identifies geometry as the limiting factor | Compare an accepted ETKDG input against one frozen MACE-OFF23/AIMNet2 geometry source; label it an input/teacher experiment, not an architecture gain |
| Geometry denoising teacher | A randomly initialized architecture has already been selected and teacher compute receives a separate budget | Start from the Frad/SliDe configuration evidence; keep official validation and test-dev sealed |
| Paper figures and write-up | An academic delivery is requested | Derive figures from accepted decision records only |

Completed work is indexed, without duplicated metrics, in
`experiments/README.md`, `experiments/_closed/README.md`, and
`production/README.md`.
