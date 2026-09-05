# Current State

> This file owns live truth only: production identity, active question,
> blockers, boundaries, and evidence pointers. Historical methods, metrics,
> and remote logs belong to their dated decision records.

## Production identity

- **Recommended model:** repaired-2M three-GPS dense pure 2D.
- **Registry:** `repaired_2m_dense_2d`; lower-cost preset:
  `repaired_2m_equal_2d`.
- **Public loader:** `load_repaired_2m_2d` and
  `predict_smiles_batch_repaired_2m_2d` in `src/molgap/inference.py`.
- **Authority:** `production/04_evaluate/project_freeze/track_a_final_decision.md`.
- The previous routed-v4 model remains registered for compatibility; its Delta
  and UQ bundles are not calibrated for the repaired-2M presets.

## Active objective

Track B selects one Gap-only PCQM4Mv2 leaderboard specialist under the hard
12-hour A100 budget. Selection uses only the official-train-derived
100K/10K internal split; official validation and test-dev stay sealed.

The no-attention shared GraphState strictly beat fresh full-GPS controls at
seeds 42, 43, and 44 while using fewer parameters and higher throughput. The
three-seed gate passed. It remains the frozen desktop handoff and comparison
anchor. The winner and complete arithmetic are in
`experiments/pcqm_gap_architecture/results/local_global_allocation_multiseed/decision.md`.
The desktop timing and memory gate is described in that directory's
`desktop_handoff.md`. The single-DCU Kunshan runtime gate
completed with mechanical acceptance; it establishes SCNet runtime
compatibility only and does not substitute for the required A100 measurement.
No full-data or official-role action is authorized before the A100 gate passes.

The bounded Ring-GraphState comparison completed and closed: its tiny seed-42
gain did not justify a 15.1% throughput loss or extra seeds, and its candidate
metrics also missed a required cache-lineage field. The exact decision is
`experiments/pcqm_gap_architecture/results/ring_graphstate_seed42/decision.md`.
The non-covalent ContactState screen passed mechanical acceptance but was
slower and less accurate than its fresh GraphState control, so that mechanism
is closed. Exact evidence is in
`experiments/pcqm_gap_architecture/results/contact_state_seed42/decision.md`.
The repaired body-order comparison completed on Kaggle1 and passed mechanical
acceptance, but its candidate was less accurate than its fresh GraphState
control. The exact mechanism is closed by
`experiments/pcqm_gap_architecture/results/body_order_moment_seed42/decision.md`.
The bounded Kunshan persistent-VectorState comparison completed and passed
mechanical acceptance after a provenance-only row-identity repair. Its
`0.000608 eV` seed-42 reduction was below the `0.001 eV` promotion threshold
and came with lower throughput, so the mechanism is closed without more seeds
or full-data training. GraphState9 remains the frozen anchor. The exact result
and repair boundary are in
`experiments/pcqm_gap_architecture/results/kunshan_vector_state_seed42/decision.md`.
The next bounded discovery question, K2 projected first/second-moment readout,
is submitted on Kunshan as job `121048634`. It compares a fresh GraphState9
control with only the final readout changed. Live status is in
`experiments/pcqm_gap_architecture/results/kunshan_moment_readout_seed42/STATUS.md`.

Recent torsion and atom--bond dual-stream questions are closed; their complete
records are on `archive` and indexed by
`experiments/_closed/pcqm_server_archive_index.md`.

## Prior architecture evidence

- Track A EdgeState three-seed evidence remains at
  `experiments/resource_bounded_architecture/results/edge_state_100k_multiseed/decision.md`.
- The independent PairGPS2D 100K validation and A100 throughput decisions are
  at `experiments/pubchemqc100k_architecture/results/`.
- QM9 R3 persistent EdgeState is retained as the pure-2D comparator at
  `experiments/top20_architecture_qm9/pair_gps_2d_r3_decision.md`.
- QM9 TGT, PairGPS R2, R4, and R5--R10 are archive-only; see
  `experiments/_closed/qm9_top20_archive_index.md`.
- The unfinished repaired-2M PairGPS2D attempt (job `1322114`) is archive-only;
  it has no final metrics and must not be resumed. See
  `experiments/_closed/pcqm_server_archive_index.md`.
- The conservative 2D+3D repair is separate and has not started training;
  see `experiments/resource_bounded_architecture/README.md`.

## Execution and boundaries

- Accepted PCQM caches and architecture comparisons are retained in their
  experiment records.
- SCNet Kunshan card1 is runtime-compatible for bounded GraphState jobs via
  the isolated DTK 23.10/Python 3.10 environment. The single-DCU GraphState
  runtime gate `120869290` completed with mechanical acceptance against the
  accepted 100K/10K geometry cache; its protocol, launch record, compact
  summary, and decision are
  `experiments/pcqm_gap_architecture/kunshan_graphstate_runtime_protocol.md`
  and `experiments/pcqm_gap_architecture/results/kunshan_graphstate_runtime_gate/`.
  Evidence for the earlier synthetic compatibility probe remains at
  `platforms/_records/scnet/training_compatibility_20260904/`.
- SCNet Xi'an Card2 now has an isolated DTK 22.10/PyG runtime with a
  device-width-corrected HIP `torch-scatter` build. A 10K train-role probe
  matched the pinned T4 throughput at batch 96, and two exact batch-48 100K
  three-epoch gates completed with accepted checkpoints. Their validation
  trajectories were not sufficiently repeatable for single-run ranking, so
  Xi'an is restricted to twice-reproduced paired pre-screens followed by a
  canonical Kunshan confirmation. The exact boundary is in
  `experiments/pcqm_gap_architecture/results/xian_card2_runtime_gate/decision.md`.
- The user reported 200 available accelerator-hours on each SCNet region on
  2026-09-06. This is availability, not permission to broaden a frozen
  experiment: Xi'an remains the rapid repeated pre-screen backend and Kunshan
  remains the canonical paired-confirmation backend.
- No molecular-research-server run is authorized before the Kaggle selection
  gate; later access is restricted to `/lustre/home/users/sm2/chou/`.
- Track B predicts Gap directly and cannot alter the Track A production
  registry. Official validation/test-dev and future sealed data are locked.
- Train and inference geometry must use the same ETKDG construction.
- Every new remote run needs a protocol, immutable cache acceptance, atomic
  checkpointing, independently retrievable outputs, and a dated decision.
- Scientific failures close a route; only infrastructure failures may be
  repaired and retried under the unchanged scientific contract.

## Evidence map

| Question | Authority |
|---|---|
| What ships? | `production/README.md` |
| What is active? | `ROADMAP.md` and `experiments/pcqm_gap_architecture/README.md` |
| PCQM closed routes | `experiments/_closed/pcqm_server_archive_index.md` |
| QM9 closed transfer routes | `experiments/_closed/qm9_top20_archive_index.md` |
| QM9 R3 comparator | `experiments/top20_architecture_qm9/pair_gps_2d_r3_decision.md` |
| PCQM 100K contract | `experiments/pcqm_gap_architecture/pcqm100k_gap_screen_protocol.md` |
| Remote handoff rules | `platforms/REMOTE_HANDOFF.md` |
| Artifact inventory | `models/README.md` |

Task ordering is defined only in `ROADMAP.md`; monitor handoff rules are in
`AGENTS.md`.
