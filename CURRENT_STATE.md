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
12-hour A100 budget. Selection uses only the official-train-derived Kaggle
100K/10K internal split; official validation and test-dev stay sealed.

The accepted comparator is Sparse Triangle plus distance-and-angle bottom
fusion on the fixed pure-2D/ETKDG contract. The single active question is
whether full GPS global attention is over-allocated. A fresh full-GPS control,
a sparse block-3/6/9 attention schedule, and a no-attention shared GraphState
are frozen in
`experiments/pcqm_gap_architecture/local_global_allocation_seed42_protocol.md`.

The deterministic smallest-ring CPU cache completed and passed no-model
acceptance. Its immutable evidence is
`experiments/pcqm_gap_architecture/results/ring_hierarchy_seed42/cache_acceptance.json`;
the ring GPU successor is deferred while the explicitly prioritized
local/global allocation screen owns the one-GPU slot. Live experiment status
is `experiments/pcqm_gap_architecture/STATUS.md`.

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

- Accepted PCQM caches, seed-42 comparisons, geometry confirmation, and sparse
  topology-wedge evidence are retained in their experiment records.
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
