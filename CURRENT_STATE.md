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

Track B's official-full geometry warm-start is closed. The eight-epoch successor
passed A100 preflight and completed seven epochs before patience-based early
stopping. Its best official-validation Gap MAE was `0.115825 eV`, versus
`0.099638 eV` for the exact source OGB-rich EdgeState GPS9 checkpoint on the
same split (`+0.016186 eV`, 16.25% worse). No official test role was read.

The completion manifest, best checkpoint, metrics, and validation predictions
are hash-consistent. PBS reported a host-memory cgroup excess only after those
artifacts and the final JSON were written; this is retained as an infrastructure
warning, not grounds to rerun a scientifically negative result. No IMS job or
heartbeat remains active. Evidence is under
`experiments/pcqm_geometry_warmstart/`.

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
- Molecular-research-server access remains restricted to
  `/lustre/home/users/sm2/chou/`; the active chain is fail-closed and bounded.
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
