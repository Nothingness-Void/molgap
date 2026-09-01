# Roadmap - Priorities and Backlog

> This file owns task order, triggers, and exit conditions. Live truth is in
> `CURRENT_STATE.md`; methods, metrics, and conclusions live in dated records.

## Goal

Select one Gap-only architecture for the official PCQM4Mv2 leaderboard that
fits a full-data run within 12 A100 hours. Kaggle 100K selection precedes any
molecular-research-server use, and all geometry must be ETKDG-consistent.

## Active queue

| Priority | ID | Task | Exit condition |
|---|---|---|---|
| P0 | B-PCQM100K-RING-HIERARCHY-SEED42 | Accept the frozen smallest-ring cache, then run one seed-42 candidate | Cache passes first; candidate beats the frozen comparator or the mechanism is closed |
| P1 | B-PCQM-A100-GATE | Benchmark the frozen winner on official-train graphs | Throughput, epoch time, and memory reserve meet the protocol |
| P1 | B-PCQM-FULL-TRAIN | Train exactly one frozen Gap-only winner on official PCQM train | Resumable run completes within 12 hours |
| P2 | B-PCQM-OFFICIAL-VALID | Evaluate the frozen full-data model once | Official validation artifacts and timing pass |
| P3 | B-PCQM-TESTDEV | Produce the final test-dev submission | Explicit user authorization after official validation |

The active experiment owns its detailed status and protocol at
`experiments/pcqm_gap_architecture/`. Do not start a GPU successor to P0
before cache acceptance.

## Mandatory gates

1. Before remote GPU submission, syntax/AST/manifest checks and immutable
   CPU-cache acceptance must pass; model execution remains remote.
2. Before external evaluation, the full-scale candidate and aligned outputs
   must be complete and frozen.
3. Before production promotion, the fixed Track A external gate, loader,
   registry, hash, latency, and smoke checks must pass together.
4. Every failure receives a dated decision and is closed; it is not rewritten
   into `CURRENT_STATE.md`.
5. Infrastructure-only failures may be diagnosed, repaired, and retried with a
   new remote version. The architecture, data roles, split, target, seed,
   optimizer, schedule, precision, and sealed-role flags must remain unchanged.

## Operating rules

- Do not modify the production registry while Track B is screening.
- Architecture claims use random initialization; no pretraining, warm start,
  fine-tuning, distillation, residual target, or prediction fusion.
- Use only the frozen official-train-derived 100K/10K Kaggle split for PCQM
  architecture selection. Never tune on common/OOD/P8-hard or sealed data.
- Do not access the molecular-research server until a candidate passes the
  three-seed Kaggle gate; later access is restricted to
  `/lustre/home/users/sm2/chou/`.
- Predict Gap directly. Track A HOMO/LUMO experiments do not authorize Track B.
- Test one material mechanism at a time. Scientific failures are not retried
  as seed, width, distance, optimizer, or schedule variants.
- Preserve invalid-molecule reason codes and ETKDG train/inference consistency.
- Router, MoE, dataset replacement, ordinary late fusion, and the old
  dual-SchNet residual remain closed unless a new question is recorded here.
- Remote monitoring is mechanical only: one Luna Max heartbeat per persistent
  monitor thread; terminal evidence is handed to the coordinator for analysis.

## Conditional backlog

| Question | Trigger | Bounded action |
|---|---|---|
| Local-heavy sparse-global schedule | Ring hierarchy fails, or the accepted winner misses the 12-hour A100 projection | Keep the accepted local EdgeState/wedge updates; enable global atom attention only after blocks 3, 6 and 9; compare with a freshly trained full-GPS control |
| Sparse non-covalent ContactState | Ring and sparse-global screens fail or leave a clear through-space hypothesis; CPU preflight is feasible | Build one ETKDG contact cache with fixed edge-count/type/coverage statistics; only after acceptance test a narrow separately normalized contact state |
| Compact Cartesian invariant body-order basis | Ring and the two higher-information screens fail, and a preflight fits the same ceiling | Isolate one CACE/CEIT-like invariant basis with `K=16`; no equivariant tensors or new optimizer |
| PairGPS2D sealed-test disposition | Explicit authorization | Establish arithmetic equivalence before any benchmark-selected precision claim |
| Conformer ensemble or NNP geometry | Accepted evidence identifies geometry as the limiting factor | Compare one frozen alternative geometry source as an input/teacher experiment |
| Geometry denoising teacher | Architecture is selected and receives a separate budget | Use the literature configuration while official roles remain sealed |
| Solid-state Delta head | A separate target is requested | Create an isolated target contract |
| Paper figures/write-up | Academic delivery is requested | Derive figures only from accepted decision records |

Track A delivery work remains in its own records and does not override the
active Track B queue. Closed-route indexes are
`experiments/_closed/pcqm_server_archive_index.md` and
`experiments/_closed/qm9_top20_archive_index.md`.
