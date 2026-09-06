# Roadmap - Priorities and Backlog

> This file owns task order, triggers, and exit conditions. Live truth is in
> `CURRENT_STATE.md`; methods, metrics, and conclusions live in dated records.

## Goal

Select one Gap-only architecture for the official PCQM4Mv2 leaderboard that
fits a full-data run within 12 A100 hours. Internal 100K selection precedes any
molecular-research-server use, and all geometry must be ETKDG-consistent.

## Active queue

| Priority | ID | Task | Exit condition |
|---|---|---|---|
| P0 | B-KUNSHAN-DISCOVERY | Execute staged seed-42 architecture screens on Kunshan | Accept and attribute each result within the bounded discovery plan |
| P0 | B-PCQM-A100-GATE | Desktop benchmarks the frozen GraphState winner on official-train graphs | Throughput, projected epoch time, and at least 15% memory reserve meet the protocol |
| P1 | B-PCQM-FULL-TRAIN | Desktop trains exactly one frozen Gap-only winner on official PCQM train | Resumable run completes within 12 hours |
| P2 | B-PCQM-OFFICIAL-VALID | Evaluate the frozen full-data model once | Official validation artifacts and timing pass |
| P3 | B-PCQM-TESTDEV | Produce the final test-dev submission | Explicit user authorization after official validation |

The GraphState three-seed gate passed; its frozen decision and desktop handoff
are under
`experiments/pcqm_gap_architecture/results/local_global_allocation_multiseed/`.
The supplemental single-DCU Kunshan runtime gate completed with mechanical
acceptance, but it does not close the A100 budget gate.
Ring-GraphState, ContactState, and the compact invariant body-order basis are
closed. None revoked the frozen GraphState desktop handoff or authorized a
second full-data model. On 2026-09-05 the user authorized further server-side
experiments entirely on Kunshan. Their ordering, hypotheses and shared compute
cap are owned by `experiments/pcqm_gap_architecture/kunshan_discovery_plan.md`.
The K1 persistent-VectorState and K2 projected-moment readout screens are
closed below the material-gain threshold. K3 conjugated-component
communication completed. Its CPU cache passed and K3a job
`121082200` completed: the descriptor-only control was directionally positive
but below the material-gain gate. K3b job `121111542` passed all three frozen
seed-42 gates, while also exposing `0.000661 eV` repeat drift in the unchanged
descriptor control. ComponentState is retained in the final shortlist, but no
confirmation seed or full-data action is implied without a separate budget
decision.

On 2026-09-06 the user reported a restored weekly allowance of 30 GPU-hours on
each Kaggle account. The two remaining random-initialized local-state questions
were frozen in `local_statistics_seed42_protocol.md` and completed on 2026-09-07.
PNA-style neighborhood statistics and gated persistent-edge retention both
lost to their separate fresh GraphState9 controls; both are closed without
confirmation seeds. GAPE remains a separate pretraining track and was not part
of these architecture claims.

Two materially distinct seed-42 Kaggle questions are now released under
`directed_bond_graphstate_seed42_protocol.md` and
`signnet_lappe_graphstate_seed42_protocol.md`. The first uses the accepted
wedge cache directly for directional edge-to-edge memory; it completed with a
sub-threshold weak positive result and no confirmation. The second passed its
CPU normalized-Laplacian cache gate, then lost its paired GPU comparison and
closed without confirmation. Neither route authorized confirmation seeds.

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
- Use only the frozen official-train-derived 100K/10K split for PCQM
  architecture selection. Never tune on common/OOD/P8-hard or sealed data.
- Keep the main databases unchanged: Track B stays on PCQM4Mv2 and Track A
  stays on the existing repaired-2M PubChemQC corpus. External datasets may
  support an explicitly labeled audit, OOD evaluation, or separately approved
  teacher route, but may not replace or silently augment the main training
  database.
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
- The active Track B architecture search is an authorized autonomous discovery
  loop. After each confirmed terminal handoff, the coordinator may accept and
  diagnose the result, close a scientific failure or repair an infrastructure
  failure, select one materially distinct backlog hypothesis, commit and push
  its frozen contract, submit exactly one successor GPU job, and retarget the
  same heartbeat without asking again. `QUEUED` and `RUNNING` never trigger
  scientific analysis or a new submission. Stop the loop when no defensible
  hypothesis remains, remote quota or immutable inputs are unavailable, a
  candidate reaches the three-seed desktop handoff gate, or the next action
  would require sealed roles, full-data training, production changes, or the
  molecular-research server.

## Conditional backlog

| Question | Trigger | Bounded action |
|---|---|---|
| Sparse non-covalent ContactState | Closed at seed 42 | Do not retry cutoff, width, depth, seed, or optimizer variants |
| Compact Cartesian invariant body-order basis | Closed at seed 42 | Do not retry width, radial count, seed, optimizer, or schedule variants |
| PairGPS2D sealed-test disposition | Explicit authorization | Establish arithmetic equivalence before any benchmark-selected precision claim |
| Conformer ensemble or NNP geometry | Accepted evidence identifies geometry as the limiting factor | Compare one frozen alternative geometry source as an input/teacher experiment |
| Geometry denoising teacher | Architecture is selected and receives a separate budget | Use the literature configuration while official roles remain sealed |
| Solid-state Delta head | A separate target is requested | Create an isolated target contract |
| Paper figures/write-up | Academic delivery is requested | Derive figures only from accepted decision records |

Track A delivery work remains in its own records and does not override the
active Track B queue. Closed-route indexes are
`experiments/_closed/pcqm_server_archive_index.md` and
`experiments/_closed/qm9_top20_archive_index.md`.
