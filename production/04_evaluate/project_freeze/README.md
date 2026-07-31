# Project Freeze Sprint

## Frozen Decisions

Model research was frozen early on 2026-07-30:

- Track A: `track_a_final_decision.md`;
- Track B: `track_b_final_decision.md`;
- machine-readable status: `final_decisions.json`.

Only packaging, inference verification, cost accounting, figures,
documentation, and correctness fixes remain. The later deadline below is the
absolute stop for any already-running work, not authorization for more model
research.

The Track A packaging gate and the `2026-08-01 12:00` registry control point
were both met early, on 2026-07-31: the repaired-2M presets have hashed
checkpoints, a tested public loader, cost accounting, and a valid/invalid/OOD
smoke test, so the recommendation moved off routed v4. The schedule and
missed-gate rows below are the plan as written on 2026-07-28 and are retained
unedited as provenance; where a fallback says "keep routed v4 registered", that
branch was not taken. Live registry state is only in `CURRENT_STATE.md`.

Packaging evidence added under this directory:

- `public_inference_consistency/` - the public path reproduces the accepted
  external metrics within `1e-4 eV`;
- `inference_latency/` - latency, parameter, and encoder-pass accounting for
  both presets and both routed-v4 measurement modes;
- `public_api_smoke_test/` - valid, invalid, and out-of-domain SMILES behavior;
- `cost_comparison/` - DFT versus ML cost on one shared ten-molecule set, with
  the three DFT scopes kept separate;
- `experimental_offset/` - size and sign of the gap against literature
  experimental values, recorded as a limitation rather than an accuracy result;
- `presentation_evidence/` - every number the 2026-08-19 interview deck quotes,
  resolved to a local file, plus the outline claims that need correcting.

## Deadline

The project enters a hard freeze on **2026-08-04 at 12:00 JST**. At that time:

- stop every unfinished training and search job;
- preserve the last atomic checkpoints, logs, and manifests;
- submit no new architecture, dataset, seed, Router, or scale-up experiment;
- change only documentation, packaging, figures, and correctness defects found
  during final verification.

The freeze exists to protect preparation time for the second screening on
2026-08-19. It overrides the broader research backlog.

## Deliverable Levels

### Minimum guaranteed

- Keep `routed_gps7_gps9_schnet_500k_v4` as the production B3LYP model.
- Keep the accepted v3 LightGBM Delta plus UQ/OOD bundle as the calibrated
  near-GW delivery baseline.
- Preserve Retention-D and the accepted multi-expert ensembles as accuracy-mode
  evidence without changing the production registry.
- Deliver one normalized model comparison, cost table, artifact inventory,
  inference smoke test, and slide-ready result package.

### Expected

- Finish the Track B PCQM nested search, retrain the selected four encoders on
  the accepted 1M cache, tune bounded fusion on scaffold development only, and
  read fixed official-validation once after architecture freeze.
- Train and accept the two repaired-2M lightweight SchNet branches, export
  aligned embeddings, and make one predeclared bounded 2D+3D Track A decision
  on common, OOD, and P8-hard evidence.

### Stretch

- Promote the Track A bounded 2D+3D candidate only if it passes every external
  gate and has a tested inference loader.
- Recompute Delta labels and refit UQ/OOD against that exact new B3LYP base.
- Produce a bounded demonstration database. Do not attempt a full commercial
  database build during this sprint.

## Resource Assignment

| Resource | Role | Constraint |
|---|---|---|
| IMS A100 queue | Track B full-1M encoders; Track A two SchNet branches; embedding extraction | Prefer at most six concurrent project GPUs: four for Track B and two for Track A |
| IMS CPU/Lustre | Immutable input acceptance and output hashing | All writes remain below `/lustre/home/users/sm2/chou` |
| SCNet Kunshan CPU/storage | Source of the accepted repaired-2M primary and secondary graph caches | No new graph construction |
| SCNet BW-1 DCU | 2D emergency fallback only | Do not reopen DCU SchNet deployment during the sprint |
| Kaggle current account | Failure fallback for one blocked GPU chain or portable evaluation | Do not duplicate healthy IMS jobs |
| Extra Kaggle accounts | Unused reserve | Do not spend time provisioning unless IMS is unavailable for more than six hours |
| Local Ryzen 9700X / RTX 5060 8 GB | Acceptance, fusion heads, external evaluation, reporting, packaging | No full encoder training |
| Colab | Last-resort interactive fallback | Avoid manual Drive workflows unless both IMS and Kaggle are blocked |

The immediate transfer dependency is approximately 10 GB of accepted Track A
primary and secondary graph shards from Kunshan to IMS, including both
acceptance manifests and all sidecars. Hash acceptance must complete before
either Track A GPU job starts.

## Schedule

| Date | Track A - production | Track B - PCQM specialist | Delivery |
|---|---|---|---|
| Tue 2026-07-28 | Package and transfer both accepted repaired-2M 3D views to IMS; prepare two resumable SchNet jobs | Finish the active 50K/100K/three-seed search and freeze one configuration per encoder | Freeze this plan and the comparison protocol |
| Wed 2026-07-29 | Accept transferred caches and launch primary plus augmented SchNet in parallel | Launch four selected full-1M encoders in parallel; preserve best/last checkpoints and embedding parts | Prepare fixed common/OOD/P8-hard and PCQM evaluation commands |
| Thu 2026-07-30 | Continue or accept SchNet jobs; begin embedding extraction as each branch finishes | Accept checkpoints and embeddings; tune identity, correction bound, head width, dropout, and seed on scaffold development only | Start normalized result and compute-cost tables |
| Fri 2026-07-31 | Run one bounded 2D+3D fusion screen and fixed external evaluation | Freeze the full architecture, then read fixed official-validation once | Record positive or negative decisions with hashes |
| Sat 2026-08-01 | Freeze the Phase 8 B3LYP base and production registry decision | Freeze the task-routed PCQM specialist decision | Benchmark inference cost and run public API smoke tests |
| Sun 2026-08-02 | If and only if a new production base was accepted by noon, recompute OE62 base predictions and Delta inputs | No new Track B training | Refit Delta/UQ or retain the existing v3 bundle with an explicit compatibility note; build a bounded demo table |
| Mon 2026-08-03 | No architecture changes | No architecture changes | Complete model database, figures, ablation table, limitations, provenance, tests, and presentation source material |
| Tue 2026-08-04 | Stop unfinished compute at 12:00 JST | Stop unfinished compute at 12:00 JST | Archive partials, run final verification, commit, tag, and freeze the project |

## Execution Control

Each day has two control points:

- `09:00 JST`: inspect schedulers, atomic checkpoints, manifests, and blockers;
  select only work already authorized by this plan.
- `21:00 JST`: accept completed artifacts, record metrics and hashes, and apply
  the declared fallback to any missed gate.

| Deadline | Required evidence | Missed-gate action |
|---|---|---|
| 2026-07-28 21:00 | Track B search has a complete or resumable state for all four encoders; Track A transfer inventory lists every primary/secondary shard and manifest | Preserve the latest search checkpoints; do not add trials |
| 2026-07-29 12:00 | Both Track A caches match their accepted hashes on IMS | Close Track A full-3D training and freeze the strongest accepted 2D candidate |
| 2026-07-29 21:00 | Four selected Track B 1M jobs and two Track A SchNet jobs are submitted or running with atomic resume | Use accepted fixed-config Track B encoders for any chain that could not start |
| 2026-07-30 21:00 | Completed encoder assets are independently accepted; remaining jobs have healthy checkpoints under the 12-hour limit | Stop unhealthy jobs after one diagnosis and use their accepted fallback |
| 2026-07-31 12:00 | Track B has all assets needed to freeze a development-selected architecture | Freeze GINE v7 or the accepted fixed-config Route B result; no deadline extension |
| 2026-07-31 21:00 | Track A has one fixed bounded-fusion external evaluation | Reject Track A fusion immediately if the first external gate fails |
| 2026-08-01 12:00 | Production B3LYP model and registry decision are frozen | Keep routed v4 registered |
| 2026-08-01 21:00 | Track B specialist decision and compute-cost report are frozen | Retain the best accepted task specialist without further tuning |
| 2026-08-02 12:00 | A deployable new B3LYP inference path exists if Delta/UQ is to be rerun | Keep the accepted v3 Delta/UQ bundle and document its base identity |
| 2026-08-03 21:00 | Comparison table, figures, manifests, tests, limitations, and presentation source package are complete | Remove nonessential figures or demos; do not restart training |
| 2026-08-04 12:00 | All remote work is terminal or explicitly stopped; atomic partials are archived | Stop every remaining job regardless of expected completion time |
| 2026-08-04 18:00 | Repository verification, commit, tag, and freeze record are complete | Documentation-only repair is allowed; scientific results remain frozen |

Every control point is recorded in `CURRENT_STATE.md` while active. Accepted
metrics and decisions go only to their owning experiment result directory.

## Decision Gates

### Track A

Promote the repaired-2M bounded 2D+3D candidate only when:

- common regression is no worse than `0.0005 eV`;
- OOD or P8-hard improves by at least `0.001 eV`;
- the other hard domain regresses by no more than `0.0005 eV`;
- at least 95% of rows have both 3D views;
- three seeds point in the same direction when three seeds are available;
- checkpoint, embedding, source-index, target, and manifest hashes pass;
- inference is reproducible and its encoder-pass cost is reported.

Failure returns the production decision to routed v4. Retention-D remains a
general-model candidate and the fixed ensembles remain accuracy-mode evidence.

### Track B

- Select encoders and fusion only on scaffold development.
- Read the fixed official-validation subset once after full architecture freeze.
- Compare against GINE v7 fixed-valid Gap MAE `0.184618341 eV`.
- Keep any accepted result task-routed and Gap-only.
- Never use official test or the sealed 20K.

### Delta and UQ

- A new B3LYP base must be frozen with a working inference path by
  2026-08-02 at 12:00 JST to trigger a Delta/UQ rerun.
- Otherwise retain the accepted v3 LightGBM Delta plus UQ/OOD baseline and
  document that it is calibrated to its historical v3 B3LYP base.
- Do not claim calibrated uncertainty for an uncalibrated new base.

## Stop Rules

Stop or skip work immediately when any of these applies:

- Track A graph transfer is not accepted by 2026-07-29 at 12:00 JST: skip the
  full Track A SchNet run and freeze the strongest already accepted 2D result.
- An encoder exceeds 12 wall-clock hours without a valid resumable checkpoint:
  diagnose once, then use the accepted fallback rather than redesigning it.
- Track A bounded fusion fails its first external gate: close it without more
  weights, seeds, or routing.
- Track B full-1M assets are incomplete by 2026-07-31 at 12:00 JST: use the
  accepted fixed-configuration Route B assets or GINE v7 and close the route.
- The production base is not frozen by 2026-08-01 at 12:00 JST: routed v4 stays
  registered and the remaining time moves to delivery.

Forbidden during the sprint:

- new molecule acquisition or another dataset scale;
- new MoE, learned Router, OOF generation, distillation, or compression;
- new QM9/Track C architecture families;
- reproducing the 2025 equivariant paper;
- new conformer protocols or cutoff searches;
- tuning against official-valid, official test, or sealed evidence;
- provisioning extra cloud accounts while a healthy accepted path is running.

## Freeze Package

The final package must contain:

1. one production B3LYP decision and registry identity;
2. one separate PCQM Gap-specialist decision;
3. one Delta/UQ compatibility decision;
4. common, OOD, P8-hard, PCQM, latency, parameter, and encoder-pass tables;
5. model/checkpoint/embedding manifests with hashes and provenance;
6. an inference smoke test with representative valid, invalid, and OOD SMILES;
7. slide-ready architecture, data-flow, residual, calibration, and comparison
   figures;
8. a limitations section that separates production, accuracy-mode, specialist,
   negative, and incomplete evidence;
9. a tagged repository state with no active remote job required for the claims.

Status of those items as of 2026-07-31:

| Item | State |
|---|---|
| 1 production B3LYP decision and registry identity | done: `track_a_final_decision.md`, key `repaired_2m_dense_2d` |
| 2 PCQM Gap-specialist decision | done: `track_b_final_decision.md` |
| 3 Delta/UQ compatibility decision | done: not refitted; stays on its v3 base |
| 4 metric, latency, parameter, encoder-pass tables | Track A done, plus a DFT-versus-ML cost record; PCQM and one normalized cross-track table remain |
| 5 checkpoint manifests with hashes | done for Track A; Track B custody copy already hashed |
| 6 valid/invalid/OOD inference smoke test | done: `public_api_smoke_test/` |
| 7 slide-ready figures | open; the numbers behind them are collected in `presentation_evidence/` |
| 8 limitations section | Track A boundaries done (`track_a_final_decision.md` claim boundary, `experimental_offset/`); one consolidated section remains |
| 9 tagged repository state | open |
