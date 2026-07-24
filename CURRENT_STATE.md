# Current State

> This is the only source of live project truth. Exact metrics and immutable
> experiment decisions live under `results/`; task ordering lives in
> `ROADMAP.md`; dated method history lives in `docs/phaseN.md`.

## Production Baseline

- **Recommended model:** routed dual-GPS v4.
- **Registry key:** `phase8_routed_dualgps_hybrid`.
- **Inference:** `src/molgap/inference.py`, lazily exported by
  `src/molgap/__init__.py`.
- **Registry:** `src/molgap/constants.py`.
- **Decision:** `results/phase8/gps_arch_routed_decision.md`.

The v3 single hybrid remains a component/compatibility loader. v2 and v1 are
historical fallbacks, not recommended defaults.

## Execution Context

- One local Agent is active. There are no parallel Agent-owned worktrees or
  handoffs to reconcile.
- The P8.19 SCNet-to-Kaggle handoff is complete. Do not rename, relocate, or
  delete its accepted inputs, raw downloads, or result records.
- The local Agent may continue documentation and bounded local work while
  monitoring those remote jobs; it must not relaunch them without evidence of
  failure.

## Active Model Candidate

The fixed equal ensemble of the original-1M and repair-v2 dual-GPS experts is
the strongest deployable candidate under review.

- A local repaired-2M 30K SchNet compute-shape A/B replicated the efficiency
  result: `176/160/6` used 77.8% of the parameters and 48.2% of the training
  time of `192/192/6`, while fused average/Gap MAE changed by only
  `+0.000154/+0.000196 eV`. However, both ordinary FusionHeads regressed
  against Retention-D alone by more than `0.002 eV`. Do not launch full-2M
  ordinary fusion; first require a positive bounded residual identity-path
  pilot. Evidence:
  `results/phase8/experiments/schnet_arch_repaired_2m_30k/decision.md`.

- It passed the independent sealed comparison but requires four GPS encoder
  passes, so it is not registered as the default.
- The 30%-teacher student passed internal exact-2M evaluation but failed fixed
  external retention: common average/Gap regressed `+0.00482/+0.00570 eV` and
  P8-hard regressed `+0.01187/+0.01481 eV`.
- It improved OOD average/Gap by `0.00211/0.00325 eV` and PCQM Gap by
  `0.00323 eV`, so it is retained only as specialist evidence. Compression and
  its conditional 3D fusion branch are closed.
- Paired residual attribution shows the 2M coverage expert has opposing domain
  behavior rather than uniformly better capacity: it improves OOD but damages
  P8-hard, while the wider frozen fusion compresses correlated 2D features
  through the same 192-dimensional bottleneck. Evidence:
  `results/phase8/scaling_residual_attribution/decision.md`.
- The sealed set is read-only and cannot be used for architecture, weight, or
  hyperparameter selection.
- Evidence: `results/phase8/multi2d_final_eval/decision.md` and
  `results/phase8/distilled_2m_scnet/decision.md`.

## P8.20 Hierarchical Oracle

- The predeclared Oracle-only gate passed. Retention-D seed 42 remains the
  general base, the accepted original-1M plus repair-v2 equal ensemble is the
  hard teacher, and PCQM GINE v4 is a deterministic task-level Gap expert.
- The molecular gain survives the 10% hard-teacher call budget on P8-hard
  without a common-set regression. This authorizes genuine scaffold-disjoint
  OOF gain-label generation, not Router training or deployment.
- The saved external gain labels are evaluation evidence only and are
  explicitly forbidden as Router training labels. The OOF manifest remains
  `pending_generation`.
- No sealed-20K rows were opened and the production registry is unchanged.
  Exact metrics, input hashes, cost accounting, and the decision are in
  `results/phase8/hierarchical_oracle_20260725/decision.md`.

## Active Remote Work

### SCNet

- Two model-improvement routes are active in parallel without changing the
  production registry. Route A job `709046` trains repaired-2M Retention-D
  GPS9-192 from the accepted 1.5M GPS9 checkpoint; dependent job `709047`
  performs common/OOD/P8-hard/PCQM evaluation. Route B uses a frozen
  scaffold-disjoint PubChemQC 100K/10K/9,997 split. Its graph-subset job
  `709051` and GPS7/GPS9/GPS11-160 jobs `709052`/`709053`/`709054` are queued
  behind Route A because the account's 16-CPU concurrency limit is occupied.
  Pending jobs must not be relaunched merely for being quiet.
- Route B's SchNet contract is the lightweight `176/160/6` architecture for
  both conformer branches. The legacy `192/192/6` SchNet is explicitly
  forbidden. Full 3D/fusion training remains gated on the pure-2D comparison.
  Manifest:
  `results/phase8/experiments/pubchemqc100k_architecture/experiment_manifest.json`.
- Repaired-2M retention-D passed the three-seed general-model gate against
  retention-B. Mean common/OOD/P8-hard average-MAE improvements are
  `0.001217/0.001496/0.000932 eV`, and every domain improves for each of seeds
  42, 43, and 44. PCQM Gap regresses by `0.001058 eV` on average and remains a
  separately routed specialist domain. Seed 43/44 models and evaluation
  artifacts were retrieved; remote/local SHA256 values match and predictions
  are finite. Keep seed 42 as the single-pass general-base checkpoint; the
  repeat seeds are stability evidence, not an automatic deployment ensemble.
  No sealed-set access or registry change occurred. Decision:
  `results/phase8/repaired_2m/retention_d_multiseed_decision.md`. Manifest:
  `results/phase8/repaired_2m/retention_d_experiment_manifest.json`.
- Retention-aware exact-2M GPS7 controls were run as the first experiment
  authorized after the scale-up failure analysis. Existing uniform exact-2M is
  control A. B (`705497` -> `705498`) completed: common/OOD/P8-hard average
  improved by `0.00242/0.00204/0.00280 eV`, but PCQM Gap regressed by
  `0.01702 eV`, so B failed the global gate. C initially cached all 500K teacher
  targets and then hit an FP16/FP32 assignment error before training. The error
  was fixed, but after B was accepted for common/OOD/P8-hard and PCQM was split
  into a separate specialist, replacement jobs `706141` -> `706142` were
  deliberately cancelled before any completed epoch to avoid wasting card
  hours. Fixed configuration and gates:
  `results/phase8/retention_2m_scnet/experiment_manifest.json`.
- P8.17 distillation jobs `703633` and `703653` completed; external job `704975`
  rejected the student as a global compression replacement. Both complete 2M
  embeddings and aligned 997,445-row FP16 prefixes remain reproducibility
  artifacts, but no fusion training is authorized. Decisions:
  `results/phase8/distilled_2m_scnet/decision.md` and
  `results/phase8/distilled_2m_external_eval/decision.md`.
- PCQM4Mv2 official-train scan `703665` completed and produced an accepted
  200K raw hard pool. Domain audit found 103,440 radicals; do not train on the
  raw pool. The fixed clean pool has 95,909 rows. Because retention variant B
  already improves common/OOD/P8-hard, a separate B-based PCQM Gap specialist
  was tested: `706147` materialized the clean pool and embeddings, `706148`
  trained the output head, and `706149` evaluated it. All three jobs completed,
  but the candidate regressed common/OOD/P8-hard average by
  `+0.01615/+0.02153/+0.01065 eV` and PCQM Gap by `+0.08837 eV`.
  The frozen-head specialist is rejected and closed.
  Official PCQM valid and test are excluded from training, and the future
  sealed 20K remains locked.
  Configuration: `results/phase8/pcqm_gap_head_pilot/experiment_manifest.json`.
  Decision: `results/phase8/pcqm_gap_head_pilot/decision.md`.
  Source decision: `results/phase8/pcqm4mv2_train_residual_scan/decision.md`.
- Independent artifact acceptance job `704402` passed all model, prediction,
  embedding-part, Parquet-part, finite-value, row-accounting, uniqueness, and
  SHA256 checks. Record:
  `results/phase8/remote/overnight_20260723_acceptance.json`.
- The full P8.19 chain completed successfully: graph construction, GPS7/GPS9,
  dual-2D head, development evaluation, frozen-embedding staging, and graph
  cache archival. The verified staging payload is published as the private
  Kaggle dataset `nothingnessvoid/molgap-2m1m-fusion-staging-20260722`.
- Local handoff: `results/phase8/multi2d_2m_hard20k/`.
- The future sealed 20K remains locked.

### Kaggle

- PubChemQC 100K Route B second-conformer preparation is running as four
  bounded CPU kernels, version 3:
  `nothingnessvoid/molgap-pc100k-conformer-r0` through `r3`. Version 1 failed
  before data processing because Kaggle did not include the sidecar
  `variant.json`. Version 2 embedded the shard identity but exposed that the
  CPU image did not contain RDKit. Version 3 installs the pinned RDKit
  dependency and embeds the shard identity. All four version-3 workers
  remained `RUNNING` after the first-minute startup check. Each writes 5K graph
  shards plus atomic progress and completion manifests. The immutable split
  input is the private dataset
  `nothingnessvoid/molgap-pubchemqc100k-arch-split-20260725`.
- The benchmark-specific PCQM4Mv2 Gap expert pilot completed as Kaggle kernel
  `nothingnessvoid/molgap-pcqm-gin-expert-pilot`, version 3. Its 11 graph
  shards and all declared artifacts passed count, uniqueness, finite-label,
  loadability, and SHA256 checks. The fixed official-valid 5K Gap MAE was
  `0.213504 eV`: `0.078186 eV` better than routed v4, but above the predeclared
  `0.20 eV` scale gate. The candidate is rejected as a hierarchical-Oracle
  prerequisite; no Router or GPS9/fusion expansion is authorized from it.
  Official test splits and the future sealed 20K were not accessed.
  Decision: `results/phase8/pcqm_gine_expert_pilot/decision.md`.
  Acceptance: `results/phase8/pcqm_gine_expert_pilot/acceptance.json`.
- The bounded version 4 continuation passed. It resumed the accepted epoch-29
  optimizer/scheduler/scaler state, reused all 11 validated graph shards, and
  selected epoch 48. Fixed official-valid 5K Gap MAE is `0.196598 eV`, improving
  routed v4 by `0.095092 eV` and passing the predeclared `0.20 eV` gate by
  `0.003402 eV`. It is accepted only as the task-level PCQM Gap prerequisite
  for the planned Oracle study; no learned Router, GPS9/fusion expansion,
  sealed-set access, or registry change is authorized.
  Accepted private artifacts:
  `nothingnessvoid/molgap-pcqm-gin-v4-accepted-20260724`.
  Decision:
  `results/phase8/pcqm_gine_expert_pilot/continuation_v4_decision.md`.
- The original-1M late-blend gate completed and closed at validation. Fixed
  alpha improved average/Gap by only `0.000024/0.000017 eV`; learned alpha
  regressed. The `0.001 eV` dual-target gate failed, so the original test and
  all external sealed sets remained locked. Decision:
  `results/phase8/archive/archive-r09-original1m-late-router/decision.md`.
- Candidate acquisition rounds R10, R11, and general R03 completed and their
  independently retrievable outputs passed manifest, return-code, checksum,
  schema, and finite-label checks. They remain candidate data until strict
  within-round, cross-round, and historical-inventory reconciliation finishes.
- Launch record:
  `results/kaggle/acquisition/launches/molgap_2m_continuation_launch_20260722/`.
- The `coverage2m`, `hard20k`, and combined `multi2d` 2D+3D fusion controls all
  completed with valid checkpoints but regressed against the existing 1M
  fusion reference. This round is closed without sealed-set access or a model
  promotion. Decision and exact accepted metrics:
  `results/phase8/multi2d_2m_1m3d_fusion/decision.md`.

## Closed Decisions

| Workstream | Current disposition | Evidence |
|---|---|---|
| Original 1M continuation | Specialist only; no global promotion | `results/phase8/expansion_1m/replay_fusion_decision.md` |
| Repair-v2 1M | Closed at pure-2D gate | `results/phase8/repair_v2_2d_external_eval/decision.md` |
| Repair-v3 1.5M | Closed at pure-2D gate | `results/phase8/repair_v3_1p5m_external_eval/decision.md` |
| Broad residual 98k | Specialist only; no global promotion | `results/phase8/broad_residual98k_external_eval/decision.md` |
| Exact-2M coverage expert | Specialist only; P8-hard regression | `results/phase8/multi2d_2m_coverage/decision.md` |
| Exact-2M GPS transplant into 500K routed-v4 | Closed; all three paired seeds regressed | `results/phase8/archive/archive-r07-exact2m-encoder-transplant/decision.md` |
| Full-1M fixed routed-v4 topology | Closed; always-dual reproduced, fixed route regressed | `results/phase8/archive/archive-r08-full1m-routed-fusion/decision.md` |
| Original-1M late soft blend | Closed at scaffold-validation gate | `results/phase8/archive/archive-r09-original1m-late-router/decision.md` |
| Archive rounds R01-R09 | Closed | `results/phase8/archive/README.md` |

Do not rerun a closed branch unless `ROADMAP.md` records a materially new
hypothesis.

## Immediate Decision Gate

The repaired-2M data gate is complete and accepted. Its row ledger reconciles
3,437,037 source rows; the fixed-size manifest keeps the targeted 500K,
retains 1,228,539 additional exact-2M rows, and replaces 271,461 rows with
quality-filtered accepted candidates. The materialized 2M table has unique
CID/SMILES identities and no sealed-source rows. Decision:
`results/phase8/repaired_2m/decision.md`.

Retention-D GPS7 passed its fixed multi-seed gate. The active Route A test is
the matched repaired-2M GPS9 run and external comparison described above.
Route B independently tests the PubChemQC 100K architecture candidates under
one frozen scaffold split. Its two lightweight SchNet branches and fusion
stage may proceed only after the pure-2D comparison is accepted. The PCQM
frozen-head pilot is closed and consumes no further compute.
The one-week critical path and stop rules are fixed in
`results/phase8/repaired_2m/one_week_plan_20260723.md`.

Decision and unified evidence:
`results/phase8/scaleup_full_analysis/decision.md`.
Inventory and repair checklist:
`results/phase8/model_inventory_audit/decision.md`.

The masked PCQM Gap-only pilot is authorized only as the explicit specialist
chain above. It cannot replace routed v4 or retention B without separate
deployment routing.

Hard constraints and the reading protocol remain authoritative in `AGENTS.md`.
