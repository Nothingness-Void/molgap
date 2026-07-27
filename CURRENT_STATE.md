# Current State

> This is the only source of live project truth. Exact metrics and immutable
> experiment decisions live under `results/`; task ordering lives in
> `ROADMAP.md`; dated method history lives in the experiment's decision record.

## Production Baseline

- **Recommended model:** routed dual-GPS v4.
- **Registry key:** `phase8_routed_dualgps_hybrid`.
- **Inference:** `src/molgap/inference.py`, lazily exported by
  `src/molgap/__init__.py`.
- **Registry:** `src/molgap/constants.py`.
- **Decision:** `production/03_train/routed_dual_gps_v4/gps_arch_routed_decision.md`.

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
  `experiments/schnet_arch/schnet_arch_repaired_2m_30k/decision.md`.

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
  `experiments/repaired_2m_scaling/scaling_residual_attribution/decision.md`.
- The sealed set is read-only and cannot be used for architecture, weight, or
  hyperparameter selection.
- Evidence: `experiments/multi2d_experts/multi2d_final_eval/decision.md` and
  `experiments/distillation/distilled_2m_scnet/decision.md`.

## P8.20 Hierarchical Oracle

- The predeclared Oracle-only gate passed. Retention-D seed 42 remains the
  general base, the accepted original-1M plus repair-v2 equal ensemble is the
  hard teacher, and PCQM GINE v4 is a deterministic task-level Gap expert.
- The molecular gain survives the 10% hard-teacher call budget on P8-hard
  without a common-set regression. This authorizes genuine scaffold-disjoint
  OOF gain-label generation, not Router training or deployment.
- The saved external gain labels are evaluation evidence only and are
  explicitly forbidden as Router training labels. Five repaired-2M
  scaffold-disjoint OOF folds are now frozen at exactly 400,000 rows each with
  zero cross-fold scaffold overlap. Prediction and gain-label contracts plus
  ten SCNet job configurations are prepared but not submitted:
  `experiments/repaired_2m_scaling/results/gps7_gps9_oof/manifest.json`.
- The Retention-D seed42/43/44 equal ensemble is retained only as a three-pass
  accuracy-mode candidate; it does not replace the default or one-pass seed42:
  `experiments/repaired_2m_scaling/results/retention_d_three_seed_equal_ensemble_decision.md`.
- No sealed-20K rows were opened and the production registry is unchanged.
  Exact metrics, input hashes, cost accounting, and the decision are in
  `experiments/repaired_2m_scaling/hierarchical_oracle_20260725/decision.md`.

## PCQM Specialist Candidate

- Local GINE v7 is the strongest task-level PCQM Gap specialist candidate.
  The nested 1M run selected epoch 2 at `0.187982 eV` scaffold-dev and reached
  `0.184618 eV` on the fixed official-validation 5K, improving local v6 by
  `0.000654 eV`.
- This is a leaderboard-oriented local validation result, not a leaderboard
  score. Official test, sealed 20K, and the production registry remain
  untouched.
- Decision and artifact pointers:
  `experiments/pcqm_gine_expert/results/local_scaleup_1m_v7_decision.md`.
- The local PCQM 1M Route B aligned graph cache is complete and accepted:
  1,001,954 of 1,005,000 rows passed all three aligned views, with all 1,203
  declared files present and matching SHA256. Wave 1 is now running on Kaggle:
  GPS9 (`nothingnessvoid/molgap-rb-gps9-probe-20260727`) and augmented SchNet
  (`nothingnessvoid/molgap-rb-aug-schnet-r1-20260727`). GPS11-160 plus primary
  SchNet are the next verified parallel wave after a real GPU-slot release. This
  does not yet have a model-quality result. Live stage contract:
  `experiments/pcqm_route_b/results/run_plan.json`.

## Active Remote Work

### SCNet

- Full repaired-2M GPS embedding and residual-head chain completed and passed
  artifact acceptance. GPS7/GPS9 each produced 40 contiguous, hash-verified
  50K-row embedding parts. The three `+-0.10 eV` residual-head seeds improve
  the fixed GPS7+GPS9 equal identity on internal exact-2M test by a mean
  `0.001821 eV` average MAE and `0.002062 eV` Gap MAE; all targets improve for
  all seeds and the average-MAE seed standard deviation is `0.000012 eV`.
  This passes the internal gate only. The three-pass candidate still requires
  frozen common/OOD/P8-hard evaluation and compute-cost accounting before any
  promotion. No sealed-20K rows were used and production is unchanged.
  Decision:
  `experiments/repaired_2m_scaling/results/three_gps_embedding_residual/decision.md`.
- Three-GPS learned routing pilot job `709815` completed and all holdout,
  checkpoint, prediction, identity, finite-value, and SHA256 checks passed.
  The learned pre-dispatch Router is rejected: it collapses to GPS9 for every
  molecule, uses GPS7 only for part of LUMO, and never calls GPS11-160. The
  three-pass dense gate is positive on internal test/common/OOD versus GPS9 by
  `0.002828/0.001865/0.003912 eV`; P8-hard average regresses a statistically
  inconclusive `0.000226 eV`, driven by `+0.000623 eV` Gap. A fixed GPS7+GPS9
  equal blend is the robust two-pass control: common/OOD/P8-hard average
  improves by `0.001044/0.001760/0.000312 eV` versus GPS9. GPS11 contributes
  useful correlated-error diversity to dense/mean blending despite its weak
  standalone result. For PCQM-valid, dense and equal-three reach
  `0.302120/0.299602 eV`, still behind routed v4 and the accepted PCQM GINE
  expert, so PCQM remains deterministically specialist-routed. Advance only
  the two-pass equal GPS7+GPS9 base and three-pass dense base to the bounded
  dual-SchNet A/B; do not advance the hard Router. This is accepted pilot
  evidence, not production registration or a substitute for formal OOF.
  Decision:
  `experiments/repaired_2m_scaling/results/three_gps_router_fusion/decision.md`.
- The follow-on hierarchical 2D+3D contract is implemented but not submitted:
  either the fixed GPS7+GPS9 equal blend or frozen three-GPS dense prediction
  is the exact identity path, and primary plus augmented lightweight SchNet
  embeddings can only add a `+-0.10 eV` bounded correction. It uses a new
  scaffold-disjoint split inside the untouched base model test rows and stops
  if fewer than 95% of those rows have both 3D views.
  This stage remains blocked on accepted primary/secondary repaired-2M graph
  caches, two trained SchNet checkpoints, and their embedding parts.
- Repaired-2M GPS11-160 jobs `709534`, `709562`, and `709563` completed and
  their checkpoint, metrics, predictions, and complete `2,000,000 x 160`
  embeddings passed artifact acceptance. The model is rejected as a global
  replacement, hard expert, PCQM expert, and automatic full-scale Fusion
  identity path. Relative to repaired-2M GPS9, average MAE regresses by
  `0.01403/0.00831/0.01988 eV` on common/OOD/P8-hard; PCQM Gap improves by
  `0.00740 eV` but remains `0.01116 eV` worse than routed v4 500K and far
  behind the accepted PCQM GINE specialist. GPS11 trained from scratch, unlike
  the warm-started GPS7/GPS9 controls, but its late plateau and broad external
  regressions do not justify continuation. Keep it only as bounded diversity
  evidence. Any later Route B pilot must compare a GPS9 identity path against
  GPS11 identity before using the full-scale Fusion protocol. Decision:
  `experiments/repaired_2m_scaling/results/gps11_160_seed42_decision.md`.
- The two parallel model-improvement routes completed their first gates
  without changing the production registry. Route A jobs `709046`/`709047`
  found that repaired-2M GPS9 improves common and P8-hard over Retention-D
  GPS7 but regresses OOD and PCQM. It is rejected as a global replacement and
  retained only as a hard-expert candidate. The target-specific Oracle then
  passed at a 10% GPS9 call budget, authorizing scaffold-disjoint OOF
  gain-label generation but not Router training. Decisions:
  `experiments/repaired_2m_scaling/results/gps9_seed42_decision.md`.
  `experiments/repaired_2m_scaling/results/gps7_gps9_oracle_20260725/decision.md`.
- Route B jobs `709051`-`709054` completed on the frozen scaffold-disjoint
  PubChemQC 100K/10K/9,997 split. GPS11-160 has the best pure-2D validation and
  test average MAE and the best test Gap MAE, so it advances with GPS7/GPS9
  controls to the two-SchNet fusion screen. Acceptance:
  `experiments/pubchemqc100k_architecture/results/remote_acceptance.json`.
- Route B's SchNet contract is the lightweight `176/160/6` architecture for
  both conformer branches. The legacy `192/192/6` SchNet is explicitly
  forbidden. The pure-2D comparison passed and Kaggle kernels
  `nothingnessvoid/molgap-pc100k-light-schnet-primary` and
  `nothingnessvoid/molgap-pc100k-light-schnet-augmented` version 1 both failed
  before epoch 0 because Kaggle assigned P100 GPUs while stock
  `torch 2.10.0+cu128` omitted `sm_60`. Version 2 conditionally installs the
  previously validated `torch 2.7.1+cu126` compatibility runtime and both
  kernels completed and both checkpoints plus embedding payloads passed strict
  artifact acceptance. The augmented model is materially stronger than the
  primary-only model even under one-view inference. The three-seed frozen
  Fusion screen selected the strict two-SchNet-pass Precision architecture:
  GPS9 + GPS11-160 + primary SchNet + two-conformer-trained augmented SchNet,
  with both SchNets evaluated on one primary conformer. Its PubChemQC 100K
  test average/Gap MAE is `0.138046/0.165819 eV`, improving pure GPS11-160 by
  `0.004424/0.005221 eV`. A third SchNet forward improves only
  `0.000699/0.000754 eV` and is rejected on cost. This authorizes a frozen
  full-scale protocol, not production promotion. A subsequent three-seed
  head A/B replaced the shared gated-sum bottleneck with a GPS11-160 identity
  path plus a bounded residual correction. A validation-only three-scale,
  three-seed A/B selected a `+-0.10 eV` correction bound. The frozen head
  reaches PubChemQC 100K test average/Gap `0.134463/0.160809 eV`, improving
  the original gated head by `0.003583/0.005010 eV`. It is the retained
  full-scale Fusion protocol; external common/OOD/P8-hard evidence is still
  required.
  Manifest:
  `experiments/pubchemqc100k_architecture/results/experiment_manifest.json`.
  Decision:
  `experiments/pubchemqc100k_architecture/results/route_b_fusion_decision.md`.
  Head decision:
  `experiments/pubchemqc100k_architecture/results/route_b_head_ab_decision.md`.
  Scale decision:
  `experiments/pubchemqc100k_architecture/results/route_b_residual_scale_decision.md`.
- A local paired 50K construction A/B keeps `ETKDGv3 + MMFF(maxIters=200)`:
  versus bare ETKDG it costs `1.589x` construction wall time but improves the
  frozen Route B equal-seed ensemble average/Gap MAE by
  `0.009868/0.008612 eV`. At measured 12-worker throughput the extra cost
  extrapolates to `0.90 h/1M`, so bare ETKDG is rejected as a default
  acceleration path. This does not change the production registry. Decision:
  `experiments/conformer_protocol/results/decision.md`.
- Repaired-2M retention-D passed the three-seed general-model gate against
  retention-B. Mean common/OOD/P8-hard average-MAE improvements are
  `0.001217/0.001496/0.000932 eV`, and every domain improves for each of seeds
  42, 43, and 44. PCQM Gap regresses by `0.001058 eV` on average and remains a
  separately routed specialist domain. Seed 43/44 models and evaluation
  artifacts were retrieved; remote/local SHA256 values match and predictions
  are finite. Keep seed 42 as the single-pass general-base checkpoint; the
  repeat seeds are stability evidence, not an automatic deployment ensemble.
  No sealed-set access or registry change occurred. Decision:
  `experiments/repaired_2m_scaling/results/retention_d_multiseed_decision.md`. Manifest:
  `experiments/repaired_2m_scaling/results/retention_d_experiment_manifest.json`.
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
  `experiments/repaired_2m_scaling/retention_2m_scnet/experiment_manifest.json`.
- P8.17 distillation jobs `703633` and `703653` completed; external job `704975`
  rejected the student as a global compression replacement. Both complete 2M
  embeddings and aligned 997,445-row FP16 prefixes remain reproducibility
  artifacts, but no fusion training is authorized. Decisions:
  `experiments/distillation/distilled_2m_scnet/decision.md` and
  `experiments/distillation/distilled_2m_external_eval/decision.md`.
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
  Configuration: `experiments/pcqm_gine_expert/gap_head_pilot/experiment_manifest.json`.
  Decision: `experiments/pcqm_gine_expert/gap_head_pilot/decision.md`.
  Source decision: `experiments/pcqm_gine_expert/train_residual_scan/decision.md`.
- Independent artifact acceptance job `704402` passed all model, prediction,
  embedding-part, Parquet-part, finite-value, row-accounting, uniqueness, and
  SHA256 checks. Record:
  `platforms/_records/scnet/overnight_20260723_acceptance.json`.
- The full P8.19 chain completed successfully: graph construction, GPS7/GPS9,
  dual-2D head, development evaluation, frozen-embedding staging, and graph
  cache archival. The verified staging payload is published as the private
  Kaggle dataset `nothingnessvoid/molgap-2m1m-fusion-staging-20260722`.
- Local handoff: `experiments/multi2d_experts/multi2d_2m_hard20k/`.
- The future sealed 20K remains locked.

### Repaired-2M 3D Handoff

- Kunshan primary graph job `117854186` stopped after 36/100 atomic shards
  because the original-1M cache was loaded before `fork`, causing copy-on-write
  memory amplification. The isolated-`spawn` repair job `117872652` resumed the
  same immutable directory and completed all 100 shards in `07:32:46` with
  `23.47 GB` peak RSS. Its completion report reconciles 2,000,000 requested
  rows as 869,142 reused, 1,119,974 newly built, and 10,884 failed conformers,
  leaving 1,989,116 graphs. Formal graph acceptance remains the first step of
  the already queued secondary job; do not treat process completion alone as
  accepted scientific evidence. Incident and completion records:
  `platforms/_records/scnet/kunshan_repaired_2m_3d/primary_oom_recovery_20260726.json`.
  `platforms/_records/scnet/kunshan_repaired_2m_3d/primary_build_completion_20260726.json`.
- Kunshan secondary-conformer job `117857094` failed before construction
  because the primary acceptance code incorrectly required graph-local `cid`
  and SMILES attributes. Repaired-2M graphs intentionally store `source_idx`,
  coordinates, and labels; identity lives in the immutable source CSV. The
  acceptance code now resolves CID/SMILES by `source_idx`, verifies optional
  embedded identity when present, and checks all three labels against the CSV.
  Replacement `117950883` exposed a second validator-only mismatch before
  construction: the real builder stores all 100 sidecars under
  `graph_shards/reports/`, not beside the graph files. The remote schema and
  sidecar fields were inspected directly, and the validator now checks that
  exact schema. Acceptance-only job `117958648` then completed in `00:30:21`
  and strictly accepted all 100 primary shards: 1,989,116 unique graphs,
  10,884 recorded conformer failures, complete source-target alignment, and
  matching graph/sidecar hashes. The retrieved manifest SHA256 matches the
  remote copy. Seed-`314159` secondary build `117966453` is now running on
  compute node `j05r4n04` with atomic 20K-row shards and resume enabled. Both
  earlier failed jobs produced no secondary shards and did not modify the
  primary cache. Incident and accepted relaunch:
  `platforms/_records/scnet/kunshan_repaired_2m_3d/secondary_acceptance_fix_20260727.json`.
  `platforms/_records/scnet/kunshan_repaired_2m_3d/secondary_relaunch_after_acceptance_20260727.json`.
  Do not duplicate the running secondary build.
  Launch record:
  `platforms/_records/scnet/kunshan_repaired_2m_3d/secondary_launch_20260726.json`.
- The repaired-2M primary graph build was stopped on Colab after 25/100
  durable shards (500,000 source rows) because the capped eight-worker CPU
  path was too slow. No complete shard was lost. Kunshan input and code hashes
  passed, and a real compute-node ETKDG preflight passed with torch 2.1.2,
  PyG 2.5.3, and RDKit 2023.09.6. The resumed CPU-only job retains 32 CPUs,
  110 GB RAM, 28 isolated workers, and the same 20K atomic-shard contract.
  Do not duplicate it while pending or running.
  Upload and execution handoff:
  `platforms/_records/scnet/kunshan_repaired_2m_3d/`.
- A second notebook builds an independently seeded conformer view only after
  all 100 primary shards pass strict acceptance.
- The full-scale Route B primary SchNet protocol is frozen at
  `176/160/6`, cutoff `10 A`, dropout `0.05`. Its split now reproduces the GPS
  seed-42 roles on all 2,000,000 source rows before filtering failed ETKDG
  molecules; the previous filtered-list split is forbidden for fusion because
  it misaligns encoder roles.
- Notebook, wheel hashes, acceptance gate, and paths:
  `experiments/repaired_2m_scaling/results/3d_colab_plan.json`.

### Kaggle

- PCQM GINE continuation v5 completed and remains the accepted remote warm
  start for the local v6/v7 branch. Best epoch 68 reaches `0.191690 eV` on the
  frozen scaffold development split and `0.187320 eV` on the fixed
  official-validation 5K, improving accepted v4 by `0.009278 eV`. Prediction
  MAE was independently reproduced and all downloaded artifact hashes match.
  This remains a 250K-sample local protocol result, not a leaderboard score;
  official test, sealed 20K, and the production registry remain untouched.
  Decision:
  `experiments/pcqm_gine_expert/results/continuation_v5_decision.md`.
  Accepted checkpoints are published privately as
  `nothingnessvoid/molgap-pcqm-gin-v5-accepted-20260726`.
- The same-data PCQM GPS9-320 architecture pilot is rejected. Its best
  development MAE was `0.462255 eV`, training became non-finite from epoch 15,
  and fixed official-validation 5K Gap MAE was `0.491629 eV`. This invalidates
  the implementation/training configuration, not GPS as a model family; it
  lacked the published positional-encoding and optimization protocol. Do not
  resume or scale this checkpoint. Decision:
  `experiments/pcqm_gine_expert/results/gps9_320_pilot_decision.md`.
- PubChemQC 100K Route B second-conformer preparation completed as four
  bounded CPU kernels, version 3:
  `nothingnessvoid/molgap-pc100k-conformer-r0` through `r3`. Version 1 failed
  before data processing because Kaggle did not include the sidecar
  `variant.json`. Version 2 embedded the shard identity but exposed that the
  CPU image did not contain RDKit. Version 3 installs the pinned RDKit
  dependency and embeds the shard identity. Local acceptance loaded and hashed
  all 24 graph parts: 119,602 of 120,000 molecules succeeded, all retained
  `source_idx` values are unique, and labels/coordinates are finite. The
  immutable split input is the private dataset
  `nothingnessvoid/molgap-pubchemqc100k-arch-split-20260725`. Exact counts:
  `experiments/pubchemqc100k_architecture/results/remote_acceptance.json`.
- The accepted second-conformer cache is published as private dataset
  `nothingnessvoid/molgap-pc100k-second-conformer-v3-20260725`, version 2.
  Version 1 is incomplete because the CLI skipped nested directories and must
  never be mounted for training.
- The benchmark-specific PCQM4Mv2 Gap expert pilot completed as Kaggle kernel
  `nothingnessvoid/molgap-pcqm-gin-expert-pilot`, version 3. Its 11 graph
  shards and all declared artifacts passed count, uniqueness, finite-label,
  loadability, and SHA256 checks. The fixed official-valid 5K Gap MAE was
  `0.213504 eV`: `0.078186 eV` better than routed v4, but above the predeclared
  `0.20 eV` scale gate. The candidate is rejected as a hierarchical-Oracle
  prerequisite; no Router or GPS9/fusion expansion is authorized from it.
  Official test splits and the future sealed 20K were not accessed.
  Decision: `experiments/pcqm_gine_expert/results/decision.md`.
  Acceptance: `experiments/pcqm_gine_expert/results/acceptance.json`.
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
  `experiments/pcqm_gine_expert/results/continuation_v4_decision.md`.
- The original-1M late-blend gate completed and closed at validation. Fixed
  alpha improved average/Gap by only `0.000024/0.000017 eV`; learned alpha
  regressed. The `0.001 eV` dual-target gate failed, so the original test and
  all external sealed sets remained locked. Decision:
  `experiments/_closed/archive-r09-original1m-late-router/decision.md`.
- Candidate acquisition rounds R10, R11, and general R03 completed and their
  independently retrievable outputs passed manifest, return-code, checksum,
  schema, and finite-label checks. They remain candidate data until strict
  within-round, cross-round, and historical-inventory reconciliation finishes.
- Launch record:
  `platforms/_records/kaggle/acquisition/launches/molgap_2m_continuation_launch_20260722/`.
- The `coverage2m`, `hard20k`, and combined `multi2d` 2D+3D fusion controls all
  completed with valid checkpoints but regressed against the existing 1M
  fusion reference. This round is closed without sealed-set access or a model
  promotion. Decision and exact accepted metrics:
  `experiments/multi2d_experts/multi2d_2m_1m3d_fusion/decision.md`.

### Colab

- Repaired-2M ETKDG graph construction is authorized and its resumable
  notebook/wheel bundle is ready under
  `platforms/colab/repaired_2m_3d/`. Identity audit found 871,693
  repaired rows in the original 1M corpus, so the builder remaps reusable
  original coordinates by CID plus canonical SMILES and constructs only the
  missing identities. It writes 100 atomic 20K-row graph shards with SHA256
  reports and strict final validation.
- Full repaired-2M SchNet training remains disabled in the notebook. The fixed
  30K pilot showed ordinary FusionHead regression, so training requires the
  explicit bounded-residual gate token after the active PubChemQC 100K
  two-SchNet screen is accepted. The eventual model contract is lightweight
  `176/160/6`, cutoff 6.0, dropout 0.0. Plan:
  `experiments/repaired_2m_scaling/results/3d_colab_plan.json`.

## Closed Decisions

| Workstream | Current disposition | Evidence |
|---|---|---|
| Original 1M continuation | Specialist only; no global promotion | `experiments/expansion_1m/results/replay_fusion_decision.md` |
| Repair-v2 1M | Closed at pure-2D gate | `experiments/data_repair/repair_v2_2d_external_eval/decision.md` |
| Repair-v3 1.5M | Closed at pure-2D gate | `experiments/data_repair/repair_v3_1p5m_external_eval/decision.md` |
| Broad residual 98k | Specialist only; no global promotion | `experiments/data_repair/broad_residual98k_external_eval/decision.md` |
| Exact-2M coverage expert | Specialist only; P8-hard regression | `experiments/multi2d_experts/multi2d_2m_coverage/decision.md` |
| Exact-2M GPS transplant into 500K routed-v4 | Closed; all three paired seeds regressed | `experiments/_closed/archive-r07-exact2m-encoder-transplant/decision.md` |
| Full-1M fixed routed-v4 topology | Closed; always-dual reproduced, fixed route regressed | `experiments/_closed/archive-r08-full1m-routed-fusion/decision.md` |
| Original-1M late soft blend | Closed at scaffold-validation gate | `experiments/_closed/archive-r09-original1m-late-router/decision.md` |
| Archive rounds R01-R09 | Closed | `experiments/_closed/README.md` |

Do not rerun a closed branch unless `ROADMAP.md` records a materially new
hypothesis.

## Immediate Decision Gate

The repaired-2M data gate is complete and accepted. Its row ledger reconciles
3,437,037 source rows; the fixed-size manifest keeps the targeted 500K,
retains 1,228,539 additional exact-2M rows, and replaces 271,461 rows with
quality-filtered accepted candidates. The materialized 2M table has unique
CID/SMILES identities and no sealed-source rows. Decision:
`experiments/repaired_2m_scaling/results/decision.md`.

Retention-D GPS7 passed its fixed multi-seed gate. The active Route A test is
the matched repaired-2M GPS9 run and external comparison described above.
Route B independently tests the PubChemQC 100K architecture candidates under
one frozen scaffold split. Its two lightweight SchNet branches and fusion
stage may proceed only after the pure-2D comparison is accepted. The PCQM
frozen-head pilot is closed and consumes no further compute.
The one-week critical path and stop rules are fixed in
`experiments/repaired_2m_scaling/results/one_week_plan_20260723.md`.

Decision and unified evidence:
`experiments/repaired_2m_scaling/scaleup_full_analysis/decision.md`.
Inventory and repair checklist:
`production/04_evaluate/inventory/model_inventory_audit/decision.md`.

The masked PCQM Gap-only pilot is authorized only as the explicit specialist
chain above. It cannot replace routed v4 or retention B without separate
deployment routing.

Hard constraints and the reading protocol remain authoritative in `AGENTS.md`.
