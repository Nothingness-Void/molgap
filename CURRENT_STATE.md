# Current State

> The only source of live project truth, kept short enough to read in full. It
> states what is true and points at the record; it never restates a metric that a
> decision document already owns. Task ordering is in `ROADMAP.md`, the code map
> is in `ARCHITECTURE.md`, and each experiment's own decision record holds its
> method and numbers.

Track A/B/C ownership is defined only in `TRACKS.md`.

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

The fixed equal ensemble of the original-1M and repair-v2 dual-GPS experts is the
strongest deployable candidate under review. It passed the independent sealed
comparison but needs four GPS encoder passes, so it is not registered as the
default. The sealed set is read-only and can never select architecture, weights,
or hyperparameters.

| Constraint in force | Why | Evidence |
|---|---|---|
| Do not launch full-2M ordinary fusion | 30K compute-shape A/B: `176/160/6` costs 77.8% params and 48.2% time of `192/192/6` at a `+0.000154 eV` fused delta, but both ordinary FusionHeads regress against Retention-D by over `0.002 eV`; a positive bounded-residual identity pilot is required first | `experiments/schnet_arch/schnet_arch_repaired_2m_30k/decision.md` |
| Compression is closed | The 30%-teacher student passed internal exact-2M but failed fixed external retention (common `+0.00482/+0.00570 eV`, P8-hard `+0.01187/+0.01481 eV`); retained only as specialist evidence | `experiments/distillation/distilled_2m_scnet/decision.md` |
| Treat the 2M coverage expert as domain-opposed, not better | Paired residual attribution: it improves OOD but damages P8-hard, and the wider frozen fusion compresses correlated 2D features through the same 192-d bottleneck | `experiments/repaired_2m_scaling/scaling_residual_attribution/decision.md` |
| Sealed comparison result | Ensemble accepted under the frozen protocol | `experiments/multi2d_experts/multi2d_final_eval/decision.md` |

## Hierarchical Oracle Authorization

The predeclared Oracle-only gate passed: Retention-D seed 42 is the general base,
the accepted original-1M plus repair-v2 equal ensemble is the hard teacher, and
PCQM GINE v4 is a deterministic task-level Gap expert. The molecular gain
survives a 10% hard-teacher call budget on P8-hard without a common-set
regression.

This authorizes scaffold-disjoint OOF gain-label generation **only** — not Router
training and not deployment. The saved external gain labels are evaluation
evidence and are explicitly forbidden as Router training labels. Five repaired-2M
OOF folds are frozen at exactly 400,000 rows each with zero cross-fold scaffold
overlap. The Retention-D seed42/43/44 equal ensemble stays a three-pass
accuracy-mode candidate and does not replace one-pass seed42.

Contracts and job configs: `experiments/repaired_2m_scaling/results/gps7_gps9_oof/manifest.json`.
Three-seed ensemble: `experiments/repaired_2m_scaling/results/retention_d_three_seed_equal_ensemble_decision.md`.
Metrics, hashes, and cost accounting: `experiments/repaired_2m_scaling/hierarchical_oracle_20260725/decision.md`.

## Track B - PCQM Leaderboard

- Local GINE v7 is the strongest task-level PCQM Gap specialist candidate.
  The nested 1M run selected epoch 2 at `0.187982 eV` scaffold-dev and reached
  `0.184618 eV` on the fixed official-validation 5K, improving local v6 by
  `0.000654 eV`.
- This is a leaderboard-oriented local validation result, not a leaderboard
  score. Official test, sealed 20K, and the production registry remain
  untouched.
- Decision and artifact pointers:
  `experiments/pcqm_gine_expert/results/local_scaleup_1m_v7_decision.md`.
- The local PCQM 1M bounded-fusion aligned graph cache is complete and accepted:
  1,001,954 of 1,005,000 rows passed all three aligned views, with all 1,203
  declared files present and matching SHA256. Kaggle wave 1 reached terminal
  `COMPLETE` for:
  GPS9 (`nothingnessvoid/molgap-rb-gps9-probe-20260727`) and augmented SchNet
  (`nothingnessvoid/molgap-rb-aug-schnet-r1-20260727`). Their output manifests
  require independent acceptance before they count as quality evidence.
  All four encoders were completed and independently accepted on the IMS A100
  queue under the isolated
  `/lustre/home/users/sm2/chou/molgap-pcqm-route-b` root. The offline
  Torch/PyG environment, all 1,203 graph files, four warm-starts, and four real
  CUDA forward/backward/checkpoint preflights are accepted. Scaffold-dev Gap
  MAE is `0.175763/0.173214/0.130572/0.128314 eV` for GPS9, GPS11-160,
  primary SchNet, and augmented SchNet respectively. All four embedding sets
  contain aligned `915,012/81,961/4,981` train/dev/official rows and passed
  independent artifact hashes. These four runs are fixed-hyperparameter
  baselines, not tuned architecture ceilings. A nested encoder search is now
  authorized: deterministic `50K` broad search, top-four `100K` promotion, and
  top-two three-seed confirmation for all four encoders. SchNet cutoff is fixed
  at `6.0 Angstrom`; it is not searched. Fixed official-valid labels remain
  unread. Fusion identity and head hyperparameters must be selected on
  development evidence before the architecture is frozen. Live stage contract:
  `experiments/pcqm_route_b/results/run_plan.json`.

## Active Remote Work

Only jobs that are running or blocked appear here. Finished rounds are recorded
in each experiment's own log, so this section stays short enough to read at once.

| Workstream | Platform | State | Detail |
|---|---|---|---|
| Track B PCQM encoder hyperparameter search | IMS | CUDA preflight passed; GPS9 completed all twelve 50K trials and promoted four configurations, while GPS11-160, primary SchNet, and augmented SchNet remain active in their dependency chains | `experiments/pcqm_route_b/results/run_plan.json` |
| Track B PCQM 1M fusion | IMS | blocked on encoder search and later development-only fusion tuning; four original fixed-config embeddings remain accepted baselines | `platforms/_records/ims/pcqm_route_b_migration/remote_acceptance/encoder_acceptance.json` |
| Track A hierarchical 2D+3D fusion | — | 3D caches no longer block it: both views are accepted and 99.34% of source rows carry both. Still blocked on two trained repaired-2M SchNet checkpoints and their embedding parts | `experiments/repaired_2m_scaling/STATUS.md` |
| Full repaired-2M SchNet training | Colab | disabled pending the bounded-residual gate token | `experiments/repaired_2m_scaling/STATUS.md` |

Completed-round records:

| Experiment | Log |
|---|---|
| Repaired-2M scaling | `experiments/repaired_2m_scaling/results/REMOTE_LOG.md` |
| PCQM GINE expert | `experiments/pcqm_gine_expert/results/REMOTE_LOG.md` |
| PubChemQC 100K architecture | `experiments/pubchemqc100k_architecture/results/REMOTE_LOG.md` |
| Multi-2D expert rounds | `experiments/multi2d_experts/multi2d_2m_1m3d_fusion/decision.md` |
| Conformer protocol A/B | `experiments/conformer_protocol/results/decision.md` |
| Distillation | `experiments/distillation/distilled_2m_scnet/decision.md` |
| Acquisition rounds R10/R11/R03 | `platforms/_records/kaggle/acquisition/launches/molgap_2m_continuation_launch_20260722/` |

Candidate acquisition rounds R10, R11, and general R03 passed manifest,
return-code, checksum, schema, and finite-label checks but remain candidate data
until within-round, cross-round, and historical-inventory reconciliation
finishes. The future sealed 20K remains locked.


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

The repaired-2M data gate is complete and accepted: the row ledger reconciles
3,437,037 source rows, the fixed-size manifest keeps the targeted 500K, retains
1,228,539 additional exact-2M rows, and replaces 271,461 rows with
quality-filtered candidates. The materialized 2M table has unique CID/SMILES
identities and no sealed-source rows.

Track A's matched repaired-2M GPS9 comparison is complete: GPS9 is rejected as
a global replacement and retained as a hard expert. The active Track A gate is
the fixed-identity bounded 2D+3D fusion path described in
`experiments/repaired_2m_scaling/STATUS.md`.

GPS7/GPS9 OOF is an optional prerequisite for reopening a molecule-level
Router, not for completing fixed-identity fusion. Its ten-job SCNet file is a
non-executable placeholder and must not be submitted as prepared work. Router
training remains forbidden until genuine held-out gain labels exist.

Track C's QM9 screen and PubChemQC 100K transfer screen are complete. Their
bounded 2D+3D fusion result is an accepted architecture input to Tracks A and B,
not a production promotion. Track B's PCQM 1M run is the active leaderboard
test. The masked PCQM Gap-only chain remains an explicit specialist and cannot
replace routed v4 or the Track A general base without a separate deployment
gate.

| Question | Record |
|---|---|
| What did the data gate decide? | `experiments/repaired_2m_scaling/results/decision.md` |
| What is the unified scale-up evidence? | `experiments/repaired_2m_scaling/scaleup_full_analysis/decision.md` |
| What is the critical path and stop rule? | `experiments/repaired_2m_scaling/results/one_week_plan_20260723.md` |
| Which assets exist and what needs repair? | `production/04_evaluate/inventory/model_inventory_audit/decision.md` |

Hard constraints and the reading protocol remain authoritative in `AGENTS.md`.
