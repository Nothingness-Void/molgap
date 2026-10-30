# Current State

> Live truth only. Historical methods, metrics, and failures live in the
> [conditional history index](docs/operations/CURRENT_STATE_HISTORY_INDEX.md)
> and the linked experiment decisions; task order lives in `ROADMAP.md`.

## Server execution status

The human authorized a separate [PPGT width-capacity500K pair](experiments/ppgt_capacity500k/README.md)
on Kaggle2. Its [first attempt](experiments/ppgt_capacity500k/STATUS.md) failed
before training because source evidence duplicated a pinned data manifest.
The minimal data-backed mount repair is CPU-verified; the unchanged retry
(retry_v3) was submitted once, with the failed window debited from the aggregate
budget, and was last observed RUNNING at 2026-10-30 22:55 JST (details in
[STATUS](experiments/ppgt_capacity500k/STATUS.md)).
Native qualification, completed outputs and Replay admission remain unobserved.

The [family-retention and GRIT challenge](experiments/family_retention_challenge_500k/README.md)
is a research proposal only: park PPGT expansion, retain ESA as an efficiency reserve, and qualify
one strong Common-Dev reference before a complete compact-GRIT comparison. No new compute or Replay claim follows.

## Closed chains

Terminal or paused, no active job or automatic successor; full text in the [archive](docs/operations/CURRENT_STATE_ARCHIVE_2026-10.md).

- PPGT/ESA500K scale task: both endpoints complete, no nomination; [archive](docs/operations/CURRENT_STATE_ARCHIVE_2026-10.md#ppgtesa500k-scale-task-terminal), [STATUS](experiments/modern2d_scale500k/STATUS.md), [decision](experiments/modern2d_scale500k/results/completed_v3/decision.md).
- ESA count-persistence pair: complete, reference registered, paused; [archive](docs/operations/CURRENT_STATE_ARCHIVE_2026-10.md#esa-count-persistence-pair-terminal), [STATUS](experiments/esa_count_persistence_100k/STATUS.md).
- PPGT/ESA structural screen: complete, ESA lead contextual only; [archive](docs/operations/CURRENT_STATE_ARCHIVE_2026-10.md#ppgtesa-structural-screen-terminal), [decision](experiments/modern2d_structure_100k/results/decision.md).
- Prepared PPGT/ESA100K pair: complete, RML closure blocked; [archive](docs/operations/CURRENT_STATE_ARCHIVE_2026-10.md#prepared-ppgtesa100k-pair-terminal), [decision](experiments/modern2d_configured_100k/prepared_pair/results/decision.md), [RML status](experiments/modern2d_configured_100k/prepared_pair/results/rml_status.json).
- Four-slot/eight-arm research campaign: terminal, no nomination; [archive](docs/operations/CURRENT_STATE_ARCHIVE_2026-10.md#four-slot-eight-arm-ppgtesa-research-campaign-terminal), [STATUS](experiments/modern2d_configured_100k/research_arms/STATUS.md), [interpretation](experiments/modern2d_configured_100k/research_arms/esa2/results/interpretation.md), [PPGT partial decision](experiments/modern2d_configured_100k/research_arms/ppgt1/results/decision.md).
- PPGT/ESA execution-repair diagnostic retry_v2: accepted, NO_TRAIN/context_only; [archive](docs/operations/CURRENT_STATE_ARCHIVE_2026-10.md#ppgtesa-execution-repair-diagnostic-retry_v2-terminal), [decision](experiments/modern2d_configured_100k/execution_repair/retry_v2/results/decision.md).
- Three-slot Modern2D campaign: terminal STOP_FOR_COST, partial; execution-equivalent optimization and budget feasibility need resolution before another bounded authority; [archive](docs/operations/CURRENT_STATE_ARCHIVE_2026-10.md#three-slot-modern2d-campaign-terminal), [LR decision](experiments/modern2d_configured_100k/esa_attribution/lr/results/decision_2026_10_21.md), [WD decision](experiments/modern2d_configured_100k/esa_attribution/wd/results/decision_2026_10_21.md).
- Configured PPGT/ESA paired100K screen: ERROR/STOP_FOR_COST, no successor; [archive](docs/operations/CURRENT_STATE_ARCHIVE_2026-10.md#configured-ppgtesa-paired-100k-screen-terminal), [STATUS](experiments/modern2d_configured_100k/STATUS.md), [decision](experiments/modern2d_configured_100k/results/decision_2026_10_20.md).
- PPGT execution profile: retry_v2 complete, execution-only; [archive](docs/operations/CURRENT_STATE_ARCHIVE_2026-10.md#ppgt-execution-profile-terminal), [STATUS](experiments/modern2d_configured_100k/ppgt_profile/STATUS.md), [retry decision](experiments/modern2d_configured_100k/ppgt_profile/retry_v2/results/decision_2026_10_21.md).
- Mature-family workflow trial: complete, workflow accepted; [archive](docs/operations/CURRENT_STATE_ARCHIVE_2026-10.md#mature-family-workflow-trial-terminal), [STATUS](experiments/family_workflow_trial_100k/STATUS.md), [decision](experiments/family_workflow_trial_100k/results/decision_2026_10_19.md).
- Modern2D100K formal retry: canceled by the human; [archive](docs/operations/CURRENT_STATE_ARCHIVE_2026-10.md#modern2d-100k-formal-retry-canceled), [STATUS](experiments/modern2d_100k_formal_screen/retry_v2/STATUS.md), [cancellation inspection](experiments/modern2d_100k_formal_screen/retry_v2/results/cancellation_2026-10-18.md).
- GPTrans four-arm campaign: Kaggle3 version3 COMPLETE, chains paused at pass10/60; bond-memory adaptation integrated, no result yet ([review](docs/operations/CURRENT_STATE_ARCHIVE_2026-10.md#post-routing-bond-memory-review), [review file](experiments/pcqm_v5_route_portfolio/bond_memory_review_2026-10-13.md)); [archive](docs/operations/CURRENT_STATE_ARCHIVE_2026-10.md#gptrans-four-arm-campaign-paused-after-first-segment), [protocol](experiments/pcqm_gptrans_research_four_arm_100k/protocol.md), [STATUS](experiments/pcqm_gptrans_research_four_arm_100k/STATUS.md), [segment decision](experiments/pcqm_gptrans_research_four_arm_100k/results/decision_pass10_accepted_2026-10-15.md).
- K1 and GPTrans execution diagnostics: closed, no promotion or Replay; [archive](docs/operations/CURRENT_STATE_ARCHIVE_2026-10.md#k1-and-gptrans-execution-diagnostics-closed), [K1](experiments/k1_infrastructure_t4_100k/results/decision_2026-10-15.md), [cached provider](experiments/k1_infrastructure_t4_100k/execution_providers/results/decision_2026-10-15.md), [GPTrans](experiments/gptrans_infrastructure_t4_100k/results/decision_2026-10-15.md), [follow-up](experiments/gptrans_infrastructure_t4_100k/followup/results/decision_2026-10-16.md).
- PPGT/ESA execution optimizations and Modern2D integration/native qualification: CPU-accepted, opt-in, V4 NO_TRAIN; [optimizations](docs/operations/CURRENT_STATE_ARCHIVE_2026-10.md#ppgtesa-execution-optimizations-cpu-accepted), [integration](docs/operations/CURRENT_STATE_ARCHIVE_2026-10.md#modern2d-integration-and-native-qualification-cpu-accepted), [native preflight STATUS](experiments/modern2d_model_integration/native_preflight/STATUS.md).

Closed architecture, diagnostic and execution studies route through the
[history index](docs/operations/CURRENT_STATE_HISTORY_INDEX.md#closed-gptrans-study-authorities).
Their own terminal decisions retain qualification, Replay, cost and custody
limits; those earlier authorizations release no automatic successor. CPU attempts/preparation
are RML-closed; stale-plan reconciliation preserved history and eligibility ([`research_memory/README.md`](research_memory/README.md)).

## Shared experiment infrastructure

The shared local CLI, typed Spec/addon registry, source packaging, prospective planning, launch binding,
output inspection, recovery and terminal/RML helpers are integrated. Family trainers own
model/loader/optimizer/selection/resume; platform adapters own submission, reconciliation and retrieval;
the CLI is never a platform submitter. Native qualification stays run/recipe/runtime-specific, and real
remote qualification is separate from local schema or synthetic checks.
Scientific-input continuation selects the original frozen run before compilation; new HEADs do not
upgrade old tasks. Live latency, fresh live submission/resumed execution and arbitrary recipe compilation
remain unestablished; no new compute or scientific release follows.

Start with the [new-agent quickstart](docs/operations/EXPERIMENT_QUICKSTART.md), then the
[addon guide](docs/operations/EXPERIMENT_ADDON_GUIDE.md) and
[registered workflow](docs/operations/REGISTERED_EXPERIMENT_WORKFLOW.md). Details: [configured Modern2D workflow](docs/operations/MODERN2D_CONFIGURED_WORKFLOW.md),
[shared training core](docs/operations/SHARED_TRAINING_CORE_SERVER_RETURN.md), [batch-topology reuse](docs/operations/TOPOLOGY_REUSE_SERVER_REVIEW.md),
[four-module runtime guide](docs/operations/MODULAR_EXPERIMENT_RUNTIME.md), [shared graph owner](docs/operations/GRAPH_SCREEN_EXTENSION.md),
[bounded verification record](docs/operations/INFRASTRUCTURE_VERIFICATION.md) and [workflow reform record](docs/operations/REFORM_MIGRATION.md)
(`INFRA_CPU_ACCEPTED`, server-local scope). Prior narrative: [archive](docs/operations/CURRENT_STATE_ARCHIVE_2026-10.md#shared-infrastructure-narrative-part-1), [reform verdicts](docs/operations/CURRENT_STATE_ARCHIVE_2026-10.md#workflow-reform-records-cpu-accepted).

## Production

- Recommended Track A model: repaired-2M three-GPS dense pure 2D.
- Registry: `repaired_2m_dense_2d`; lower-cost preset: `repaired_2m_equal_2d`.
- Loader: `load_repaired_2m_2d` in `src/molgap/inference.py`.
- Authority: `production/04_evaluate/project_freeze/track_a_final_decision.md`.
- Track B experiments cannot change this registry.

## Track B ownership

Neural-Atom K1 and GPTrans-T remain the frozen desktop-owned comparison
architectures. The desktop full chain and bounded continuations are terminal and
accepted; EdgeState remains the official-validation reference. Desktop owns any
new full-scale training, official evaluation, and final submission. Details and
decision pointers are in the [historical index](docs/operations/CURRENT_STATE_HISTORY_INDEX.md).

## Live boundaries and next actions

Cross-scale infrastructure is CPU-accepted; V2 IMS data passed [role acceptance](docs/operations/PCQM_UNIFIED_TWO_P_ACCEPTANCE.md).
Job `1733250.ccpbs1` exited0 with a retained memory warning; Kaggle mirrors are NOT_PUBLISHED.
The [configured Modern2D workflow](docs/operations/MODERN2D_CONFIGURED_WORKFLOW.md) is infrastructure
support, not a new remote release: exact native preflight remains required, each new recipe needs fresh
native qualification, and old campaign authority cannot release these new recipes. The V4 qualification
certificate releases no formal training or automatic successor. Integration, CPU verification and
qualification history is in the [archive](docs/operations/CURRENT_STATE_ARCHIVE_2026-10.md#modern2d-integration-and-native-qualification-cpu-accepted).

- Preserve terminal interpretations and independent qualifications through the
  history index. The accepted 500K bridge does not establish full-scale qualification.
  Any successor training, full run, extra seed or protected-role access needs
  separate authority.
- Use one immutable baseline per matching scientific contract. Track B predicts
  direct Gap on fixed PCQM4Mv2; Track A remains on repaired-2M PubChemQC.
- ETKDG must match between training and inference. Official validation was
  consumed for desktop selection; test-dev/challenge remain sealed, and any new
  protected-role access requires the owning authority's one-time record.
- GPU/DCU work follows immutable cache acceptance, atomic checkpoints,
  independently retrievable outputs, and a dated decision. Scientific failures
  close a route; unchanged-contract infrastructure failures may retry.
- Molecular-research-server access is governed by
  [`platforms/REMOTE_HANDOFF.md`](platforms/REMOTE_HANDOFF.md). The server does
  not adopt, monitor, or resume desktop-owned jobs, including the user-reported
  Kaggle1 dual500K outside server custody.

## Pointers

| Question | Authority |
|---|---|
| What is active? | This file |
| Historical state and authority map | `docs/operations/CURRENT_STATE_HISTORY_INDEX.md` |
| What happens next? | `ROADMAP.md` |
| Track meanings | `TRACKS.md` |
| Experiment directory/evidence index | `experiments/README.md` and `experiments/EVIDENCE_INDEX.md` |
| Artifact inventory | `models/README.md` |
