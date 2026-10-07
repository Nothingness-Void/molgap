# Roadmap - Priorities and Backlog

This file owns task order, triggers, and exit conditions. Live state is in
`CURRENT_STATE.md`; completed evidence belongs to experiment decisions, and
track ownership is defined in `TRACKS.md`.

## Reuse Navigation

First use the [modular workflow](docs/operations/EXPERIMENT_WORKFLOW.md), then
the [operation map](docs/operations/EXPERIMENT_ADDON_GUIDE.md#pick-the-operation)
and [local CLI](docs/operations/EXPERIMENT_CLI.md). Platform work stays in
`kaggle-molgap-workloads`, `ims-molgap-workloads` and `scnet-bw-dcu-molgap`.
Reuse existing family, analysis and acceptance owners; this is not execution authority.

## Goal and Active Queue

Select a direct-Gap PCQM4Mv2 model under the applicable frozen contract.
Desktop owns its screens/full/evaluation/submission; server discovery stays independent.
Use [BRANCHES](BRANCHES.md#retained-checkout-map) for branch-local evidence missing
from this checkout; [CURRENT_STATE](CURRENT_STATE.md) owns observations/blockers.

| Priority | Task | Exit condition / owner |
|---|---|---|
| Closed diagnostic | Complete K1 local module/state audit and sparse endpoint average | Accepted NO_TRAIN; use [consolidated route](experiments/pcqm_k1_complete_local_audit/analysis_zh.md#10-下一步一条主线两个问题分开验收) and its explicit remaining discriminators. No model adoption, new remote job or full release; existing consistency500K acceptance precedes conditional training-state/cost work. |
| K1 focus1 | Explain and reduce500K epoch cost | Freeze a bounded training-only native profile; separate loader/forward/backward/optimizer/development/checkpoint costs and matched single/two-pass work. Attribute measured seconds before an equivalence-gated execution change; no unmeasured speed claim or recipe change. See CURRENT_STATE and the mapped500K attribution owner. |
| K1 focus2 | Recover500K enhancement benefit | Complete and accept the authorized weight0/0.1 pair, compare raw and fixed BN-calibrated endpoints with aligned rows and measured native costs, then retain/remove the penalty according to its frozen gate. The two-pass pair does not isolate performance overhead; missing capacity/pretraining evidence stays unresolved. |
| Closed | K1 fixed-fusion output distillation | Both40epoch students independently finalized; weak/strong fail the frozen1meV compression criteria. Accepted evidence imported, full rejected history archived; no blind retry or scale-up. See `experiments/pcqm_k1_fusion_distillation/terminal_acceptance/decision.md` |
| Closed | K1 dropout pair acceptance and attribution | Both40epoch arms finalized VALID/COMPLETE; canonical desktop RML retains the1.5288meV positive numerical benefit below3meV. Missing control native cost and resumed provenance remain explicit strict replay exclusions; no successor or scale-up |
| Closed diagnostic | K1 fixed equal-blend transfer | Common-cohort frozen inference supports fusion nomination with measured two-pass cost; the separate distillation pair is closed below its compression gate, no model promotion. See `experiments/pcqm_k1_consistency_fusion_transfer/terminal_decision.md` |
| Closed | Chemical auxiliary custody and canonical discovery | Trained arms and summary-only CPU terminal dispositions retained; strict qualification gaps remain unknown, no new training |
| Conditional | Final official test-dev submission | Explicit user authority after full-run and official-validation acceptance under the owning frozen submission contract |

Centered-logit replay exclusions are closed evidence, not an active training queue.
Server DSAR/DSMR drafts are not desktop tasks; incomplete job identity must not
be replaced with stale RUNNING claims or automatic takeover.
Old pretraining/geometry/precision qualification work is administratively paused
in archive, not an automatic active queue. See [custody routing](BRANCHES.md#retained-checkout-map);
reopen only with explicit direction, preserving original scientific blockers.
Completed evidence is indexed by [experiments](experiments/README.md) and
[RML](research_memory/README.md); review canonical decisions/attribution before
another module. Do not reopen a closed candidate through seed/schedule changes.

New screens follow the [V4 reference](experiments/pcqm_gptrans_t_100k_v4/README.md)
and [execution contract](platforms/V4_EXECUTION_CONTRACT.md): fixed data/row order,
seed, FP32/no-TF32, physical BS128, optimizer, schedule, loss, selection, exposure,
roles and runtime certificate for each platform/software tuple.
Historical geometry, pretraining, full fusion and precision work retain their
separate frozen contracts; this navigation cleanup does not amend them.

## Mandatory Gates

1. **Before remote GPU submission:** local syntax/AST/manifest checks and
   immutable CPU-cache acceptance must pass. Authorized local diagnostics may
   execute models within their declared data/role/resource scope; they do not
   replace the owning contract's remote GPU runtime or memory qualification.
   New research diagnostics still require a prospective trajectory first.
2. **Before external evaluation:** the standalone full-scale candidate and its
   aligned predictions must be complete and frozen.
3. **Before production promotion:** the fixed Track A external gate must pass;
   then public loader, registry, hashes, latency, and smoke tests are updated in
   one production decision.
4. **On failure:** write a dated decision beside the experiment evidence and
   close the branch. Do not move the failure into `CURRENT_STATE.md`.

## Operating Rules

- Do not modify the production registry while Track B is screening.
- Architecture claims use random initialization; no pretraining, warm start,
  fine-tuning, or distillation may be credited as an architecture gain.
- Do not tune on common/OOD/P8-hard or sealed data.
- New PCQM screens use the accepted official-train identities and the V4-frozen
  100K train / 50K development roles. Historical Kaggle runs retain their
  original 100K/10K contract and are not silently relabeled V4. Do not rebuild
  caches or redefine roles. QM9 and PubChemQC are inspiration only, not
  advancement gates or reusable weights.
- The authorized full K1/GPTrans-T chain uses IMS only under
  `/lustre/home/users/sm2/chou/`; follow `platforms/REMOTE_HANDOFF.md` for every
  command and preserve its declared dependencies and output isolation.
- Architecture screens predict Gap directly. Any separately authorized full
  K1/GPTrans-T fusion must obey its own frozen 20/80 official-valid contract;
  further test-dev/challenge access requires separate explicit authority.
- Continued pure-2D discovery tests one materially new architecture or explicitly
  authorized training objective at a time. The FLAG question preserves the K1
  architecture and changes the supervised embedding objective; its prospective
  and resource/endpoint stop rules live on the mapped experiment owner.
  Every failure gets a decision record and cannot be retried as a seed or
  schedule variation; a new question must change information flow or declare a
  distinct learning constraint with evidence review and a cheapest falsifier.
- Do not rerun the rejected `0.10 eV` frozen-2D plus dual-SchNet residual.
- Geometry paths must preserve ETKDGv3+MMFF train-inference consistency.
- Router, MoE, OOF gain labels, and dataset replacement remain closed unless a
  new question and stop rule are added here first.
- Invalid molecules remain visible with reason codes; do not silently filter.
- Historical v3 Delta/UQ outputs must not be described as calibrated for the
  repaired-2M production model.

## Delivery Queue

These Track A delivery tasks remain valid but do not override the active Track B
experiment unless the project objective changes.

| ID | Task | Trigger |
|---|---|---|
| P10.2 | Batch SMILES to B3LYP CSV with provenance and rejection reasons | Track A model bundle remains frozen |
| P10.3 | Element, molecular-weight, and topology applicability gates | Before database generation |
| P10.4 | Reproducible disagreement-based OOD screening signal | Before database generation; never label it calibrated UQ |
| P10.5 | Layered real-capability sounding | Before public accuracy claims |
| P10.6 | Curate the commercial-molecule universe | 10K pilot and inference contract pass |
| P10.7 | Build the versioned B3LYP property database | P10.2-P10.6 complete |
| P11.1-P11.3 | Package, expose, and document the database | P10 exit gate passes |

## Conditional Queue

| Task | Trigger |
|---|---|
| PairGPS2D sealed-test disposition | Explicit authorization to reopen the independent branch after its validation-only decision; arithmetic equivalence must be established before using benchmark-selected TF32 for an accuracy claim |
| Experimental solid-state Delta head | A specific experimental target is requested |
| Extend the supported element set | Rejected-use analysis justifies refetch and retraining |
| Conformer ensemble or NNP geometry | Residual evidence identifies geometry as the limiting factor |
| Paper figures and write-up | An academic delivery is requested |
