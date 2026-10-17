# Roadmap - Priorities and Backlog

This file owns task order/triggers/exits. CURRENT_STATE.md owns live state,
experiment decisions own evidence, TRACKS.md owns tracks.
## Reuse Navigation

First use the [modular workflow](docs/operations/EXPERIMENT_WORKFLOW.md), then
the [operation map](docs/operations/EXPERIMENT_ADDON_GUIDE.md#pick-the-operation)
and [local CLI](docs/operations/EXPERIMENT_CLI.md). Platform work stays in
`kaggle-molgap-workloads`, `ims-molgap-workloads` and `scnet-bw-dcu-molgap`.
Reuse existing family, analysis and acceptance owners; this is not execution authority.

## Goal and Active Queue

Select a direct-Gap PCQM4Mv2 model under its frozen contract; desktop and server remain independent.
Use [BRANCHES](BRANCHES.md#experiment-custody-index) for branch-local evidence missing
from this checkout; [CURRENT_STATE](CURRENT_STATE.md) owns observations/blockers.

| Priority | Task | Exit condition / owner |
|---|---|---|
| Immediate | Repository cleanup and environment/test repair | User-directed maintenance pauses new experiments. Preserve canonical evidence, unresolved owner work and durable artifact custody; validate scoped changes before integration. Existing jobs are not cancelled. |
| Reconciliation | Existing desktop-owned jobs and terminal acceptance | Follow exact identities and blockers in [CURRENT_STATE](CURRENT_STATE.md); query scheduler and verify durable artifacts before any continuation decision. No automatic retry or successor. |
| Terminal custody | K1 execution pair, original seed42 cost/quality, partial single/EMA500K, spectral/EMA100K | Finish evidence-supported acceptance/routing on the [owning branch](BRANCHES.md#experiment-custody-index). Preserve failed gates, unknown cost/provenance and strict replay exclusions. No new training release. |
| Conditional | New K1 cost/500K enhancement question | Cleanup is accepted and the user releases new work; query RML and canonical attribution, define a genuinely new question, prospective scope and cheapest falsifier. No baseline retraining by default. |
| Conditional | Final official test-dev submission | Explicit user authority after full-run and official-validation acceptance under the owning frozen submission contract. |

Closed experiments and accepted diagnostics belong in [experiments](experiments/README.md)
and [RML](research_memory/README.md), not the active queue. Review canonical
attribution before selecting related work; seed/schedule changes do not reopen a
failed question. Archived pretraining/geometry/precision qualification remains
paused with original blockers. Server DSAR/DSMR drafts are not desktop tasks;
unknown job identity cannot justify takeover or a RUNNING claim.

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
   route its branch under BRANCHES after terminal acceptance. CURRENT_STATE may
   summarize a remaining blocker; the decision owns the evidence.

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
  authorized training objective at a time.
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
