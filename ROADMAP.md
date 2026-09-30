# Roadmap - Priorities and Backlog

This file owns task order, triggers, and exit conditions. Live state is in
`CURRENT_STATE.md`; completed evidence belongs to experiment decisions, and
track ownership is defined in `TRACKS.md`.

## Reuse Navigation

First use the [operation map](docs/operations/EXPERIMENT_ADDON_GUIDE.md#pick-the-operation)
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
| P0 | Chemical auxiliary reconciliation when actionable | Query exact owning attempt, then per-arm acceptance and scientific disposition on its dedicated branch; no automatic successor |
| P1 | Pretraining evidence review / integration | Review branch-local terminal decision and uncommitted fusion supplement; retain INCONCLUSIVE and qualification gaps, no scale-up or reference retraining |
| P1 | Geometry technical closure / Git routing | Resolve reference/readiness binding and supported cost correction under owning records; preserve no-full fixed-blend decision, no new geometry run |
| P1 | Precision acceptance bookkeeping | Resolve existing adapter/reference-binding gaps on owning branch; failed speed gate does not release another run |
| Conditional | Final official test-dev submission | Explicit user authority after full-run and official-validation acceptance under the owning frozen submission contract |

Centered-logit replay exclusions are closed evidence, not an active training queue.
Server DSAR/DSMR drafts are not desktop tasks; incomplete job identity must not
be replaced with stale RUNNING claims or automatic takeover.
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
- Continued pure-2D discovery tests one materially new architecture at a time.
  Every failure gets a decision record and cannot be retried as a seed or
  schedule variation; the next attempt must change the information flow.
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
