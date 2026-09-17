# Roadmap - Priorities and Backlog

This file owns task order, triggers, and exit conditions. Live state is in
`CURRENT_STATE.md`; completed evidence belongs to experiment decisions, and
track ownership is defined in `TRACKS.md`.

## Goal

Select and validate a direct-Gap PCQM4Mv2 model using the fixed V4 comparison
contract for new screens. The desktop integration branch owns Kunshan screens,
full training, official evaluation, and submission; the server integration branch owns its separately authorized discovery.
Branch and worktree routing is defined in `BRANCHES.md`.
## Active Queue

| Priority | ID | Task | Exit condition | Owner |
|---|---|---|---|---|
| P0 | B-KUNSHAN-V4-REFERENCE | Reconcile and accept the single GPTrans-T seed-42 100K/50K V4 reference | Verify terminal scheduler state, 60 physical-BS128 epochs, runtime certificate, outputs, and frozen acceptance; do not infer status from the 2026-09-12 snapshot | Branch `codex/exp/gptrans-t-100k-v4` |
| P0 | B-FULL-CONVERGENCE | Reconcile the repaired GPTrans-T and K1 convergence chains | Inspect GPU jobs `1516700`/`1516701` and dependent acceptance jobs `1516703`/`1516704`; accept terminal artifacts or diagnose without resubmitting from a stale snapshot | Branch `codex/exp/gptrans-full-convergence` |
| P1 | B-GEOMETRY-V4-BRIDGE | Consider one 500K distance-angle candidate under a matched V4 reference | Only after the V4 reference is accepted; issue a new frozen contract/runtime certificate and explicit launch authorization | Branch `codex/exp/gptrans-k1-geometry-500k` |
| P2 | B-OGB-TESTDEV | Produce the final official test-dev submission | Explicit user authorization after full-run and official-validation acceptance | Relevant frozen submission experiment |

The V4 runtime policy and detailed baseline protocol remain on their active
experiment branch until that reference is accepted. V4 requires fixed data and
row order, seed, FP32/no-TF32, physical BS128, optimizer, schedule, loss,
selection rule, sample exposure, role access, and a runtime certificate for
each platform/software tuple.

The historical Kaggle recurrent graph-state experiment completed below its
frozen comparator and is archived at commit `285e1dc`; it is not active.

The 500K distance-angle OOF fusion is a positive nomination recorded on branch
`codex/exp/gptrans-k1-geometry-500k` at `6d8630e`. Its own audit found that
runtime certificates and a matching frozen V4 reference were absent, so it is
not yet eligible as a V4 cross-platform or causal geometry result. Keep the
branch isolated until a separately authorized V4 bridge is accepted.

Completed PCQM, OGB submission, and QM9 screen decisions are indexed in
`experiments/README.md`; do not reopen closed candidates by changing seeds or
training schedules.

## Mandatory Gates

1. **Before remote GPU submission:** local syntax/AST/manifest checks and
   immutable CPU-cache acceptance must pass. Forward/backward and memory checks
   run only in the remote GPU preflight; models are not executed locally.
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
  test-dev and challenge-test stay sealed until separate explicit authority.
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
