# Roadmap - Priorities and Backlog

> This file owns task order, triggers, and exit conditions. Live truth is in
> `CURRENT_STATE.md`; methods, metrics, and conclusions live in dated records.

## Goal

Select one Gap-only method for the official PCQM4Mv2 leaderboard. New
architectures use the three-stage funnel: cheap QM9-30K triage, paired
PCQM-100K transfer with a once-read shadow audit, then an explicit
desktop/full-scale decision. Training-only optimization of an already validated
inference architecture may enter directly at paired PCQM-100K when the user
records that classification. All
geometry must be ETKDG-consistent. The default full-run ceiling is 12 A100
hours unless the user records a separate override.

## Active queue

| Priority | ID | Task | Exit condition |
|---|---|---|---|
| P0 | B-ESGPS6-304-500K | Run the compute-matched paired Kunshan scratch60 vs relation20+Gap40 test at batch 128 | Both arms complete from one accepted cache, or terminal infrastructure failure is diagnosed; report paired delta without opening official roles |
| P1 | C-EVIDENCE-REVIEW | Re-rank only untested mechanisms after the adaptive-denoising closure; SCNet work remains out of scope | One evidence-selected question and frozen protocol, or an explicit stop decision |
| P2 | B-SHADOW-AUDIT | Audit only a PCQM-100K method that clears its material nomination gate | Shadow direction agrees; no scale-up is automatic |
| P3 | B-DESKTOP-HANDOFF | Hand exactly one strong transfer to `molgap-desktop` | Explicit full-run budget, one-time official validation, and submission authority are recorded |

GraphState9 remains the accepted 100K efficiency discovery, while the accepted
desktop EdgeState continuation remains the strongest full-scale evidence.
Post-GraphState candidates and pretraining surrogates are summarized by
`results/architecture_failure_attribution_2026-09-08/decision.md`.

The sparse relative-value question is closed by its paired seed-42 decision.
The replacement funnel and its evidence boundaries are frozen in
`results/three_stage_screening_reset_2026-09-08/decision.md`.

The EdgeState local-hierarchy 10/30 allocation produced a strong paired
`0.00234205 eV` gain but did not clear its predeclared `0.003 eV` nomination
gate. That optimization question is closed; no shadow, further allocation,
extra seed, or scale-up follows. Its decision is
`results/local_hierarchy_allocation10_30_seed42/decision.md`.

Matched GAPE-lite failed its same-task batch-128 QM9 gate against both fresh
and equal-compute controls. Its exact generator, noise, width, seed, and
pretraining allocation are closed in
`experiments/qm9_gape_pretraining/decision.md`.

Adaptive atom-local denoising is closed by
`experiments/qm9_adaptive_denoising/decision.md`. It improved numerically but
missed both frozen promotion margins, so it receives no PCQM transfer, another
seed, or noise-parameter variant. Any further Track C admission requires a new
evidence review and a materially different mechanism. SCNet/Fragment-State
work belongs to desktop and is ignored here.

## Mandatory gates

1. Before remote GPU submission, syntax/AST/manifest checks and immutable
   CPU-cache acceptance must pass; model execution remains remote.
2. A bounded architecture screen cannot directly authorize full-data work.
   New promotion must pass the resource-bounded ladder in
   `experiments/pcqm_gap_architecture/scale_up_protocol_v2.md`. Every new 50K or
   100K training screen uses physical batch exactly 128 per model and device.
   Stage-specific high-batch optimization is frozen before S1, and candidate
   and baseline match exactly within every stage.
3. Before external evaluation, the matched full-scale candidate, delivery
   baseline, and aligned outputs must be complete and frozen.
4. Before production promotion, the fixed Track A external gate, loader,
   registry, hash, latency, and smoke checks must pass together.
5. Every failure receives a dated decision and is closed; it is not rewritten
   into `CURRENT_STATE.md`.
6. Infrastructure-only failures may be diagnosed, repaired, and retried with a
   new remote version. The architecture, data roles, split, target, seed,
   optimizer, schedule, precision, and sealed-role flags must remain unchanged.

## Operating rules

- Do not modify the production registry while Track B is screening.
- Random-initialized architecture claims never use pretraining, warm start,
  distillation, residual targets, or prediction fusion. Separately labeled
  pretraining questions require their own scratch and equal-compute controls
  and cannot retroactively change an architecture conclusion.
- Track C QM9 screens are direct-Gap, one-seed paired triage only. A win grants
  no PCQM or full-scale conclusion. Inputs use the transferable OGB categorical
  atom/bond and RWSE contracts; geometry questions use ETKDG, not QM9 DFT
  coordinates.
- Only arms from the same task and platform under
  `experiments/SCREENING_POLICY.md` may decide a screen. Cross-task,
  cross-platform, or non-128 metrics remain context evidence only.
- Track B uses the frozen official-train-derived 100K/10K selection contract.
  Before the first post-reset transfer, freeze one disjoint train-derived
  shadow role; read it only once after selecting the candidate. Never tune on
  common/OOD/P8-hard, the shadow role, or sealed official roles.
- Keep the main databases unchanged: Track B stays on PCQM4Mv2 and Track A
  stays on the existing repaired-2M PubChemQC corpus. External datasets may
  support an explicitly labeled audit or OOD evaluation, but may not replace
  or silently augment the main training database. Privileged geometry,
  teacher models, and teacher distillation are prohibited for the OGB
  leaderboard line.
- The server discovery agent does not launch full-scale training. A strong
  Track B transfer is handed to desktop for a separate budget decision; any
  later molecular-research-server access remains restricted to
  `/lustre/home/users/sm2/chou/`.
- Predict Gap directly. Track A HOMO/LUMO experiments do not authorize Track B.
- Test one material mechanism at a time. Scientific failures are not retried
  as seed, width, distance, optimizer, or schedule variants.
- Compare every scale directly with the same delivery baseline. Chained
  architecture improvements and historical mismatched checkpoints are not
  scale-up evidence.
- Preserve invalid-molecule reason codes and ETKDG train/inference consistency.
- Router, MoE, dataset replacement, ordinary late fusion, and the old
  dual-SchNet residual remain closed unless a new question is recorded here.
- Remote monitoring is mechanical only: one Luna Max heartbeat per persistent
  monitor thread; terminal evidence is handed to the coordinator for analysis.
- Continuous Track B architecture search is closed. A monitor may collect and
  hand off terminal evidence, but it never selects or submits a successor.
  Each Track C question and Track B transfer needs a frozen protocol and queue
  entry. Seeds 43/44 require a separate shortlist and compute-budget decision.

## Conditional backlog

| Question | Trigger | Bounded action |
|---|---|---|
| Sparse non-covalent ContactState | Closed at seed 42 | Do not retry cutoff, width, depth, seed, or optimizer variants |
| Compact Cartesian invariant body-order basis | Closed at seed 42 | Do not retry width, radial count, seed, optimizer, or schedule variants |
| PairGPS2D sealed-test disposition | Explicit authorization | Establish arithmetic equivalence before any benchmark-selected precision claim |
| Alternative deployment-time geometry | A rule-compliant source available identically at training and inference is approved | Compare one frozen source without privileged teachers or sealed-role geometry |
| Local hierarchical pretraining beyond MolCHG-lite | MolCHG-lite transfers through PCQM-100K | Change one supervision level at a time with equal-compute scratch |
| Solid-state Delta head | A separate target is requested | Create an isolated target contract |
| Paper figures/write-up | Academic delivery is requested | Derive figures only from accepted decision records |

Track A delivery work remains in its own records and does not override the
active Track B queue. Closed-route indexes are
`experiments/_closed/pcqm_server_archive_index.md` and
`experiments/_closed/qm9_top20_archive_index.md`.
