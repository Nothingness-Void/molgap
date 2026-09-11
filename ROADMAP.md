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
| P1 | B-DESKTOP-HANDOFF | Review frozen K1 for one full-scale execution | Explicit compute budget, optimizer-step/sample-exposure schedule, one-time official validation, and submission authority are recorded |

GraphState9 is closed: its accepted 100K efficiency gain did not survive
full-scale transfer. It is historical evidence on `archive`, not a candidate,
baseline, or restart point. The accepted desktop EdgeState continuation
remains the strongest full-scale evidence. Post-GraphState candidates and
pretraining surrogates are summarized by
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

The cardinality-preserving channel is closed by
`experiments/qm9_cardinality_channel/decision.md`. The size-only control
regressed and source content recovered most of that loss, but the candidate did
not beat the fresh EdgeState GPS9 comparator. The exact support-size/source-sum
mechanism receives no hop, placement, width, seed, optimizer, or schedule retry.

The post-failure evidence audit selected one materially different question:
replace dense atom-to-atom attention with a content-addressed multi-slot
Neural-Atom communication bottleneck. Its evidence selection and frozen
one-slot causal control are in `experiments/qm9_neural_atom_mixer/`.

That comparison completed and is closed by its decision record. Latent
replacement was efficient and directionally useful, but the four-slot arm
missed the baseline gate and barely separated from one slot. Slot count,
latent width, placement, seed, and schedule variants are therefore prohibited.
Two post-cardinality scientific attempts remain; no next task is released until
the new evidence audit names a distinct local or representational bottleneck.

`C-FOURIER-EDGE`, route 2/3, passed QM9 but its Fourier mechanism failed the
architecture-matched K1 gate on PCQM-100K and is closed. The same task exposed
a large K1-versus-full-GPS gain, consistent across target quartiles. K1 is now
frozen without modification. Its one-time independent shadow audit and fixed
500K scale bridge passed all scientific and resource gates. Repeating a
selection role, adding a seed, or tuning K1 is prohibited. The final route-3/3
attempt remains unused and parked during desktop handoff.

The K1 500K scale benchmark completed and passed on 2026-09-11 using exact
cross-platform fixed-data identity. Its decision is
`experiments/pcqm_k1_scale500k/decision.md`. Full-scale execution remains
desktop-owned and requires a separate budget and protocol; the 500K 40-epoch
count must not be copied directly onto the 6.76-times larger full role.

## Bounded autonomous Track C window

The user authorized the server coordinator on 2026-09-10 to continue the
architecture funnel while they are away. This authorization is deliberately
bounded to one remote GPU task at a time and does not broaden any data,
platform, seed, or scale boundary.

After the accepted cardinality failure, the remaining autonomous discovery
budget is exactly three new scientific routes. The Neural-Atom mixer consumed
route 1/3 and failed its frozen gate. A route consumes one attempt only when a mechanically accepted scientific
result exists; unchanged-contract infrastructure repairs do not consume an
attempt. Every scientific failure requires a written attribution and a fresh
all-history evidence audit before one successor may be frozen. If three routes
fail, stop and report; do not create a fourth.

- A mechanically accepted active-candidate result that clears every frozen gate
  authorizes the coordinator to freeze and submit exactly one paired
  PCQM-100K transfer with physical batch 128. It does not authorize a shadow
  read, another seed, desktop/full training, or official roles.
- A mechanically accepted scientific loss closes the exact mechanism. Before
  another submission, the coordinator must audit all local decision records,
  select one materially different untested pure-2D mechanism, write its
  evidence selection and frozen protocol, pass static/manifest/cache checks,
  and submit exactly one seed-42 QM9-30K paired screen.
- Infrastructure failures may be repaired and retried only when the scientific
  contract is unchanged. Every retry receives a new source identity, launch
  record, and terminal evidence.
- Closed GPS-depth/width, extra-global-attention, path-softmax, ring,
  fragment-state, PNA, directed-bond, GAPE, geometry, teacher, residual,
  fusion, and adaptive-denoising variants cannot be reopened by this window.
- Luna Max performs only low-cost mechanical polling. Each terminal state is
  handed once to this coordinator; the coordinator performs attribution and
  releases at most one successor before retargeting the persistent monitor.
- Stop rather than spend compute when no candidate has a distinct information-
  flow hypothesis, an explicit causal control, a transferable OGB feature
  contract, and a plausible path under the parameter/runtime gates.

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
