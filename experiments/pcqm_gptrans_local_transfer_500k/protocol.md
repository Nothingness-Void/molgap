# Equal-update500K local-stream bridge

Planning scope recorded on 2026-10-07. Status and release authority belong to
`CURRENT_STATE.md`; this protocol is not a submission receipt or a validated
prelaunch. The user's subsequent Kaggle2 submission request on 2026-10-07
authorized execution of this bounded plan subject to the real release gates.
The release/response records, not this prose, establish any actual submission.

## One question and one intervention

Does the original real-bond local stream preserve a practically useful advantage
over G1 after genuine fixed500K optimization, without changing the update budget?
The only architecture intervention is the accepted64-channel real-bond stream
before each of12 GPA blocks. There is no amplitude cap or additional mechanism.

Hypothesis: explicit local bond messages supply useful information that remains
valuable when the optimizer sees more distinct molecules. Alternatives: the
addon merely accelerates early fitting; later propagation erodes the benefit;
or the extra degrees of freedom improve fitting without generalization.

Evidence, including limitations, remains owned by:

- [Local discovery and equal-update scale study](../pcqm_gptrans_capacity_relations_100k/gpu/results/interpretation.md).
- [Frozen cohort / derivative audit](../pcqm_gptrans_bottleneck_audit/decision_terminal.md).
- [Failed amplitude intervention](../pcqm_gptrans_local_control_100k/gpu/results/interpretation.md).

The100K local result did not pass its historical material gate. This bridge
would test a distinct, separately released research question; it would not
retroactively promote that result or reopen the rejected cap route.

## Two scientific arms, no duplicate baseline by default

| Arm | Model | Parameters | Execution preference |
|---|---|---:|---|
| A | G1 degree-scaled GPTrans, primary EMA0.999 | 5,246,817 | Enroll and freeze the retained same-budget500K result after qualification |
| B | The same G1 core with the original uncapped real-bond stream | 5,871,201 | One from-scratch500K training stream |

Two comparison arms do not imply two new optimizations. The
[retention audit](reference_retention_audit.json) found the old A artifacts
available and hash-valid. Reference enrollment must verify the exact500K
identity, row stream, initialization, recipe, runtime, selection and evidence
bindings through the owning release gate. It must register the primary EMA999
view explicitly rather than confuse two EMA views with two trained models.

The old study's noncausal outcome and original100K contextual reference remain
unchanged. Any new reference/self-reference trajectory binding is additive and
must use existing RML owners, actual artifacts and honest recovery semantics.
Do not invent a new complete historical trajectory or backfill absent data.

If A cannot qualify from retained artifacts, mark the release pending and name
the exact mismatch. Do not automatically retrain A. A separately approved new
benchmark may train its reference once; only in that case are A and B two new
T4 workers. Never change a contract just to justify occupying an idle device.

## Fixed recipe and scale semantics

Use the accepted cross-platform fixed500K graph cache, never a rebuild:

- Manifest SHA256: `630d30046d6cdc1f91fb169cd1eb4720bd5b352dc1ebeb641a11ead1ebae9751`.
- Training source indices: `[0,500000)`; internal development: `[500000,550000)`.
- Preserve G1 input semantics and accepted OGB features; no path-mean,3D or
  auxiliary supervision is introduced. Do not add RWSE consumption if the
  frozen GPTrans model did not consume it.
- Seed42; FP32, TF32 disabled; physical batch128; accumulation1; deterministic
  algorithms; no partial optimizer batches.
- AdamW, one parameter group, lr0.001, weight_decay0.05, foreach=false,
  fused=false, gradient clip1. Preserve the owning optimizer implementation.
- Use the original100K train-only target-transform asset:
  `../pcqm_gptrans_author_alignment/recovered_reference/target_transform.json`.
  Canonical asset SHA256:
  `9462cf73ea022b030675a7f18f2e7682eda6f90b7a42b47affc2f0f4071353be`.
  Do not refit target statistics on500K.
- Primary EMA0.999 after each optimizer step; select minimum primary EMA
  internal-development MAE over the60 fixed observation rungs. A's historical
  EMA0.9999 view is descriptive only and cannot be selected instead.
- Exactly46,860 optimizer steps and5,998,080 presentations;60 rungs of781
  updates each. Preserve warmup4 / cosine60 / minimum lr1e-6 in the existing
  rung-to-step sequence, including the owning scheduler's ordering.
- Reuse `gptrans_scale_ema.rung_indices`: continuous seed42+cycle permutations,
  3,906 complete batches per500K cycle, dropping32 rows from that permutation.
  Freeze and check the actual ordered-index fingerprint before release.

These are approximately12 data passes, not60 full500K epochs. This estimates
equal-update architecture benefit with more molecule diversity, not full
convergence. Additional updates or a different schedule need a separately
matched reference and covered decision; never selectively extend B or restart
its cosine schedule.

B consumes the retained seed42 **initial** local tensors, not its100K best
checkpoint. Frozen initialization file SHA256:
`d471d9f94ff2c10436261b7f70f3e2d9111978f1ac8aadfabea2da32a5fe9a1d`;
tensor-state SHA256:
`a65034dae2d01d82eb0074eba8f2ee697d93ca259188ae45a4213e9041d1ed1b`.
The shared core initialization must match A; each arm freezes its own complete
architecture-dependent initialization. Local output projections retain their
original zero initialization. Preserve dropout0.1 and drop_path0.1.

Never use100K internal-development rows as independent validation after500K
training: those indices now belong to the expanded training prefix. Compare
A and B on the same full50K development role, aligned by source index. Do not
subtract absolute100K and500K MAEs as a transfer estimate.

## Bottleneck observations and checkpoint interventions

Keep all60 primary canonical observations: step, presentation count, actual
pass, LR, live train/dev, EMA dev, checkpoint identity and cumulative wall cost.
Retain selected predictions and weights, final state, and complete checkpoint
chunks at zero-based rungs9/19/29/39/49/59. Preserve model, optimizer, scheduler,
EMA, all RNG states and the continuous sampler cursor atomically every rung.

Three observational layers have distinct meanings:

1. **Selection/optimization:** matched-rung live/EMA gaps and A-minus-B gains;
   selected endpoints, fixed-final endpoints and the final-ten-rung mean gain.
   These distinguish early acceleration from retained equal-budget benefit.
2. **Information flow:** at retained EMA999 rungs19/39/59, use one frozen512-row
   development panel `[500000,500512)` to measure node/pair RMS, local-update
   ratios, return/parameter gradients and readout connectivity. Use eval-mode
   normalized Gap-L1 derivatives; do not label them historical train gradients.
3. **Compute:** bounded loader/collate, H2D, forward/loss, backward/optimizer,
   validation and checkpoint/hash timings, peak allocated/reserved VRAM and
   complete allocation cost. B timings are not automatically comparable with
   A's old aggregate cost; lacking a matched timer remains a limitation.

For the diagnostic panel, compare B's unmodified prediction with three
temporary interventions: zero only the local returned update in layers1–4,
5–8 or9–12. Keep the GPA core, weights, virtual node, pair inputs and readout
unchanged. Record exact affected layers and prediction/error differences.
No intervened variant may become a selected model or another trained arm.

Save observations atomically; no extra optimizer update is permitted for
diagnostics. Check unchanged tensors, buffers and CPU/CUDA RNG; isolate probe
gradients from any resumed optimizer. Endpoint scoring uses all50K rows;
the512-row panel is only a localization sample, not representative evidence
about rare, large or OOD molecules. Removing a trained branch may itself be
off-distribution: intervention sensitivity is dependence, not proof that
retraining without that branch would improve MAE.

Frozen-checkpoint probes must be a separate linked NO_TRAIN action after
training acceptance, with their own manifest, role/cost records and truthful
comparison class. A's existing checkpoints may be reused without retraining.
Reserve the bounded diagnostic allowance before release; do not improvise an
unplanned follow-up or call this diagnostic a causal training Replay pair.

## Proposed decision criteria, frozen before any future release

Define gain as A MAE minus B MAE, in eV, using independently selected primary
EMA endpoints on the same full50K role. Also report matched-step gain and
fixed-final gain so a favorable early selection cannot hide late erosion.

A provisional **directional-transfer research signal** requires:

- selected gain at least0.001 eV;
- upper bound of the candidate-minus-reference paired-row95% bootstrap below0;
- positive fixed-final gain and positive mean gain over the final ten rungs;
- complete comparable artifacts, actual runtime qualification and native cost.

Use the existing paired-analysis owner with a prospectively frozen bootstrap
recipe/seed; do not search over resampling settings. This0.001 criterion is a
proposed practical research filter for this500K question, not a measured
training-stochasticity floor or a replacement for the historical100K gate.
Reference stochasticity is unavailable and must stay marked unavailable.
Single-seed results cannot establish seed robustness or release full training.

If only early gains survive, conclude early equal-budget acceleration rather
than a durable endpoint win. If B improves fitting but loses dev advantage,
retain the generalization-limited interpretation. Disconnected gradients can
localize an implementation limitation; large norms alone cannot prove a
harmful module. If both curves still improve at the limit, convergence remains
unresolved; do not declare an exposure deficit proved or automatically extend.
No further cap/width/LR grid follows any outcome.

## Reuse plan and unsupported gaps

| Need | Existing owner | Smallest missing capability |
|---|---|---|
| Local architecture / frozen initial loader | `gptrans_capacity.construct/load_initial` | None; preserve original source semantics |
|500K sampler / resumable update loop | `gptrans_scale_ema.rung_indices/train` | Parameterize model/initial loader and native per-arm identity; do not copy the trainer |
| Native optimizer / EMA / selection | `pcqm_gptrans_v4` | Reuse, not a new graph-screen recipe |
| Frozen derivatives | `gptrans_bottleneck.capture_layers/derivative_probe` | Narrow checkpoint/panel binding and local-return intervention hook; old hardcoded diagnostic cannot be called unchanged |
| Source, IO, receipt, native allocation | Existing `experiment_*`, `training_reproducibility`, RML trace owners | Exact new source/config/inputs and measured diagnostic overhead |
| Saved-output acceptance | Native GPTrans acceptance owners |500K local architecture comparison against the newly qualified A bundle; the old scale-EMA acceptor is noncausal and cannot grant this claim |
| Terminal / actual Replay admission | Existing RML terminal and reference owners | Qualified500K reference/self-binding and actual candidate-reference pool check |

Focused checks must cover source/initial identity, sampler and resume equivalence,
exact exposure/LR/EMA, diagnostic RNG/weight invariance, artifact ambiguity,
all25 retained hashes, release-time reference binding and actual Replay
eligibility. These are preparation requirements, not tests already executed.

## Budget, release and closure boundary

Proposed platform: server-owned Kaggle2, T4 only; no account/API/quota probe has
been performed for this planning action. Account identity, remaining budget
and exact mounts must be verified by the owning workload skill before a POST.

Preliminary allowance:4–6 notebook wall hours for training, maximum7; bounded
post-acceptance diagnostic at most0.25 wall hours. An allocation of two T4s
counts8–12 estimated training device-hours, maximum14, even if A is reused and
one device is idle. Diagnostic allocation may add up to0.5 T4-hour. These are
estimates/ceilings proposed for later approval, not measured cost or released
resources; accepted optimizer-inclusive preflight must substantiate feasibility.

If both physical arms are separately approved, isolate each model, CUDA-visible
device, process RNG, sampler, optimizer, EMA and checkpoint directory. If A is
reused, justify one trained model and record all idle allocation; do not fill
the second T4 with another hypothesis.

Before release, freeze the real reference bundle, portable transform, source
commit/archive, recipe, initial states, roles, trace/runtime plan, budget and
expected artifact inventory using existing prospective/RML APIs. Complete the
actual reference-evidence gate and `check-release`. Missing or mismatched
evidence holds release; a placeholder SHA or an asserted strict status cannot
stand in for the real verifier. This planning directory must not be discovered
as an ACTIVE training trajectory without that transaction.

At completion, independently verify row-aligned finite predictions, source,
checkpoint, runtime, trace, roles, cost, acceptance and decision bindings.
Finalize RML, rebuild derived outputs, and explicitly inspect actual candidate
and reference admission in the Replay pool. Mechanical acceptance alone is not
Replay readiness; diagnostic NO_TRAIN evidence remains separately classified.

Rebind only the existing Luna B after a real returned job/version is reconciled.
Healthy running is silent; one durable terminal/fault event wakes existing A.
Submission uncertainty is SUBMIT_UNKNOWN, never permission for blind retry.
No extra seed, automatic successor, selective extension, full-scale run,
official validation, test-dev/challenge or desktop monitoring is authorized.
