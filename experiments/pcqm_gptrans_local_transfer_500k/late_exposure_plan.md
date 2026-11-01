# Matched late-exposure pair — planning proposal

Recorded on 2026-10-08 (Asia/Tokyo). **PLANNING ONLY**: no new source release,
prospective training transaction, platform submission or monitor activation.
This proposal does not modify the accepted short-budget protocol or outcome.

Supersession recorded on 2026-10-08: desktop's completed60-pass local package
was subsequently inspected. The [revised contribution proposal](late_contribution_plan.md)
reuses that result only after qualification and fills the missing matched
no-local control. The24-pass pair below is preserved as an unexecuted proposal,
not the selected follow-up recipe or release authority.

## Question and evidence

Does the uncapped real-bond local stream retain a material advantage over G1
under a prospectively matched longer-training recipe?

The [accepted bridge](gpu/results/interpretation.md) retained a 0.00565014 eV
selected/final advantage after 46,860 updates, approximately 12 fixed500K
passes. Both models selected the final observation. The gain had decreased
from 0.01457693 eV at 7,810 updates. Longer-budget retention is unresolved;
these observations do not prove an exposure deficit or eventual reversal.

Related closed routes do not justify another parameter or control grid:
[node/FFN expansion](../pcqm_gptrans_capacity_nodes_100k/gpu/results/interpretation.md),
[mean-readout replacements](../pcqm_gptrans_readout_100k/gpu/results/decision.md),
and [local amplitude cap](../pcqm_gptrans_local_control_100k/gpu/results/interpretation.md).

The official [GPTrans-T configuration](https://github.com/czczup/GPTrans/blob/main/configs/pcqm4mv2/gptrans_tiny_pcqm4mv2.yaml)
uses 300 epochs. This establishes a recipe difference, not a causal explanation
of the local accuracy gap or a recommendation to reproduce its compute budget.

## Two physical arms

| Arm | Architecture | Parameters | Proposed device |
|---|---|---:|---|
| A: new long-budget reference | Frozen G1 degree-scaled GPTrans | 5,246,817 | T4 worker 0 |
| B: long-budget candidate | Same core plus the accepted uncapped 64-channel real-bond local stream | 5,871,201 | T4 worker 1 |

Both start from their retained seed42 **initial tensors**, with the shared core
preserved. Neither starts from a selected or final short-budget checkpoint.
No architecture, width, depth, cap, readout or local-stream change is introduced.

A is a one-time reference for a genuinely new exposure/schedule/selection
contract, not a repeat of the completed short-budget reference. If a retained
reference already matches the complete new recipe, qualify and reuse it instead.
The old 12-pass scalar/bundle cannot be the terminal comparator for 24 passes.

## Proposed common recipe

Reuse the bridge's exact fixed500K cache, train membership `[0,500000)`, full
internal-development membership `[500000,550000)`, OGB input ABI, train-only
portable target transform, physical BS128, drop-last, accumulation1, seed42,
FP32/noTF32, deterministic execution, AdamW single group, lr0.001, wd0.05,
clip1, dropout/drop-path0.1 and primary EMA0.999 update semantics.

- Exactly **93,720 updates / 11,996,160 presentations**, approximately 24 passes.
- Continuous permutations seed42+cycle; preserve drop32 per full500K cycle.
- 120 optimizer segments of 781 updates. Segment is not a full dataset epoch.
- Warmup remains four segments / 3,124 updates. Cosine spans the new 120-segment
  horizon, reaching 1e-6 in segment119, using the original pre-segment ordering.
- For zero-based segment `s`: warmup LR is `0.001*(s+1)/4` when `s<4`;
  otherwise `1e-6+(0.001-1e-6)*(1+cos(pi*(s-4)/115))/2`.
- Evaluate after each pair of segments: 60 fixed observations at updates
  `1562*k`, `k=1..60`. Both arms use exactly the same selection opportunities.
  Preserve live and both existing EMA views; only EMA0.999 selects the endpoint.
- Select minimum primary EMA MAE among those 60 observations. Save selected and
  fixed-final predictions separately, including unique source indices/targets.
- Atomic full resume state after every segment; independently retrievable
  checkpoint chunks at least every ten segments. Freeze all chunk/input hashes.

This is a **new long-training recipe**, not an unchanged continuation. Its first
46,860 updates have a different LR trajectory from the old short schedule.
Compare A and B within the new recipe; do not attribute old-to-new changes solely
to exposure. An isolated exposure-versus-schedule study would need more controls
and is outside this proposal. Do not extend only B or restart cosine mid-run.

## Prospective interpretation and stop rules

Retain the historical 0.003 eV material gate for this proposed paired screen.
A retained-benefit signal requires selected gain and fixed-final gain both at
least 0.003 eV, final-ten-observation mean gain at least 0.003 eV, and the
candidate-minus-reference paired-row 95% interval upper bound below zero.
Use the existing 5,000-resample / seed20260912 saved-prediction analysis owner.
Training stochasticity remains unmeasured; row bootstrap cannot establish it.

Report matched-step curves, live/EMA gaps, selected versus final scores, and
early versus late gains. Do not substitute an early favorable checkpoint for
an adverse final retention result. No data-dependent early-stop rule is assumed
calibrated. A budget stop yields partial/MATCHED_PREFIX evidence, not a terminal
superiority or complete training-Replay claim.

If the gain remains material, retain for an explicitly reviewed desktop package;
do not automatically release full training or another seed. If the gain erodes,
stop expansion of this recipe and analyze its retained curves. If both curves
still improve, convergence remains unresolved; do not automatically extend.

## Cost and execution qualification

The bridge measured 10,011.94 s training, 4,263.31 s evaluation, and 14,447.54 s
total wall. Doubling training while retaining 60 evaluation points projects
approximately 6.8 h for B before concurrency/IO uncertainty. This is an estimate,
not an observed dual-worker throughput or a quota measurement.

Propose **7–9 wall hours / 14–18 allocated T4-hours**, with a hard ceiling of
10 wall hours / 20 allocated T4-hours. Queue/provisioning/billed quota remain
separate. Verify account ownership and available quota before any release.

Use one T4x2 notebook with independent model/process/RNG/optimizer/sampler/EMA
and directories; set one CUDA-visible GPU per worker before CUDA import. Actual
dual-worker preflight must cover host RAM, loader contention and evaluation,
not merely single-device VRAM. Require at least 15% device and host-memory
headroom and a budget-feasible whole-notebook estimate. If qualification fails,
hold release; do not silently change BS, precision, rows or recipe.

## Reuse, missing support and release boundary

Reuse `gptrans_scale_ema.train/rung_indices`, the native GPTrans state/optimizer,
source preparation/check-release, exact-version retrieval, saved-output paired
analysis, acceptance projection, and shared RML finalization. Parameterize only
the future segment horizon, observation cadence, scheduler and exposure checks;
preserve the old60-segment defaults and accepted artifacts unchanged. Add focused
sampler/LR/resume/selection and acceptance regressions before release.

The shared `ExperimentSpec.prospective.same_run_replay` can freeze a joint
reference/candidate pair and close the reference first. It **does not waive**
the server release helper's actual-reference-evidence requirement. A completed
24-pass reference is not present in the short-budget bridge. Resolve a supported,
explicitly authorized new-reference calibration/paired-release route before
compute; do not fabricate a reference bundle, use nonexistent terminal pointers,
or pass the mismatched short-budget bundle through the strict gate. If no valid
release route qualifies, the proposed parallel training remains pending.

Freeze real source, initial tensors, common recipe, joint reference identity,
runtime/role/trace/cost plans and budget before a POST. No ACTIVE trajectory is
published by this planning document. Observed STRICT_CAUSAL and Replay-Ready
are earned only after actual bound artifacts and both pool admissions verify.

No new models run locally. No official validation, test-dev/challenge, extra
seeds, geometry, teacher, Router, desktop custody or automatic successor is
included. Reuse existing Luna B only after a reconciled job/version: healthy
status silent, one durable terminal/fault event to A, then pause or close.

## Deferred architecture leads

A persistent true-bond memory could test whether local communication needs its
own evolving state, rather than more node width. GPTrans already evolves pair
states, so redundancy and local-state connectivity need a cheap static/frozen
audit before a training proposal. The accepted local result alone does not
qualify this new mechanism.

A connected real-pair return could address the documented final Pair Transition
disconnection. A repaired connection would be a new mechanism, not a justified
retry of the failed configuration; demonstrate readout/gradient connectivity and
incremental information first. Neither deferred lead occupies a third arm.
