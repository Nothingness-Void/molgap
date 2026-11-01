# Late-training local contribution — revised comparison proposal

Recorded on 2026-10-08 (Asia/Tokyo). **PLANNING ONLY**: no source release,
prospective training transaction, platform submission or monitor activation.
This proposal supersedes the [24-pass proposal](late_exposure_plan.md), without
changing any accepted historical experiment or taking custody of desktop jobs.

## Evidence that changed the plan

The [server bridge](gpu/results/interpretation.md) measured local-stream benefit
after 46,860 updates, approximately12 fixed500K passes. Desktop subsequently
completed the original G1 plus uncapped real-bond local stream at60 full passes
in its [package study](https://github.com/Nothingness-Void/molgap/blob/031a890b4d604fc4614cde2da10e6e76cf2f836e/experiments/pcqm_k1_gptrans_package_transfer_500k/terminal_acceptance/decision.md).
Its [saved-prediction analysis](https://github.com/Nothingness-Void/molgap/blob/031a890b4d604fc4614cde2da10e6e76cf2f836e/experiments/pcqm_k1_gptrans_package_transfer_500k/terminal_acceptance/analysis.json)
reported local EMA0.999 selected MAE0.10188769014835358eV at one-based epoch50;
final MAE0.1022721454501152eV. The local package's parameter count, initial
tensors and core/local implementation match the server model. Entire runner
and scientific-recipe equivalence have not been qualified.

Desktop's other package was pretrained K1 with a different objective, optimizer
and target transform. It is not a G1 no-local control. Desktop qualified fixed
equal-blend complementarity, not local-module causality or strict component
Replay. Those outcomes remain unchanged. The separately fetched K1 consistency
ablation reached only23/60 passes and cannot supply a terminal reference.

The missing question is therefore: **does local communication still improve
the endpoint after an exactly matched60-pass training budget?** Repeating the
local model at24 passes or adding another module would not answer it.

## Two scientific arms; only one proposed new training stream

| Arm | Architecture | Parameters | Proposed execution |
|---|---|---:|---|
| A: removal falsifier | Degree-scaled G1, without the local stream | 5,246,817 | New seed42 training under the exact matched60-pass recipe |
| B: retained comparator | The original G1 plus all12 uncapped64-channel real-bond branches | 5,871,201 | Reuse desktop's completed outputs, only after explicit evidence qualification |

These are two comparison arms, not permission to train two new encoders.
Reference orientation in RML is B; A is the architecture-removal candidate.
Shared core initial tensors must match exactly. A starts from the retained G1
initial state, not a selected checkpoint, cropped trained local model or
warm-start. B is not resubmitted, extended, monitored or adopted by server.
There is no third candidate, cap variant, width/readout change or new module.

## Retained-reference qualification is the first release prerequisite

Enroll B additively for this new, prospectively declared comparison; never
rewrite its original noncausal package study as a causal experiment. Before A
receives compute, require explicitly reviewed evidence transfer and actual
repository-local evidence qualification through the existing release gate:

- Real selected/final aligned predictions, exact50K source indices and targets,
  selected weights, model/source manifests,60 observations and terminal acceptance.
- Frozen architecture and initial core tensors; source differences must be
  reviewed for optimizer, sampler, RNG, EMA, metric and resume semantics.
- Actual recipe, row order, target-transform content/schema/canonical identity
  and byte SHA. The desktop recipe and observed scientific contract use different
  transform-fingerprint representations; verify their underlying asset rather
  than treating either string as proof of equality.
- Applicable role history and runtime calibration; real pointer resolution,
  SHA verification and reference/result/self-trajectory bindings.
- Native allocation/cost provenance with all unavailable historical costs
  explicitly preserved. Desktop's observed versions2–5 subtotal is not complete
  historical allocation; no invented version1 cost or budget certification.
- Actual reference eligibility and the chosen comparison class, including any
  remaining completeness limitations, verified by the owning V5/RML validators.

Git metadata and a scalar score alone are insufficient. Desktop binary paths
are not assumed to exist on this machine. If B cannot qualify for the intended
strict reuse, **hold this release** and report the missing evidence. Do not
retrain B automatically, change the benchmark to evade the gate, or promise
complete Replay admission from retrospective metadata alone.

## Proposed matched60-pass recipe

Freeze executable values and hashes after qualification, not only these labels:

- Accepted fixed500K manifest
  `630d30046d6cdc1f91fb169cd1eb4720bd5b352dc1ebeb641a11ead1ebae9751`;
  train `[0,500000)`, internal development `[500000,550000)`.
- The same consumed OGB input tensors and degree scaling0.0897. Retaining RWSE
  in a cache does not authorize introducing RWSE consumption into this model.
- Seed42, FP32/noTF32, physicalBS128, accumulation1, deterministic execution,
  drop-last32, and identical epoch-wise seed42+epoch permutations and loader RNG.
- Exactly60 full passes,3,906 updates/pass,234,360 updates and29,998,080 sample
  presentations. These are not the old60 short observation rungs.
- AdamW single group, lr0.001, wd0.05, foreach=false, fused=false, clip1;
  normalized Gap-L1, dropout/drop-path0.1 and the retained100K train-only target
  transform. Transform file SHA:
  `0bbe2b4378afe85fe4fab0a197e0f4b4ec42c5bfb24d58e17576656e42d4a026`.
- Match desktop's actual warmup4/cosine60/min1e-6 and scheduler ordering:
  warmup spans15,624 updates, not the server bridge's3,124.
- Primary EMA0.999 after every update. Select minimum EMA development MAE over
  the same60 end-of-pass opportunities. Keep live observations, independently
  selected endpoint, fixed-final endpoint and the complete checkpoint lineage.

The architecture-local stream is the only declared scientific intervention.
Differences needed only for transport/staging must preserve and verify training
math, sampler/RNG progression and selection. No new patience rule, LR restart,
target-stat refit or selective continuation is allowed.

## Readout of the experiment

Define local contribution as `MAE(A no-local) - MAE(B local)` within this new
matched comparison, not by subtracting old12-pass and desktop60-pass scores.
Report selected, fixed-final and final-ten-pass mean contributions, matched-step
curves, train/dev gaps and live/EMA gaps. Use the retained paired-row analysis
recipe:5,000 resamples, seed20260912. Row uncertainty is not seed uncertainty.

Propose the same0.001eV directional research filter as the server500K bridge:
selected contribution at least0.001eV, local-minus-no-local paired
interval upper bound below zero, and positive fixed-final and late-mean
contributions. Freeze this before compute; do not rewrite older gates.
RML keeps its declared orientation: a positive local contribution means removal
candidate A is worse than retained reference B, not a promoted new candidate.

- Sustained positive contribution supports retained local information beyond
  early acceleration, for this seed/recipe/cohort only.
- Early positive but negligible late contribution supports an acceleration-only
  interpretation for this recipe, not a claim that locality is universally useless.
- Reversal closes expansion of this local package under this matched recipe.
- Improving final curves leave convergence unresolved; no automatic extension.

None of these outcomes releases full training, official validation, extra seeds,
fusion optimization or an automatic successor. Comparing against older EdgeState
official scores is contextual, not an architecture claim from this experiment.

## Time, staging and reuse

Desktop's local arm measured54,074.407s of epoch windows, approximately15.02h
for60 passes. A has fewer operations, but its actual throughput is unmeasured.
Use **12–16 active wall hours** as a planning range, not an overnight guarantee.
One trained worker on a two-T4 allocation still costs both allocated devices:
approximately24–32 allocatedT4-hours, plus bounded preparation/acceptance costs.
This is a proposed budget, not released compute or confirmed account quota.

Plan durable stages at passes30 and60, each provisionally capped at8 wall hours.
Qualify the first stage's measured feasibility before release. Stage1 is partial
evidence, never terminal superiority or complete training Replay. Publish its
atomic model/optimizer/scheduler/EMA/all RNG/sampler cursor and independently
retrievable chunks before any covered stage2 continuation. Keep the original
60-pass LR clock, source and initial lineage across stages. Budget exhaustion
closes as partial; it does not silently shorten the scientific contract.

Reuse `gptrans_capacity`, the owning native GPTrans optimizer/EMA/scheduler,
500K role loading, shared source/check-release, receipt reconciliation, chunk
retrieval, saved-output comparison and terminal/RML owners. The server's
`gptrans_scale_ema` hardcodes781-update rungs and cannot execute this recipe
unchanged. The smallest needed hook is an explicit full-pass horizon/sampler
and stage cursor with compatibility tests preserving old defaults, plus the
qualified retained-reference acceptance binding. Do not copy desktop's whole
K1/GPTrans package runner or manufacture a new platform submitter.

Preparation checks must cover exact data/initial/transform/source identities,
full-pass LR and presentation counts, loader RNG and uninterrupted-versus-resumed
equivalence, EMA selection, real reference release validation, partial-stage
classification and candidate/reference Replay admission. Only remote authorized
preflight may execute model math; no local model training/inference is needed.

Use T4 only; justify one trained worker, isolate its visible device, and record
the full allocation including idle peer. Reuse existing Luna B only after an
actual server job/version is reconciled. Healthy status is silent, terminal or
actionable failure produces one durable event. No desktop live operations occur.

## Planning boundary

No executable protocol, prospective hashes, ACTIVE RML trajectory, remote job,
new reference qualification or test result is created by this document. Training
requires a subsequent explicit compute decision and passing actual release gates.
No protected-role access, model execution, source integration or submission was
performed for this revision.
