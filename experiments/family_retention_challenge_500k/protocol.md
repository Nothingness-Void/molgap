# Proposed protocol: qualified reference and complete GRIT

Design date:2026-10-28. Status: **PLANNED_NOT_SUBMITTED**.
This is a research proposal, not a frozen executable recipe or compute release.

## Questions and estimands

1. Should PPGT/ESA receive more server compute under the tested budget?
   Use their existing Common-Dev predictions/curves and a newly qualified
   strong Common-Dev reference. Historical recipe differences remain explicit;
   these portfolio contrasts are contextual, not strict causal comparisons.
2. Can a complete compact GRIT improve on that reference under one shared
   recipe and six million presentations? This is the prospective architecture
   comparison. It must pass the existing strict prelaunch and terminal gates.

Do not conflate model-family capacity, recipe suitability and resource value.
A frozen-checkpoint re-evaluation does not simulate500K retraining. This plan
uses actual500K training for the two new arms.

## One new two-arm notebook

| Arm | Architecture | Reason |
|---|---|---|
| A: reusable reference | The accepted G1 GPTrans real-bond-local architecture from the6M local-transfer experiment | Its matched500K benefit is established; avoids relying on a later30M scalar or adding unisolated modules |
| B: compact complete GRIT |8 blocks, width192,8 heads, node FFN expansion2, RRWP16, mean graph readout | Tests coupled nonlinear node/pair processing rather than a positional addon |

Arm A explicitly reuses **prior local-transfer Arm B**, mode
`degree_bond_local_ema999`,5,871,201 parameters. Its exact architecture identity
is bound in the prior [screen configuration](../pcqm_gptrans_local_transfer_500k/gpu/screen_config.json)
at `arms.scale_ema.comparison_identity.architecture_config_identity`.
This is not the plain G1 control. Preserve that architecture identity; the new
benchmark/data/selection identities must be separately frozen.

Arm B retains joint pair updates, relation-valued messages, real-degree scaling and the author's
normalization semantics. The proposed size is approximately5M, **not measured**.
Freeze an actual parameter count (ceiling6M) and immutable configuration before release.
Do not reduce batch size, drop pair updates or change width on the running job.

PPGT and ESA are not retrained. Their saved selected/final outputs are reused
only where present, aligned and accepted. Missing payloads stay missing.

## Common training and evaluation proposal

- Accepted cross-platform indexed train500K and Common-Dev50K assets; exact
  manifests, row identities, target identities and disjointness must resolve
  from existing acceptance. No platform graph rebuild or role substitution.
- Direct Gap; seed42; FP32; TF32 disabled; physical BS128; drop_last; one
  optimizer update per batch; no accumulation, pretraining, geometry or teacher.
-46,860 updates /5,998,080 presentations for both arms, with an additional
  saved checkpoint and complete prediction outputs at15,620 updates /1,999,360
  presentations. This is about12 physical500K passes, not60 physical passes.
- Reuse the accepted GPTrans6M recipe: AdamW LR1e-3, WD0.05, gradient clip1,
  one parameter group, foreach/fused disabled; warmup3,124 updates and cosine
  to1e-6 at46,860 updates. Reuse its exact train-only target-transform asset.
- Same deterministic row-order policy and evaluation cadence: every781
  updates, all50K Common-Dev rows. Both independently initialized with seed42;
  different architectures do not claim tensor-identical initial weights.
- Proposed primary selection: best-development **Live** at the fixed cadence.
  Also retain final-Live and every development observation. EMA0.999 is tracked
  identically but diagnostic-only: no switching to whichever weight view wins.
  BatchNorm buffers and their handling must be explicit and validated. No
  data-dependent post hoc BN recalibration is introduced.

The Live selection is a new common benchmark convention, not a rewrite of the
older EMA-selected reference. It avoids making the architecture verdict depend
on an unqualified cross-family EMA/BatchNorm interaction. GRIT's published
LR/WD differs; this tests applicability under the incumbent recipe, not faithful
paper performance. A negative does not authorize automatic LR searches.

Record the2M checkpoint as a **prefix of the6M schedule**, not an exact replica
of PPGT's2M cosine endpoint. PPGT-to-reference prefix comparisons remain
contextual even if row identities and exposure match.

## Input identity and reference qualification

Both arms receive one accepted topology feature bundle with the same parent
graph/row order, OGB atom/bond categories, true degree, SPD and RRWP16 fields.
Their encoders' use of those fields is the declared architecture intervention.
RRWP is deterministic CPU topology preprocessing only; it needs a hashed
sidecar keyed to parent rows and node order, not a new molecular dataset.

Check this interpretation against the existing feature-identity and
architecture-intervention validators. Do not give different features a fake
shared fingerprint, widen the validator, or silently lower the claim. If the
existing contract cannot represent the difference, freeze an honest alternative
comparison scope before release.

First search the reference registry for an already qualified exact benchmark.
If found, reuse it and cancel the proposed reference training. Otherwise A is
the **one-time reference establishment** for this new benchmark, not routine
baseline repetition. A's historical training and dev roles are not sufficient.

The existing paired workflow must explicitly support this prospective
reference-establishment design, both frozen initializations and actual evidence
bindings. If its release gate instead requires a completed reference, establish
and accept A first; B is not released using invented future artifacts. That
changes scheduling, not scientific conditions, and requires a budget recheck.
Do not create an infrastructure bypass to force simultaneous launch.

## Decision rules to freeze before any new scores

### New architecture

Let gain be A MAE minus B MAE on exactly aligned Common-Dev rows.
Proposed material margin:

`delta = max(0.001 eV, 0.01 * A MAE, accepted training-stochasticity floor)`.

The formula, not a favorable realized threshold, is frozen before training.
This is a proposed experiment-specific gate, not a replacement global V5 rule.
If the registry cannot supply a defensible stochasticity floor, any positive
result is a provisional research lead; no formal promotion or automatic seeds.

- Retain B for a subsequent bounded research decision only if selected gain
  clears delta, the paired molecule95% interval favors B, final-Live and the
  final five observations do not reverse the advantage, and resource gates pass.
- Predeclared size/degree/ring-count groups diagnose concentration of gains.
  They do not select new subgroups, checkpoints, experts or routing policies.
- Otherwise stop investment in this compact GRIT under this contract. Separate
  valid negative evidence from incomplete execution, cost stop and invalid
  comparison. No automatic extension, auxiliary loss, seed or family retuning.
- A molecule bootstrap does not estimate training-seed variability; Common-Dev
  is a reused selection role, not an untouched official generalization test.

### Existing family retention

- PPGT: park further training under the nine-hour envelope. Its continuing
  decline is retained as uncertainty; do not declare intrinsic convergence.
  Reopening requires a materially cheaper execution path or separately bounded
  long-horizon authority, not another tiny addon.
- ESA: retain only as a low-cost research reserve unless its accepted saved
  Common-Dev output is competitive with A. A practical efficiency nomination
  would require a gap no larger than delta and a separately comparable measured
  cost advantage. Historical timing alone does not meet that qualification.
- Define a separate practical margin `delta_portfolio = max(0.001 eV, 0.01 * A MAE)`.
  If `ESA MAE - A MAE >= delta_portfolio` and its paired molecule95% interval
  excludes zero on the unfavorable side, stop routine ESA architecture sweeps
  under the tested resource envelope. This is a resource decision, not a
  seed-robust scientific rejection; it does not waive the new-candidate floor.
  The recipe
  difference prevents a pure architectural cause claim. If A does not outperform
  it, do not discard ESA based on the old incompatible scalar ranking.
- Preserve implementations, checkpoints and negative/contextual evidence. No
  archive deletion or historical RML upgrading follows from this proposal.

## Resource envelope and fail-fast conditions

One T4x2 allocation, one independent arm per visible GPU. Maximum nine elapsed
hours includes install, native qualification, training, development, chunk
publication and terminal output. Log18 allocated T4-hours at the full ceiling
as physical capacity, separately from Kaggle session quota accounting.

Before formal training, reuse the bounded native train-only profile and restore
the frozen initialization/RNG/optimizer state. The projection uses whole
train+dev+checkpoint cycles, accounts for padded node-pair work and graph-size
tail, and reserves15% wall headroom. CPU topology acceptance happens beforehand.
If either arm cannot fit BS128 or its fixed endpoint within the envelope, stop
before formal training. No hidden microbatching, early truncation or extra job.
Actual GRIT throughput and parameter count are unknown until qualification.

Measure exclusive loader/H2D/forward/backward/optimizer/EMA/dev/checkpoint times,
peak allocated/reserved memory, useful graphs/s and full-allocation idle cost.
Keep profiler windows bounded and separate from clean throughput measurement.

## Reuse and minimum new work

Use the experiment front door, immutable recipe/source packaging and
check-release, indexed FamilyTraining data, existing TrainingSession/state/
checkpoint machinery, FamilyExecutionProfiler, Kaggle adapter/skill, receipts,
terminal acceptance and RML pipeline. Do not add a trainer or submission loop.

New work is limited to a faithful compact GRIT model/family hook, its accepted
CPU topology sidecar if absent, and the reviewed GPTrans Common-Dev binding.
Validate author-operation parity, true degree before pair completion, padding
masking, feature alignment, frozen restart and BN/EMA behavior. Pin an exact
author commit and preserve its license. Main agent implements; configured Luna
workers execute affected checks. Tests cannot certify remote release alone.

## Evidence completion

Before compute: both identities, initialization artifacts, recipe/source SHA,
reference qualification/establishment semantics, role applicability, input
acceptance, native-runtime certificate, cost ceiling and complete trace plan.

After compute: all scheduled observations with steps/presentations/LR/units;
selected and final-Live predictions with row IDs/targets;2M and6M checkpoints;
optimizer/RNG/scheduler/EMA and normalization state; role events; profiler and
actual allocation ledger; SHA-bound independently retrievable chunks; terminal
decision; canonical traces and qualified reference binding. Run existing
terminal finalization, RML validate/rebuild and actual replay admission checks.

Strict comparison and Replay-Ready are post-run verified outcomes, never
promises from a plan. Keep legacy PPGT/ESA analyses as separate contextual
records. The complete notebook must answer both questions even if GRIT loses:
publish the reference benchmark, practical retention decision, GRIT verdict,
cost frontier and unresolved causal limits. Submit no successor automatically.
