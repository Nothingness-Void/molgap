# Two family-specific scale falsifiers — 2026-10-25

## Decision and authority

The human offered two further experiments to decide whether PPGT and ESA merit
investment, explicitly including a500K opportunity. Reserve one question for each
family. Each question compares100K and500K training at equal optimizer updates;
it is not permission for two tuning grids, extra seeds, a new addon or full training.
This is a **design draft, not an executable release or prospective RML enrollment**.
No training was submitted by this decision. The complete release prerequisites
below must pass before either opportunity is spent.

The [accepted ESA result](results/completed_v2/decision.md) supported a modest
within-family count gain, with declining late benefit and worsening development
scores despite improving online training fit. The
[prepared PPGT result](../modern2d_configured_100k/prepared_pair/results/decision.md)
ended at its best20th pass. Its
[conditioning falsifier](../modern2d_structure_100k/results/decision.md) did not
justify that addon. Consequently the unresolved hypotheses differ: ESA may need
more distinct training molecules rather than more repetitions; PPGT has both
optimization-budget and execution-cost limitations. Neither was an equal-budget
reproduction of a paper's best model. Weak100K endpoints do not establish a
family-wide expressivity ceiling.

## Reserved experiments

| Question | Fixed model |100K/500K common endpoint | Reference source |
|---|---|---|---|
| ESA distinct-molecule benefit |ESA/4 with the accepted512-parameter count readout; LR3e-4 |46,860 updates,5,998,080 presentations |The accepted count trajectory, only if exact schedule/evaluation/transform/runtime reuse qualifies |
| PPGT distinct-molecule benefit |PPGT/3 with the established mean readout; LR1e-3; no rejected conditioner |15,620 updates,1,999,360 presentations |Retained20-pass PPGT outputs are context until actual reuse qualification; never reconstruct missing prospective authority |

Physical BS128, seed42, FP32/TF32off, model dimensions, optimizer/WD, loss and
accepted graph feature pipeline stay fixed within each question. Both scales
start independently from the identical original initial tensors. No100K weights
warm-start500K. Reuse a qualified reference rather than train it again. If a new
matched100K control is necessary, it occupies the paired T4 in the same
family-specific experiment and is declared before release; it is not a third
model-design question. Independent arms retain isolated state and visible GPUs.

Use the accepted fixed train500K membership and the same Common-Dev50K. Freeze
the100K train-only target mean/std for both scales to avoid a transform change
masquerading as data diversity. Training-scale membership, order and sampling
identities differ explicitly; the development targets/order must align exactly.
Report both total presentations and unique members actually observed.500K
cardinality alone is not proof that every member was seen.

Freeze the **entire LR-by-update sequence**, gradient clipping, evaluation
update indices and selection rule against the source100K recipe. Changing a
four-pass warmup to four500K passes would multiply the warmup by five and defeat
this design. A500K endpoint may lie inside a pass:46,860 is12 updates short of
12 complete500K passes;15,620 is4 updates short of4 complete500K passes. Save
the actual cursor and sampler/RNG state. Do not add tail updates or relabel this
as a completed integer-pass run. Required comparisons include the same-update
final live endpoint and selected checkpoints using the same frozen evaluation
opportunities. Selection/weight semantics cannot silently change on reuse.

## Decision gate, frozen before release

1. A data-diversity signal requires500K-minus100K improvement of at least0.005eV
   at the matched final endpoint, a positive paired-row95% lower gain bound, and
   a positive mean gain across the final five matched evaluation points. This
   proposed screening threshold is specific to this rescue question; it is not
   a V5-wide scientific threshold or a training-seed confidence statement.
2. That signal alone does **not** qualify the family for further investment.
   Freeze a strong K1 or GPTrans retained comparator before release only after
   checking the same Common-Dev IDs, labels, evaluation semantics and budget.
   A500K rescue must reach within0.005eV of that qualified comparator, or beat
   it, within the allowed cost. Different recipes are a descriptive competitive
   screen, never a strict causal architecture ranking. If no comparator qualifies,
   competitive nomination is INCONCLUSIVE; do not substitute an old scalar.
3. If complete accepted endpoints miss the diversity gate and remain clearly
   inferior in the qualified competitive screen, close further budgeted research
   on that family. Preserve its implementation and evidence. This closes an
   investment decision under this configuration/budget, not all possible paper
   implementations or larger-budget variants.
4. STOP_FOR_COST, native failure, missing output or uncertain runtime is not a
   completed scientific negative. Preserve the partial trace and cost; do not
   silently extend the horizon or consume a third opportunity. No automatic
   multi-seed, architecture change,500K continuation or full training follows.

No MAE gate can decide family potential if the run is still strongly improving
at an incomplete endpoint. Report whether it is competitive within this budget
and separate remaining uncertainty about ultimate convergence.

## Runtime and release prerequisites

Historical measured PPGT capture-off throughput was144.51 graphs/s and its
complete100K-pass median807.54s. At1,999,360 presentations, training-only division
gives about3.84h; the actual20-pass process measured16,756s (about4.65h).
Plan roughly5–7h for the matched PPGT pair, with a hard10h whole-slot ceiling.
The admitted ESA count60-pass process measured6,456s (about1.79h), so plan
roughly2–3h for its same-update scale screen, also bounded by10h. These are
estimates, not native500K throughput measurements: graph sizes, memory mapping,
evaluation and admission overhead can change them. Verify a warmed native
projection including remaining evaluations and checkpoint reserve before formal
training; retain measured profiles. No recipe change is a speed fallback.

Kaggle session quota is wall time; physical allocated T4 cost includes both
devices, including idle time. Use T4x2 for an actual independent control/candidate
pair. A single reused-reference candidate must declare its idle-device reason
and charge the full allocation. Query actual account quota at release. Kaggle1
remains outside this server chain; use authorized Kaggle2/3 capacity only.

Concrete blockers identified during this review:

- The pinned Modern2D private export/qualification metadata binds train100K plus
  Common-Dev, not an accepted train500K export. Dataset-loader support for500K
  is not proof of published, byte-verified500K inputs. Use the existing scheduled
  CPU selected-role export/acceptance/publication owners; never rebuild graphs or
  read protected roles. CPU prerequisite work is not a GPU architecture trial.
- The configured Modern2D profile currently requires complete integer passes.
  It cannot represent the exact partial-pass endpoints above. Add a narrowly
  explicit step-budget mode to the existing profile/compiler and strengthen its
  existing Session/resume workflow; do not round exposure silently.
- The configured scientific release path permits architecture-only same-run or
  retained-reference differences. This plan changes training membership and must
  be represented as a transfer study, not forced into STRICT_CAUSAL or passed by
  disabling the gate. Preserve legitimate PAIRED_ENDPOINT/context semantics and
  define actual replay applicability before release. Generic noncausal support
  elsewhere is not proof that this family's release/terminal path supports it.
- Source-train normalization is also part of the input/feature metadata and its
  runtime checks. A reused100K transform on500K needs an explicit frozen-transform
  binding through those owners; do not edit only the recipe scalars or disable
  the selected-train normalization verification.

Implement any necessary adapter changes through existing Recipe/Session,
indexed inputs, family acceptance and RML owners, with focused Luna verification.
Freeze prospective authority, package, role identities, cost, comparator and
decision thresholds only after the executable design is supported. Every
training run must retain trace, runtime/native proofs, observed roles, exact
selected/final predictions, checkpoint/resume state and terminal decision. Missing
evidence blocks qualification; no retrospective fabricated Replay-Ready labels.

The two scientific opportunities remain unspent while these gates are closed.
