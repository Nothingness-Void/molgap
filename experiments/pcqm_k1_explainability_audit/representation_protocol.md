# Representation diagnostic design

Prepared 2026-09-30. Planning only; not a compute-release record.

## Question and scope

Distinguish concentrated global communication from collapse of retained local
states, and measure whether exchange updates affect the downstream prediction.
The [saved-evidence reassessment](results/slot_compression_reassessment.md)
explains why dispersion, deletion, and scalar-strength sweeps are not repeated.
No model training, candidate promotion, or architecture modification is part
of this diagnostic. Effective rank and sensitivity are not accuracy proxies.

## Inputs and release prerequisites

- Reuse the immutable 100K K1 model identity and original-payload reproduction
  gate in `src/molgap/pcqm_k1_cross_scale_diagnostic.py`.
- Obtain the accepted 500K K1 checkpoint from reference job `122743291`, named
  in `../pcqm_k1_pair_token_scale_attribution/results/round2_decision.md`.
  That decision proves endpoint provenance, not local availability of weights.
  Resolve and hash actual weights, matching loader, target transform, retained
  prediction payload, and accepted training contract before release. Do not
  substitute the older scale500K K1 merely because its model name matches.
- Both checkpoints must reproduce retained predictions on their allowed
  roles before instrumentation. Hooked and unhooked predictions must agree
  within the existing reproduction tolerances. No parameter or buffer changes.
- Freeze one label-independent panel: select 1,024 source IDs with smallest
  SHA256 of ASCII `k1-representation-v1:<source_idx>` within 500000..549999;
  tie-break by source ID and execute in ascending source-ID order. Freeze the
  resulting row manifest/digest before execution. Report all panel rows; no
  reselection by residual or representation findings.
- Use the accepted fixed500K development shard loader. These rows are outside
  both training prefixes but are reused development data, not independent test.
  Never evaluate the 500K checkpoint on the old 100K development role.
- Bind runtime, source, input hashes, NO_TRAIN role events, output/cost plan,
  platform receipt and reference evidence through existing V5 release helpers.
  Until these are present this document does not authorize a runnable release.

## Minimal implementation

Reuse frozen loaders, temporary-hook patterns, unchanged-weight checks,
atomic chunks and acceptance from `pcqm_k1_cross_scale_diagnostic.py` and
`k1_relation_intervention.py`. Their bound constants are not interchangeable;
add only the explicit accepted 500K loader binding and representation hook.
Do not rewrite a trainer or launch framework.

At layers 3/6/9, record per-molecule node H and update U statistics before and
after H+U using `molgap.representation_diagnostics.exchange_summary`. Record
centered bond-state spectra separately, if the matching model exposes them.
Account for directed duplicate bonds in interpretation. Do not compare raw
rank across molecule sizes without its rank ceiling.

Optionally compute d(prediction_eV)/d(H+U) with autograd on activations only,
weights frozen, evaluation mode and FP32. The derivative of sum of per-graph
predictions is valid only after verifying no cross-graph coupling in eval.
No loss, target-conditioned gradient, optimizer or update is allowed. Record
gradient/update cosine and their inner product, not a claimed removal effect.
This hook is not yet implemented or runtime-qualified.

Use BS128. Eight panel batches, two frozen checkpoints, baseline and hooked
passes; no coefficient grid. Freeze a native-cost cap after the owning platform
and accepted runtime are selected; do not invent timing from unrelated runs.
Write row-identified statistics/predictions in atomic retrievable chunks.
Intermediate full states need not be retained when the source and deterministic
statistics are bound; retain the row statistics needed for independent analysis.

## Analysis and stopping rules

Report paired same-row changes in spectra, dispersion and sensitivities,
overall and in the existing topology strata. Analyze within-checkpoint links
to residuals, controlling at least for atom count. Label-selected residual bins
remain explanatory and cannot become inference routing rules.

Two checkpoint endpoints cannot separate sample exposure, checkpoint selection,
training randomness, and data-scale effects. Do not call their difference a
strict causal architecture result or pool it into a training replay comparison.

- Stable local spectra with changed sensitivity: investigate communication
  content/addressing only if it also aligns consistently with residual changes.
- Low rank without aligned residual/sensitivity evidence: no architecture lead.
- Inconsistent strata or non-reproducing checkpoints: stop; no automatic model.

Any follow-up mechanism requires its own prospective comparison. Close this
diagnostic using NO_TRAIN evidence and measured cost; never fabricate an
optimizer trace or declare it training replay-ready.
