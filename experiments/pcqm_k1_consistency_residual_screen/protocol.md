# K1 consistency residual screen — 2026-10-04

Question: does the accepted consistency gain reflect removable output bias,
distributed error improvement, or complementary prediction errors? This is a
new saved-prediction diagnostic, not a reopening of the closed training recipe.

## Evidence and authority

Reuse the accepted mean2 and consistency2 selected-epoch37 predictions from
`../pcqm_k1_dropout_consistency/terminal_acceptance/`. The paired gain is
1.5287825 meV, below its frozen3 meV gate. Both completed40epochs/31240updates.
Historical clean K1 runtime is not a strict causal comparator. Width256, slot96,
SSMA and FLAG do not establish a better next capacity/aggregation direction.
The user requested analysis and continued improvement; this permits this CPU
saved-artifact diagnostic, not training, checkpoint inference or promotion.

Read only retained internal development rows100000:150000. This role already
selected both checkpoints; a new modulo partition is not an untouched test set.
Official validation/test roles remain sealed for this question. No graph,
checkpoint, raw dataset or accelerator is consumed.

## Frozen probes and decision

1. Reproduce accepted paired endpoints with the existing `paired_metrics`.
2. Use existing residual attribution to measure signed bias, correlation and
   descriptive true-Gap bins [0,2),[2,4),[4,6),[6,8),[8,infinity).
3. Measure improvement/harm contributions and the control's top1% absolute-error
   cohort. These label-informed strata explain distribution, not deployable routes.
4. Fit one constant median residual correction per model on source_idx%5==0.
   Evaluate corrections on %5!=0. No slope, nonlinear calibrator or parameter sweep.
5. Compare fixed50:50 blend and the existing0.001-grid convex blend fitted only
   on %5==0. Record weights and evaluate the40K held-out rows against consistency2.
6. Use1000 paired-row bootstrap draws, seed42, on the held-out differences.

Additional gain>=1meV with positive95% lower row bound nominates further
independent qualification of that specific postprocessor; otherwise close that
cheap route. This is a prospective diagnostic nomination threshold, not a
replacement for the original3meV training gate or a training variance estimate.
Even a positive probe does not release full evaluation, adoption or training.
No Oracle row selection, graph model, coefficient/seed/schedule retry is fitted.

CPU budget: one pass over two50K tensor payloads plus frozen bootstrap/grid;
estimated wall<=180seconds; accelerator not applicable. Retain actual processCPU
and wall separately. Outcome NO_TRAIN; preserve original accepted verdicts.

## Reuse

`analyze_pair.paired_metrics`, `residual_attribution._metric_block`,
`pcqm_k1_gptrans_fusion.calibration_mask/fit_convex_blend_weight/mae`,
`router.paired_bootstrap_mean`,
`training_reproducibility.atomic_json/sha256_file`, RML `plan` and `finalize`.
The experiment adapter only adds the above decomposition and constant correction.
