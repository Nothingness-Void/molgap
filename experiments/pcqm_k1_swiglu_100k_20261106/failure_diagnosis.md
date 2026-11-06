# SwiGLU numerical failure diagnosis

2026-11-06 JST. The retained checkpoint reproduces model-side nonfinite
development predictions. No data-order, row-count, checkpoint-corruption or
train/eval mode-switch defect was found. The original INCONCLUSIVE outcome stays.

## Observed evidence

Only the exact-version root `experiment/swiglu/last_checkpoint.pt` was additionally
retrieved from Kaggle job142921674/version1: 59,211,363 bytes, step8576,
pass_index10/next_batch766. The next pass boundary is8591. Original model tensors
are finite; tracked BN variances are finite and positive, with8576 tracked batches.
The earlier completed trace endpoint remains step7810; these are different facts.

The ordered development scan stopped at batch28, source103584..103711. Its target
values, shapes and source order were valid; source103636 produced NaN. With the
unchanged checkpoint in eval mode, local-block norm3 maxima grew from13.33 through
138.52,1200,4.28e4,1.08e7,1.40e12 to2.08e22. The first nonfinite module output was
`neural_atom_mixers.9.node_norm`: finite input, nonfinite LayerNorm output.

On the same batch, an in-memory diagnostic using batch statistics only for BN
(Dropout still eval) produced finite predictions, max absolute8.59689eV. This
supports sensitivity to the frozen-statistics eval path and SwiGLU's repeated
multiplicative amplification. It is not an approved inference method or evidence
of improved model quality. No optimizer update or persistent model mutation ran.

CPU Torch2.7.1 reproduces this failure class on the actual pre-failure state.
Kaggle used Torch2.11.0; this is not native replay qualification or identification
of the exact final cloud operand. The formal runtime-pin rejection was preserved.

## Minimal infrastructure repair

Desktop `adb45e32`, owner `0eaa13c2`, separates row-identity failures from
nonfinite predictions/targets and reports count plus the first eight source IDs.
The real checkpoint now fails through `evaluate_development` with:
`Development predictions non-finite: count=1; source_idx=[103636]`.
The guard still rejects the result; no NaN replacement, skipped molecule, retry,
precision change or relaxed qualification was introduced. The existing real K1
kernel check was extended and passed on CPU in9.88s; no new test node or file.
This repair improves diagnosis, not model stability.

## Reproduce and next decision

Retained local diagnostics are under
`D:/molgap-artifacts/reconciliation/k1-swiglu-100k-20261106-failure/checkpoint_diagnosis/`:
`retrieval.json` owns transport identity; `development_scan.json` owns scan results;
`diagnostic_report.json` owns checkpoint/batch/layer observations. The checkpoint
is retained there once. Run `diagnose_checkpoint.py <that-directory>` with the
primary checkout's Python; it reuses the model factory, loader and evaluator.
It consumes only the already-authorized internal development role and performs
no training or network calls. Temporary exploration scripts were removed.

The user chose proposal-before-submission. Proposed, untested stabilization:
place an affine-free LayerNorm before the addon's gate/value projection, leaving
residuals, backbone BN, matrix budget, optimizer, batch and training budget unchanged.
This addresses amplification before multiplication, rather than hiding the final
NaN. It requires user confirmation, a new addon version and frozen contract, and
the original initialization; it is not equivalent continuation of the old state.
After confirmation, reuse the existing sandbox for a bounded gradient/train-eval
check and this failing batch for numerical diagnosis. No new validation framework
or default test suite is proposed. No successor was submitted by this diagnosis.
