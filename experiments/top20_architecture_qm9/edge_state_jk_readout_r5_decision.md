# EdgeState-JK readout R5 QM9 validation decision

## Decision recorded 2026-08-26

The validation-only R5 run completed on Kaggle2 kernel
`kaseichou/molgap-edgestate-jk-readout-r5-qm9-validation`, version 1. It used
source commit `04f303f8ebf51fccfb85eb477e4fe92dbc4a7553`, the frozen split
fingerprint `01656b1a538f89c8`, and the accepted RWSE16 cache. The QM9 test role
was not constructed or read.

The 4,767,779-parameter candidate passed its execution and artifact checks but
failed both accuracy gates. The accepted 4,739,651-parameter R3 EdgeState model
therefore remains the sole validation winner.

| Model | Validation average (eV) | Gap (eV) | Gate |
|---|---:|---:|---|
| accepted R3 EdgeState | 0.1052765 | 0.1261376 | retained |
| R5 EdgeState-JK | 0.1079212 | 0.1304649 | fail |

R5 regressed by 0.0026447 eV on validation average and 0.0043273 eV on Gap.
HOMO and LUMO also regressed, so the result is not a target-specific tradeoff.
Its best checkpoint was epoch 19, matching the end-of-contract behavior of R3;
the failure was not caused by a crash, parameter cap, non-finite value, or
premature early stop.

## Architecture conclusion

Exposing normalized layer-3, layer-6, layer-9, and final edge summaries through
a zero-initialized bottleneck did not improve the accepted final-layer mean
representation. The residual adapter preserved the R3 prediction at
initialization, but after optimization it added capacity without useful
validation signal. This result closes the R5 readout branch; it does not
authorize a longer run, another seed, PubChemQC transfer, or test evaluation.

## Independent acceptance

The inference-free local acceptance verified the source commit, split and RWSE
identities, exact parameter count, finite remote forward/backward preflight,
train/validation-only roles, selection arithmetic, and SHA-256 hashes for the
model, checkpoint, payload, and metrics. The selected candidate recomputed from
the frozen gates is `edge_state_structural_gps`.

## Evidence

- Compact acceptance: `results/edge_state_jk_readout_r5_acceptance.json`
- Submission record: `results/edge_state_jk_readout_r5_remote_submission.json`
- Downloaded output:
  `platforms/_records/kaggle/training/edge_state_jk_readout_r5_qm9_validation_v1/`

The QM9 test gate remained sealed when this decision was recorded.
