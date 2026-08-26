# Edge-conditioned Structural GPS R6 QM9 validation decision

## Decision recorded 2026-08-26

The validation-only R6 run completed on Kaggle2 kernel
`kaseichou/molgap-edge-conditioned-r6-qm9-validation`, version 1. It used
source commit `da204ad3ddb57d0d60a9e2a19338aa5c06c7810d`, the frozen split
fingerprint `01656b1a538f89c8`, and the accepted RWSE16 cache. The QM9 test role
was not constructed or read.

The 4,765,123-parameter candidate passed every execution and artifact check but
failed both strict accuracy gates. Accepted R3 therefore remains the sole
validation winner.

| Model | Validation average (eV) | Gap (eV) | Gate |
|---|---:|---:|---|
| accepted R3 EdgeState | 0.1052765 | 0.1261376 | retained |
| R6 edge-conditioned EdgeState | 0.1062025 | 0.1265785 | fail |

R6 regressed by 0.0009259 eV on validation average and 0.0004409 eV on Gap.
HOMO and LUMO also regressed. Its best checkpoint was epoch 19, so the result
was not caused by a crash, parameter cap, non-finite value, or patience stop.

## Architecture conclusion

Injecting the current persistent bond state into atom states before every GPS
block was substantially closer to R3 than the failed graph-level R5 readout,
but it still did not add complementary signal under the fixed contract. The
shared FiLM path changed the optimization trajectory without improving its
endpoint. This closes R6 without a retry or seed variation.

The next architecture may retain EdgeState but must change information flow.
A recurrent graph token is the next untested bounded operation from the frozen
top-20 audit; it is distinct from R5 readout and R6 edge conditioning.

## Independent acceptance

The inference-free local acceptance verified source, split, RWSE, parameter,
role, selection, and artifact identities. The recomputed winner is
`edge_state_structural_gps`.

## Evidence

- Compact acceptance: `results/edge_conditioned_r6_acceptance.json`
- Submission record: `results/edge_conditioned_r6_remote_submission.json`
- Downloaded output:
  `platforms/_records/kaggle/training/edge_conditioned_r6_qm9_validation_v1/`

The QM9 test gate remained sealed when this decision was recorded.
