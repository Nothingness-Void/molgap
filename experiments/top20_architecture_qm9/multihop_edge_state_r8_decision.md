# Multihop EdgeState Structural GPS R8 QM9 validation decision

## Decision recorded 2026-08-26

The validation-only R8 run completed on Kaggle2 kernel
`kaseichou/molgap-multihop-edgestate-r8-qm9-validation`, version 1. It used
source commit `56abea806ff88778bcbb847d569a266da074eee1`, the frozen split
fingerprint `01656b1a538f89c8`, the accepted RWSE16 cache, and the independently
accepted 17-part multihop cache. The QM9 test role was not constructed or read.

The 4,739,907-parameter candidate passed every execution and artifact check but
failed both strict accuracy gates. Accepted R3 remains the sole validation
winner.

| Model | Validation average (eV) | Gap (eV) | Gate |
|---|---:|---:|---|
| accepted R3 EdgeState | 0.1052765 | 0.1261376 | retained |
| R8 multihop local EdgeState | 0.1130129 | 0.1365600 | fail |

R8 regressed by `0.0077364 eV` on validation average and `0.0104224 eV` on
Gap. HOMO and LUMO also regressed. Its best checkpoint was epoch 19, so the
result was not caused by a crash, parameter cap, non-finite value, or patience
stop.

## Architecture conclusion

Turning every distance-two-to-four atom pair into a local persistent edge
overwhelmed the chemical-bond inductive bias. The extra paths were valid and
cheap in parameters, but local message passing could no longer distinguish
bond transport from topological context strongly enough. R8 is closed without
a distance-cap, seed, or schedule retry.

The accepted path cache can support a materially different question without
new graph construction: retain R3 local messages on real bonds only and use
the distance-one-to-four pairs exclusively in a shared sparse attention branch
with learned shortest-path bias.

## Independent acceptance

The inference-free local acceptance verified source, split, RWSE, multihop
cache, parameter, role, selection, and artifact identities. The recomputed
winner is `edge_state_structural_gps`.

## Evidence

- Compact acceptance: `results/multihop_edge_state_r8_acceptance.json`
- Cache acceptance: `results/multihop_edge_state_r8_prep_acceptance.json`
- Submission record: `results/multihop_edge_state_r8_remote_submission.json`
- Downloaded output:
  `platforms/_records/kaggle/training/multihop_edge_state_r8_qm9_validation_v1/`

The QM9 test gate remained sealed when this decision was recorded.
