# Directed EdgeState Structural GPS R10 QM9 validation decision

## Decision recorded 2026-08-26

The validation-only R10 run completed on Kaggle2 kernel
`kaseichou/molgap-directed-edgestate-r10-qm9-validation`, version 1. It used
source commit `06bf8f439783cced552760b873e1702a0098c802`, the frozen split, and the
accepted RWSE16 cache. The QM9 test role was not constructed or read.

The 4,776,515-parameter candidate passed every execution and artifact check.
It improved Gap but narrowly failed the strict average gate. Accepted R3
remains the sole validation winner.

| Model | Validation average (eV) | Gap (eV) | Gate |
|---|---:|---:|---|
| accepted R3 EdgeState | 0.1052765 | 0.1261376 | retained |
| R10 directed EdgeState | 0.1054167 | 0.1257885 | fail |

Relative to R3, R10 changed HOMO by `+0.0001891 eV`, LUMO by
`+0.0005805 eV`, Gap by `-0.0003491 eV`, and average by `+0.0001402 eV`.
Its best checkpoint was epoch 19, so the result was not caused by a crash,
parameter cap, non-finite value, or patience stop.

## Architecture conclusion

Non-backtracking directed bond memory contributed useful Gap signal but did
not improve the joint frontier-orbital representation. R10 is closed without a
width, seed, initialization, or schedule retry.

The next candidate changes the local aggregation statistic rather than edge
recurrence: a shared PNA-style real-bond context will expose mean, maximum,
minimum, and standard-deviation neighborhood messages while retaining the R3
EdgeState and GPS backbone.

## Independent acceptance

The inference-free local acceptance verified source, split, RWSE, reverse-edge
coverage, parameter, role, selection, and artifact identities. The recomputed
winner is `edge_state_structural_gps`.

## Evidence

- Compact acceptance: `results/directed_edge_state_r10_acceptance.json`
- Submission record: `results/directed_edge_state_r10_remote_submission.json`
- Downloaded output:
  `platforms/_records/kaggle/training/directed_edge_state_r10_qm9_validation_v1/`

The QM9 test gate remained sealed when this decision was recorded.
