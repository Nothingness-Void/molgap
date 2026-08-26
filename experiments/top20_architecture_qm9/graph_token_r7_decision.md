# Graph-token Structural GPS R7 QM9 validation decision

## Decision recorded 2026-08-26

The validation-only R7 run completed on Kaggle2 kernel
`kaseichou/molgap-graph-token-r7-qm9-validation`, version 1. It used source
commit `6b46c3a4867b1929b549179501d6b8cf25c73928`, the frozen split fingerprint
`01656b1a538f89c8`, and the accepted RWSE16 cache. The QM9 test role was not
constructed or read.

The 4,787,475-parameter candidate passed every execution and artifact check.
It improved HOMO and Gap but regressed LUMO enough to fail the strict average
gate. Accepted R3 therefore remains the sole validation winner.

| Model | Validation average (eV) | Gap (eV) | Gate |
|---|---:|---:|---|
| accepted R3 EdgeState | 0.1052765 | 0.1261376 | retained |
| R7 graph-token EdgeState | 0.1056126 | 0.1254950 | fail |

Relative to R3, R7 changed HOMO by `-0.0006601 eV`, LUMO by
`+0.0023109 eV`, Gap by `-0.0006426 eV`, and average by `+0.0003361 eV`.
Its best checkpoint was epoch 19, so the result was not caused by a crash,
parameter cap, non-finite value, or patience stop.

## Architecture conclusion

The recurrent graph token carried useful global signal for HOMO and Gap, but
its shared graph-to-node broadcast displaced LUMO information. The result does
not support another token width, seed, schedule, or readout retry under this
screen. R7 is closed.

The next bounded architecture may retain accepted EdgeState while changing
the local information path. A train/validation-only sparse shortest-path cache
is the next untested mechanism: persistent edge states can then propagate over
real bonds and topology-derived virtual edges without adding another global
readout or target-specific path.

## Independent acceptance

The inference-free local acceptance verified source, split, RWSE, parameter,
role, selection, and artifact identities. The recomputed winner is
`edge_state_structural_gps`.

## Evidence

- Compact acceptance: `results/graph_token_r7_acceptance.json`
- Submission record: `results/graph_token_r7_remote_submission.json`
- Downloaded output:
  `platforms/_records/kaggle/training/graph_token_r7_qm9_validation_v1/`

The QM9 test gate remained sealed when this decision was recorded.
