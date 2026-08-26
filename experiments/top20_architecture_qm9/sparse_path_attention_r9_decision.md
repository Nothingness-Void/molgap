# Sparse path-attention EdgeState R9 QM9 validation decision

## Decision recorded 2026-08-26

The validation-only R9 run completed on Kaggle2 kernel
`kaseichou/molgap-sparse-path-attention-r9-qm9-validation`, version 1. It used
source commit `0ffdef4ee046e6ffdba256d6c1b05758c61f2416`, the frozen split, accepted
RWSE16 cache, and accepted multihop cache. The QM9 test role was not constructed
or read.

The 4,752,327-parameter candidate passed every execution and artifact check but
failed both strict accuracy gates. Accepted R3 remains the sole validation
winner.

| Model | Validation average (eV) | Gap (eV) | Gate |
|---|---:|---:|---|
| accepted R3 EdgeState | 0.1052765 | 0.1261376 | retained |
| R9 sparse path attention | 0.1068024 | 0.1282017 | fail |

R9 regressed by `0.0015259 eV` on validation average and `0.0020641 eV` on
Gap. HOMO and LUMO also regressed. Its best checkpoint was epoch 19, so the
result was not caused by a crash, parameter cap, non-finite value, or patience
stop.

## Architecture conclusion

Keeping virtual paths out of the local convolution recovered most of R8's
large regression, but shortest-path-biased sparse attention still did not add
net information beyond R3's RWSE and dense GPS attention. The shortest-path
branch is closed without a rank, distance, seed, or schedule retry.

The next candidate changes a different information path: non-backtracking
directed bond-to-bond updates. This tests whether adjacent chemical bonds need
explicit state exchange before each GPS block, without virtual edges, new
readout logic, or target-specific computation.

## Independent acceptance

The inference-free local acceptance verified source, split, cache, parameter,
role, selection, and artifact identities. The recomputed winner is
`edge_state_structural_gps`.

## Evidence

- Compact acceptance: `results/sparse_path_attention_r9_acceptance.json`
- Submission record: `results/sparse_path_attention_r9_remote_submission.json`
- Downloaded output:
  `platforms/_records/kaggle/training/sparse_path_attention_r9_qm9_validation_v1/`

The QM9 test gate remained sealed when this decision was recorded.
