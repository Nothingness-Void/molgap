# PCQM Gap100K Seed-42 Structural GPS versus EdgeState

## Question

Did persistent real-bond EdgeState improve a matched OGB-categorical Structural
GPS9 on the internal official-train-derived PCQM Gap100K split?

## Frozen contract

- Data: 100,000 effective training graphs and 10,000 internal-validation
  graphs from the official PCQM4Mv2 training role.
- Cache aggregate SHA-256:
  `eb7c843e33f430ac755bc575d80153aba87677cea1ad5bb0dcf73cca906e2c21`.
- Seed 42, FP32, batch 48, AdamW, learning rate `1.6e-4`, weight decay
  `1e-6`, 40 epochs, patience 8, direct scalar Gap prediction.
- Pure 2D OGB categorical atoms/bonds plus RWSE16; no checkpoint, residual
  target, prediction fusion, auxiliary HOMO/LUMO target, or 3D input.

## Accepted result

| Candidate | Parameters | Best epoch | Internal-validation Gap MAE | Mean epoch time | Mean throughput |
|---|---:|---:|---:|---:|---:|
| OGB Structural GPS9 | 3,809,089 | 34 | 0.1433127675 eV | 118.47 s | 844.32 graphs/s |
| OGB persistent EdgeState GPS9 | 4,771,073 | 38 | 0.1379826321 eV | 147.52 s | 678.11 graphs/s |

EdgeState improved MAE by `0.0053301353 eV` (`3.7192%`) while adding
`961,984` parameters (`25.26%`) and increasing mean epoch time by about
`24.52%`. The complete sequential P100 task took `10,841.76 s` (about
3.01 hours).

Downloaded artifacts passed the no-inference acceptance in `acceptance.json`.
All model, checkpoint, trace, and validation-payload hashes matched. The run
did not read official PCQM validation or test-dev.

## Decision

The EdgeState result became the frozen seed-42 comparator for subsequent
official-PCQM 100K architecture discovery. It did not authorize seeds 43/44,
full-data training, official-validation evaluation, or test-dev inference.
Subsequent candidates must change information flow materially, use the same
cache and training contract, and beat `0.1379826321 eV` before a multiseed gate
can open.

The retrievable Kaggle output is mirrored locally under
`platforms/_records/kaggle/training/pcqm_gap100k_r1_seed42_v1`.
