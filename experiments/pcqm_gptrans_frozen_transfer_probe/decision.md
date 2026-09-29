# Frozen 100K GPTrans weights on the 500K development cohort

## Result

On the fixed 500K internal-development rows `500000:550000`, the accepted 100K
`pair_update_norm` checkpoint outperformed an accepted 100K GPTrans reference
checkpoint from a separate paired job:

| Frozen 100K checkpoint | 500K-cohort MAE (eV) |
| --- | ---: |
| GPTrans reference | 0.156090 |
| `pair_update_norm` | 0.150981 |
| Reference minus candidate | **0.005109** |

The row-bootstrap 95% interval for the paired gain is `[0.004251, 0.006023]`
eV. It measures uncertainty over these rows, not training-seed variation. Both
models used the same target normalization and graph schema; input checkpoint
and shard hashes are pinned in `protocol.md` and `results/full/summary.json`.
All 50,000 source indices and targets match the accepted GPTrans 500K
development-prediction artifact exactly.

The original 100K screen reported a `0.004843 eV` gain against its own frozen
reference. The separate 100K reference used here scored `0.156014 eV` on that
cohort, compared with `0.156627 eV` for the original reference. Therefore this
probe is a cross-job diagnostic, not a strict reproduction of the original
candidate/reference pair.

## Interpretation

The frozen candidate's advantage did **not** disappear when the evaluation
cohort changed. A rule that sends a candidate to 500K training when its frozen
transfer gain exceeds the existing `0.003 eV` materiality floor would have
allowed this case through. The actual accepted 500K retraining gain was only
`0.001140 eV`, below that floor. Thus this transfer check cannot, by itself,
serve as a reliable 500K budget gate. It may still flag some high-risk cases;
this single miss does not estimate sensitivity or false-stop risk.

The discrepancy is consistent with a small-data/sample-efficiency advantage
that shrinks as both arms see more training data and optimizer steps. It does
not isolate data scale from training horizon or seed effects. The 500K
development role has been repeatedly used for selection; this is not an
independent holdout or a new promotion result. No new model was trained and no
protected evaluation role was read.

The general cutoff decision remains `INCONCLUSIVE`. Keep frozen transfer as a
diagnostic and put decision weight on matched-prefix 500K evidence after the
policy has been calibrated on more completed cases. The exact original 100K
reference checkpoint should be used if it later becomes locally available.

## Local files

- `evaluate.py`: local inference with SHA-pinned checkpoints and graph shard.
- `results/full/summary.json`: full-cohort metrics and input/output hashes.
- `results/full/predictions.npz`: aligned per-row predictions and labels.
- `results/smoke/`: 128-row loader smoke test; not used in the decision.
