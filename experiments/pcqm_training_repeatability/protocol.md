# Same-allocation scratch repeatability protocol

## Question

How much best-development MAE drift remains when the historical seed-42
GraphState9 scratch control is repeated on two T4 devices in one Kaggle
allocation?

The `0.0026921320 eV` observation being audited came from two different Kaggle
accounts, not from Xi'an. Xi'an separately showed substantially larger
training drift and is excluded from model ranking by
`experiments/pcqm_xian_determinism/`.

## Frozen contract

- Model source: packaged from commit
  `c2302c7d433dfec73d293cdbd71ee46aa9e4cae5`, the source identity used by the
  matched 500K baseline/candidate run.
- Fixed graph dataset: `kaseichou/pcqm4mv2-ogb-fixed-100k-v1`.
- Cache identity:
  `bc83a4bd9a7fd7fd6fc6fd085d78ba3caab0e40f997d1932e0f344e701dd40c5`.
- Encoder: `ogb_distance_angle_triangle_edge_state_graph_state9`, 3,665,809
  parameters.
- Both workers must report the same initialization SHA-256; the value is
  recorded rather than inferred from the unavailable historical source image.
- Two independent worker processes, both seed 42. They run concurrently when
  Kaggle supplies T4x2, or sequentially on the same device when Kaggle supplies
  its single-P100 fallback.
- FP32, batch 48, AdamW, learning rate `1.6e-4`, weight decay `1e-6`,
  cosine schedule, 60 epochs.
- Train/development roles: 100K plus the fixed first 10K rows of the available
  50K development role, all derived only from official training data.
- Report best development MAE through epoch 40 and through epoch 60, complete
  epoch traces, final checkpoint model-state hashes, and wall time.

Batch 48 is retained to make the result comparable in shape to the historical
noise observation. The deleted historical source dataset and unreachable old
source object prevent a byte-identical reconstruction, so the historical
`0.0026921320 eV` remains a reference rather than a paired result. This run
does not replace the batch-128 architecture-screen policy.

## Interpretation

This pair estimates technical repeatability under one allocation. It cannot
establish an architecture gain. A candidate advances only through paired
multi-seed evidence; per-molecule paired uncertainty and technical-repeat
drift must be reported separately. Official validation and test-dev remain
sealed.
