# Synthetic execution checks, 2026-10-11

Engineering-only scratch models/graphs on a nonexclusive RTX5060, FP32/noTF32,
native AdamW defaults, BS128, single-forward normalized L1 and original clipping.
No molecular dataset, retained checkpoint, development/protected role, quality
training or RML publication. Parent source is `b8f95004`; each result records
tested source hashes, runtime, shapes, raw timings and limits. These are not
whole-dataset throughput or scientific quality comparisons.

## Rejected projection rewrite

[Raw result](synthetic_projection_result.json): 3068 real nodes,6392 directed
edges, max40 nodes/graph; five warmup and24 measured steps per case per round,
three mode-order rotations. Median step times:

| Execution | Seconds/step |
|---|---:|
| Original | 0.091279 |
| CPU layout | 0.086047 |
| CPU layout + pre-gather/pre-padding projections | 0.086471 |

Projection rewriting was about0.49% slower than CPU layout alone. One-step
loss and RNG matched, but312 gradient tensors and308 updated-state tensors
differed; maximum gradient/state absolute differences were2.33e-8/1.73e-5.
Peak allocated memory decreased about10.2%, but execution speed did not improve.
This is not measured accuracy loss. Reject the rewrite for this context rather
than spend quality-training budget. Its flags and model edits were removed.

The first harness attempt failed on layout/membership object identity before
producing timing evidence; corrected harness uses the existing transfer inside
the same batch's owned layout context. Failure is retained separately, not
treated as a model or platform result.

## Bounded prediction copies

[Raw result](synthetic_copy_result.json): original versus existing CPU layout
matched one-step loss, metric, RNG, gradients and updated state exactly. Median
training steps were0.093295/0.088935s; nonexclusive short timings are not a new
full-run gain claim.

Synthetic evaluation used32 batches/4096 rows, with three chunk-order rotations:

| Prediction copy | Median seconds | Time reduction |
|---|---:|---:|
| Every batch | 1.506345 | Reference |
| Every <=512 rows | 1.500411 | 0.39% |
| Every <=4096 rows | 1.491616 | 0.98% |

Targets, predictions, row order and final MAE matched bitwise in all cases.
The small, noisy benefit does not establish worthwhile deployment speedup.
`k1_screen_training._evaluate(..., prediction_chunk_rows=4096)` is an opt-in
shared hook, not a registered execution policy or default replacement. `None`
preserves historical batch-copy behavior. Arithmetic, selection and validation
frequency are unchanged; pending scalar predictions have a bounded row limit.
All existing recipe/preflight/optimizer policies remain unchanged.

## Reuse and verification

Reusable synthetic-only entry, using the checkout's project Python:

```powershell
$env:PYTHONPATH = 'D:/w/k1-fused-t4/src'
$env:CUDA_VISIBLE_DEVICES = '0'
D:/文档/molgap/.venv/Scripts/python.exe -m molgap.k1_execution_profile --synthetic-local --output <fresh-output-directory>
```

The shared entry now compares original and CPU layout plus bounded prediction
copies; it does not retain rejected projection code. It requires RTX5060,
refuses existing output directories and enforces a180-second step-boundary
ceiling. Rejected and final exact tested sources are retained as hashed ignored
`source_snapshot/` files beside their staging results; they are local custody,
not a portable committed source package or replay-ready training evidence.

Focused regression:83 passed,1 deselected. The deselected historical A100
payload-byte test lacks `selected.pt` in this owner checkout's ignored staging;
that historical artifact verification was not completed here. Two stale transfer
mocks were updated to accept/assert declared field names; validation was not
weakened. GPU scratch checks completed in26.06s and32.78s respectively.
No default adoption, remote submission, Git integration or scientific closure.
