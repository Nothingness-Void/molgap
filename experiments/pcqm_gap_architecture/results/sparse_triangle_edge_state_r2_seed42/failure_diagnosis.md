# Sparse Triangle EdgeState GPS9 Seed-42 Failure

This record describes the single authorized Kaggle2 GPU attempt under the
Sparse Triangle EdgeState GPS9 protocol. It is a failure record, not an
accuracy result.

## Remote outcome

- Kernel: `kaseichou/molgap-pcqm-triangle-edge-state-r2-s42`, version 1
- Source commit: `35fadc9de63e22de7a1cfbe21e4f1af8888e075f`
- Seed: 42
- Stage reached: model preflight construction
- Epochs completed: 0
- Validation metrics: none
- Checkpoint: none
- Official validation/test-dev roles read: false

Kaggle returned `KernelWorkerStatus.ERROR`. The downloaded failure payload was:

```text
AttributeError: 'AtomEncoder' object has no attribute 'out_features'
```

The traceback reaches `OGBSparseTriangleEdgeStateGPSWrapper.__init__` at
`src/molgap/pcqm_gap_architecture.py:262`, where the wrapper reads
`self.node_emb.out_features` after replacing the base embedding with the OGB
categorical `AtomEncoder`. The installed OGB encoder does not expose that
attribute. This is a source/API compatibility defect, not evidence about the
architecture or its accuracy.

The log also reports that Kaggle assigned a Tesla P100 with CUDA capability
6.0 while the installed PyTorch build supports `sm_70` through `sm_120`.
That environment warning is retained, but the fatal traceback occurred at
the encoder attribute access before any training step.

## Preserved remote evidence

Downloaded under
`platforms/_records/kaggle/training/pcqm_gap100k_sparse_triangle_edge_state_r2_seed42`:

| File | SHA-256 |
|---|---|
| `molgap-pcqm-triangle-edge-state-r2-s42.log` | `aff85ce9ebbeb8fa4e26022ec91798e661099efe21a1f785a71942249f2e0c86` |
| `pcqm_gap100k_sparse_triangle_edge_state_r2_seed42/failure.json` | `105d372d11e07c20c0a8d2fd82b54de1349fef2892d60646646e943be4517179` |

The metadata-only slug/title failures preceding this run remain archived and
are not counted as model results. At failure acceptance, no retry, seed-43/44
run, full-data run, or official evaluation was authorized. On 2026-08-29 the
user explicitly authorized one implementation-only R3 retry because R2 never
reached an epoch; that authorization does not change this R2 failure record.
