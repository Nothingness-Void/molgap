# Geometry Cache V1 Failure Diagnosis

Decision date: 2026-08-30

Kaggle2 kernel `kaseichou/molgap-pcqm-geometry-cache-s42`, version 1,
terminated about 22 seconds after launch while validating its parent sparse-
wedge cache. It produced no geometry shard and never entered ETKDG generation.
The GPU successor was not submitted.

The terminal exception was:

```text
RuntimeError: Wedge cache contract changed for source_commit
```

The parent cache is not corrupt. Its accepted manifest records source commit
`35fadc9de63e22de7a1cfbe21e4f1af8888e075f` and aggregate SHA-256
`dc62b8289b0d85bd71a2eca9a16b6223f53206dd9a901670bb799125eff77406`.
The geometry runner incorrectly required
`76dd6efa76c8236ce80a82a8a43d9f5df426165e`, which is the later Sparse
Triangle GPU wrapper repair commit rather than the source commit that produced
the wedge cache.

This is an infrastructure-only identity pinning error. It provides no evidence
for or against distance, angle, or combined bottom fusion. A retry would require
an explicit authorization and must change only the expected parent cache source
identity plus its static contract assertion. No architecture, split, target,
seed, optimizer, schedule, cache payload, or sealed-role setting needs to
change.

No local model inference, official validation/test-dev access, GPU training, or
molecular-research-server access occurred.
