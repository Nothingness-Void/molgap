# GAPE-lite GPU v1 infrastructure failure — 2026-09-09

Kaggle2 kernel `kaseichou/molgap-qm9-gape-lite-s42`, version 1, terminated
before training. Both isolated workers reached the accepted graph-cache loader,
then PyTorch 2.6 applied its new `torch.load(..., weights_only=True)` default
and rejected the trusted `torch_geometric.data.Data` payload. The first traces
appeared at about 16 seconds; no epoch, checkpoint, validation payload, metric,
or parameter report was produced.

This is an infrastructure compatibility failure, not a scientific result. The
repair explicitly loads the project-built, hash-accepted PyG cache with
`weights_only=False`. It does not change the cache, split, model, graph
features, GAPE objective, seed, FP32 precision, optimizer, schedule, physical
batch 128, epoch counts, comparison gate, or sealed-role policy. One repaired
remote version is therefore allowed under the existing protocol.

The retained log SHA-256 is
`d7caa909ecaa590a443b43ba3838efeb7af11455002957af4e3c04104fb94973`.
Full downloaded evidence remains under
`platforms/_records/kaggle/training/qm9_gape_lite_s42_v1/`.

