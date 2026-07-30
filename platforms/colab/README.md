# Colab Adapter

This directory contains durable Colab launch bundles, currently centered on
repaired-2M 3D graph construction. Notebooks are platform adapters; reusable
graph and acceptance logic remains in `src/molgap/`.

Each long run must emit resumable bounded parts and manifests. Live status is
owned by `CURRENT_STATE.md`; downloaded evidence belongs in
`platforms/_records/`.

## Paid-Accelerator Gate

A Colab training notebook must fail before epoch 0 unless all of these pass:

1. the notebook requires exactly one project wheel and its declared SHA256;
2. the accepted input manifest and every staged shard match SHA256;
3. the environment records Python, Torch, CUDA, PyG, extension, GPU, wheel,
   input, and checkpoint identities;
4. first and last shards pass real forward/backward with finite loss and
   gradients;
5. a warm-start or recovery checkpoint reproduces its frozen validation
   baseline and a full test result consistent with the same random split;
6. output uses a new run directory or an exact resume contract;
7. the first non-finite loss or gradient aborts training;
8. the selected state is hashed, final validation is replayed, and test is
   evaluated twice before a completion manifest is written.

File size, cell completion, and the presence of `model.pt` are never acceptance
evidence. A failed run directory is quarantined and must not be reused by a
replacement run.
