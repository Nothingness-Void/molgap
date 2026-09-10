# K1 500K cache v2 terminal evidence

- Kaggle2 kernel: `kaseichou/molgap-pcqm-k1-scale500k-cache`, version 2.
- Terminal state: `ERROR`.
- Retrieved evidence:
  `platforms/_records/kaggle/training/pcqm_k1_scale500k_cache_v2/`.
- Failure time: 20.68 seconds, before the first graph was constructed.
- Failure: `ModuleNotFoundError: No module named 'rdkit'` while OGB imported
  `ogb.utils.mol.smiles2graph`.
- The version-2 dependency command installed OGB and PyG with `--no-deps` but
  omitted RDKit. This was an infrastructure packaging error, not a data,
  architecture, optimization, or scientific failure.
- Only `split.json` was emitted. No cache manifest, graph shard, model,
  checkpoint, validation result, official validation/test-dev access, or
  shadow-label read occurred.

The bounded monitor had already consumed its one unchanged-contract retry
(version 1 to version 2), so it did not submit version 3 or the GPU task.
