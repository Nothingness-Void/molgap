# K1 500K cache v3 terminal evidence

- Kernel: `kaseichou/molgap-pcqm-k1-scale500k-cache`, version 3.
- Terminal state: `ERROR`; no GPU training was submitted.
- The job installed RDKit and constructed atomic graph shards before stopping.
- Terminal row: source index `3003839`, whose five-valent Si SMILES produced
  `MolFromSmiles=None`; OGB then raised `AttributeError` in `smiles2graph`.
- Retrieved evidence is retained under
  `platforms/_records/kaggle/training/pcqm_k1_scale500k_cache_v3`.

This failure exposed a contract defect rather than a model result: the draft
random 500K/10K split included the invalid row and regenerated the shadow
exclusion instead of consuming the accepted role identities. The user then
required exact SCNet data matching. Version 4 therefore uses source indices
`0..499999` for train and `500000..549999` for development, pins the recorded
RDKit release, and forbids row replacement or omission. No v3 artifact is
eligible for training or scientific comparison.
