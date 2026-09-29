# GPTrans author/local attribution matrix status

- P0 CPU real-graph path preflight: Kaggle2 v1 exited before reading a graph;
  the accepted cache hash check passed, but `torch.load` could not import the
  `molgap` package from the minimal source archive. See `results/submission_v1.json`.
  A source-layout-only repair is prepared for v2; v1 is preserved.
- G1 initialization and G2 path-input training: not submitted; prospective
  source/data/role/reference/RML and model-state gates are incomplete.
- O1 and S1: not released.

The earlier Kaggle2 synthetic/Cython parity kernel is complete and is not
resubmitted under this matrix.
