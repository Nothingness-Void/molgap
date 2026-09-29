# GPTrans author/local attribution matrix status

- P0 CPU real-graph path preflight: v1 ended in an infrastructure-only
  source-layout failure; v2 completed and passed independent output
  acceptance. The v1 and v2 receipts and `results/acceptance.json` retain
  distinct physical identities. No training or labels were used.
- G1 initialization and G2 path-input training: not submitted; prospective
  source/data/role/reference/RML and model-state gates are incomplete. Exact
  prelaunch gaps are in `results/gpu_prelaunch_gaps.json`; retained V4 reference
  checkpoint/prediction hashes verify, so no baseline retraining follows merely
  from the missing V5 bundle.
- O1 and S1: not released.

The earlier Kaggle2 synthetic/Cython parity kernel is complete and is not
resubmitted under this matrix.
