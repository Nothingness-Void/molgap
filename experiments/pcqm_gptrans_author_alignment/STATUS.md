# GPTrans author/local attribution matrix status

- P0 CPU real-graph path preflight: v1 ended in an infrastructure-only
  source-layout failure; v2 completed and passed independent output
  acceptance. The v1 and v2 receipts and `results/acceptance.json` retain
  distinct physical identities. No training or labels were used.
- I0 CPU real-input initialization-scale diagnostic: v1 failed at a Kaggle
  mount-path assumption before any graph or weight read. Version 2 completed
  with SHA-addressed mount lookup and passed independent per-shard, source,
  row, role and native-cost acceptance in `results/initial_acceptance.json`.
  Both physical receipts remain separate under `results/initial_submission_v*.json`.
- The retained GPTrans V4 checkpoint, 50K aligned prediction, runtime certificate,
  trace and fixed-data identity were recovered without training or inference into
  `recovered_reference/reference_bundle.json`; repository V5 structural validation
  passes. Its original trace reports EMA development MAE only, without a separate
  live-model development metric at each epoch. This fails the V5 strict causal
  trace requirement; the bundle supports a paired endpoint or historical context,
  not an automatic strict-causal GPU release.
- G1 initialization and G2 path-input training: not submitted; prospective
  source/data/role/RML and model-state gates remain incomplete, as does a GPTrans
  T4 runtime calibration. Exact gaps are in `results/gpu_prelaunch_gaps.json`.
  Re-training one baseline with a complete trace would require a separate
  explicit compute decision, not a silent repair to the old record.
  The user subsequently authorized that separate 100K reference rerun;
  its live operational state is in
  `../pcqm_gptrans_v5_audit_reference/STATUS.md`.
- O1 and S1: not released.

The earlier Kaggle2 synthetic/Cython parity kernel is complete and is not
resubmitted under this matrix.
