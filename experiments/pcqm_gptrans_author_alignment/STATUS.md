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

## 2026-09-30 input-dependency release

The user authorized one G1/G2 dual-arm comparison against the separately
accepted complete V5 reference. The CPU-only preparation package passed
`check-release`, and its rechecked Kaggle POST returned physical version 1,
kernel ID 136492377. Source dataset creation returned Ok/ready; pulled remote
entry, fixed-data mounts and GPU-disabled metadata matched the released files.
The first qualified status observation was RUNNING. Identity and source hashes
are in `preparation_submission.json`; the existing server B owns only mechanical
monitoring through `monitor_binding.json`. No GPU arm was released at that
observation, and no input-stage result was a model-science conclusion.

CPU output acceptance: `accept_preparation.py --output-root <retrieved
gptrans_author_inputs> --staged-package
platforms/_records/kaggle/packages/gptrans_author_inputs_preparation_v4
--acceptance-output <record.json>`. It checks complete paths/source hashes,
independent CPU rederivation, exact two-table transformation, role exclusions
and native cost. Acceptance does not substitute for separate per-arm prospective
RML/comparison admission and a freshly bound GPU release package.

The request-to-submission five-minute target was missed. The recorded Kaggle
kernel POST itself took 1.516 seconds; that excludes input engineering,
preparation, upload and repeated release verification.

The CPU v1 task later ended with ERROR during independent chunk rederivation.
Its infrastructure diagnosis, original evidence retention and release boundary
are in [the failure record](results/preparation_failure_v1.md). The local
chunk-index fix does not qualify complete input acceptance or GPU compute.
