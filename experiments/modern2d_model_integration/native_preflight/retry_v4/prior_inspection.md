# Exact-version inspection, 2026-10-18 JST

Result: **INFRASTRUCTURE_FAILURE / NATIVE_NOT_ACCEPTED** for Kaggle2
`kaseichou/molgap-modern2d-native-100k-v2`, kernel139267796/version1.
This inspection repaired no code and made no remote submission. The two
scientific routes received neither a score nor a negative architecture result.

## Retained observations

- The authenticated platform observation at2026-10-17T17:40:28.606718+00:00
  reported ERROR. Private/T4 metadata, source/data references and entry bytes
  still matched the submitted package and source commit1caf5356.
- Exact-version retrieval obtained the bootstrap report, qualification manifest,
  execution report, allocation ledger and environment reports. Shared pinned
  retrieval verified every one of the manifest's13 artifacts, including both
  worker logs and failure records; no graph dataset or bulk output was downloaded.
- Both mounts selected the unique `datasets/kaseichou/<slug>` location. The
  earlier V1 bootstrap-location failure did not recur. The environment reported
  Python3.12.14, torch2.7.1+cu126 and torch-geometric2.6.1; two Tesla T4s were observed.
- Each worker reached its first `model(batch)` return and then raised
  `ValueError: Model output device differs from its input batch` at packaged
  `shared_model_adapter.py:260`, called by `FamilyKernel._model_forward` at165.
  No backward or optimizer step completed on that path. Repeat/resume checkpoint
  proofs, finite-output certificates and worker qualification results were absent
  from the sealed manifest. Formal scientific training exposure remained0.
- ESA exited1. The supervisor terminated its PPGT sibling (returncode-15), but
  the retained PPGT failure log independently contained the same device exception;
  this was not evidence that only ESA failed while PPGT passed.

## Device-check diagnosis and limits

The packaged preflight passed its `device="cuda"` default into `FamilyKernel`,
which retained `torch.device(device)` as `self.device`. Model and batch transfers
used that device. The output validator compared `output.device` directly with
the retained, potentially unindexed request, rather than the staged batch's
concrete tensor device. The existing sibling `forward_normalized_gap` instead
passed `batch.y.device` to the same validator.

This wiring exposes a CUDA alias-equality false rejection: an unindexed CUDA
request and a concrete local CUDA device are different device objects even when
the request transfers tensors to that device. This is the leading diagnosis
from frozen code and the matching two-arm traceback. The worker exception did
not log actual output/input device values, so those native values were not
retrospectively invented. The submission's31 CPU preparation/identity checks
had not established this native CUDA-forward behavior.

A delegated CPU-only micro-verification confirmed
`torch.device("cuda") != torch.device("cuda:0")` without querying or initializing
CUDA. The frozen validator accepted a matching CPU device but reproduced its
device exception for the corresponding `cpu`/`cpu:0` descriptor mismatch. A
control-flow fixture left its simulated optimizer-step count at0 after that
exception; the real before-backward ordering was independently visible in the
frozen core. This proved the descriptor-comparison defect, not actual native
output values or a repaired GPU execution. No model or graph was constructed.

A minimal subsequent repair should bind validation to the concrete staged input
device (and retain strict rejection of genuinely different devices), with CPU
device-alias/control-flow checks before any separately authorized native retry.
Neither suppressing validation nor changing model/data contracts is justified.
Finite numerics, backward, six-update repeat/resume, throughput and memory still
need native qualification even if this device gate is repaired.

## Cost and evidence boundary

Bootstrap Python observation lasted101.589091535s across two T4s,
203.17818307 device-seconds. The overlapping allocation ledger recorded
101.561267313s/203.122534626 device-seconds; these windows must not be summed.
Pre-Python provisioning, queue and final platform allocation/billing were missing,
so neither value is accepted as the complete billed GPU cost.

Raw evidence and delegated CPU diagnostic outputs were retained under
`E:/MolGap/artifacts/modern2d-native-preflight/retry_v2/inspection_20261017T174028Z`.
Key SHA256 pins:

- Bootstrap: `a4cb0844ab9cac7607395c3a48ce5dcd1e0459a514afc9cd20f22d26df4f4b2d`.
- Manifest: `f72354c0ace7f2a21beaf4331702bb9205e76c71c7d544250df3ef318e02db3a`.
- Execution report: `9daa176b60b54f4929d48cdddca0b9f0e21e41334512b4505f85f401644b50ce`.
- Both worker failure records: `1844caaa846f334608b8d06a8d72b33e971a4339248d3be107b4ca3b603eff5a`.
- Manifest-bound retrieval: `ac6cf20e081c2a14c917befd87d610fa86b7ac3cbf552c485f20f13cdcd69308`.
- Delegated device-check script: `a960f9319be1337a9cd8a2cb32209cb64d6826b479e4e0dae2110834951e4fe3`.
- Delegated device-check result: `5c6d5275dcce95460be5779dd111c739a8297e87499ef30f9f60689bd9370eef`.

Terminal RML publication was not performed by this read-only execution
inspection. Existing prospective snapshots, historical V4/V5 conclusions,
data/weight/source packages and unrelated user changes were preserved.
The later optional RML closeout did not produce a completed validator result;
its validation command was interrupted, its frozen check was not started and
its user-pin extraction did not complete. Submission-time RML and14-pin checks
remain historical evidence, not independently repeated inspection-time results.
