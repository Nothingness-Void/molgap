# Exact-version qualification inspection, 2026-10-18 JST

Result: **native numerical/update and exact repeat/resume proofs accepted for
both arms, QUALIFICATION_ONLY**. Complete platform allocation/billing cost:
**INSUFFICIENT_EVIDENCE**. Scientific acceptance: **NOT_EVALUATED**.

This inspection changed no model, recipe, training code, remote artifact or
submission. No additional GPU job, scientific training, evaluation or successor
was started. The qualification certificates retained
`native_execution_released=false`; they were not substituted for ordinary
screen runtime certificates.

## Exact binding and artifact retention

- Kaggle2 `kaseichou/molgap-modern2d-native-100k-v4`, numeric
  kernel139279865/version1, was observed COMPLETE without a failure message at
  2026-10-17T22:34:29.178922+00:00. Both isolated workers exited0.
- Private/T4 metadata, both declared dataset sources and remote entry bytes
  matched the frozen release. Source commit:
  `24419d42e49fe3aa1852674c2463bf58a77463a3`; archive SHA256:
  `4d492817f005f0161205baeae08d27e2f9a34fe4a38ba30e4436acae951bcf87`.
- Spec: `a207acd2e09b18f5f04bb310154a58e7799eb4c4f5948397289c2c469b2dbe23`;
  package: `16cc8f4a5410addea47dac547a063a635b9c79c3dc0b6e85c0b153a732e48144`;
  entry: `42083b5485289c74be30d148330c4fcbf574e4401198f89c254614c02d2f4f89`.
  The existing [submission record](submission_2026-10-18.json) retains the
  original launch/authority and private source/data version bindings.
- Existing exact-version, manifest-pinned retrieval retained all28 manifest
  artifacts (701050318 bytes):20 metadata/log files and8 checkpoints
  (701022916 bytes). Each file matched its declared SHA256 and size. The
  bootstrap report and manifest were retained separately. No bulk environment,
  graph payload, unselected output or protected dataset was downloaded.
- The selected source/data roots were the unique owner-namespaced mounts. The
  environment reported Python3.12.14, torch2.7.1+cu126, PyG2.6.1 and two Tesla T4s.

A separate delegated Luna metadata audit passed source/archive, calculated
Spec/package identities, launch/authority, both qualification contexts,
initialization, accepted data/role/version pins, and certificate/provenance/
resource/worker-result cross-bindings. It independently checked all20
non-checkpoint manifest hashes. Runtime contexts retained
`platform_version=null`; version1 was independently bound by the original
submission, authoritative exact-version observation and version-specific output
retrieval, not retrospectively inserted into the frozen runtime contexts.

## Independently checked native state proofs

A delegated Luna worker loaded one checkpoint at a time, CPU-only with
`weights_only=True`, reusing the frozen source's `_state_fingerprint`,
`_probe_recipe`, `read_recipe`, `validate_session_resume` and `state_dict_sha256`.
It did not construct models, read graph payloads, train or initialize CUDA.

Both arms passed:

- Manifest, certificate and retrieval-signal byte hashes matched all8 states.
- The original V1 initialization's tensor names, shapes and dtypes matched the
  retained model states. Endpoint tensors differed from initialization: these
  were actual parameter updates, not initial-state echoes.
- Each of `repeat-a`, `repeat-b` and `resumed` ended at optimizer step2,
  256 sample presentations, next batch2 and a complete Session cursor.
  Every optimizer parameter's step was2.
- The serialized interruption origin was paused at step1,128 presentations and
  next batch1; every optimizer parameter's step was1. Restoring it reached the
  same final model, optimizer, schedule, RNG and cursor fingerprint as both
  uninterrupted repeats.
- Model/optimizer floating tensors were finite. Each state retained one CUDA
  RNG stream, no EMA and no development metric or development selection.

| Arm | Terminal full-state fingerprint | State tensors |
|---|---|---:|
| PPGT/2 `ppgt_author_v2` | `c963ded224c5ffea00b9fa4ac3284e86611f4f3d4fb3d7aed8644c392b5e1028` |370|
| ESA/3 `esa_native_v3` | `ad3faa2c39ac93152a1aefc6c26950929b5f5bed7eb936f544fc24a345194f63` |130|

Each model performed **six scratch updates total across three independent
two-update paths**, not six consecutive scientific-training steps. The frozen
probe enabled finite loss/gradient checks every step; the shared optimizer
owner rejected non-finite loss/gradient before update. Both native certificates
reported `QUALIFICATION_VERIFIED`, six scratch updates and zero formal-training
exposure. The earlier bootstrap/device-interface failures did not recur in this
attempt; their original records and scientific interpretations were preserved.

The accepted input binding remained train100K, seed42 and physical BS128.
Common-Dev permits were frozen as metadata, but the probe disabled development
loading/selection; no Common-Dev MAE, Audit, official validation or test result
was produced. These proofs establish bounded execution/recovery, not model
quality, model ranking or 100K-to-500K transfer.

## Resource observations and cost limits

| Observation | PPGT/2 | ESA/3 |
|---|---:|---:|
| Worker qualification wall seconds |267.216234743|178.341214770|
| Peak allocated GPU bytes |1855784448|1084035584|
| Peak reserved GPU bytes |1876951040|1178599424|
| End-to-end scratch sample presentations/second |2.874076854|4.306351697|

These timings include input admission, construction, repeated execution,
serialized recovery and proof publication. They are not steady-state training
throughput, epoch-time forecasts or a scientific comparison. PPGT used isolated
device0 and ESA device1. The two workers shared one T4x2 allocation; per-arm
allocation-ledger copies had identical hashes and must not be added.

The pair runtime observed331.618697290s including59.284187546s environment
setup. The overlapping allocation ledger observed331.623976288s /
663.247952576 device-seconds. The broader bootstrap Python window observed
333.589550485s /667.179100970 device-seconds, approximately0.1853 observed
T4-hours. These overlapping windows are **not additive**.

Pre-Python provisioning, final deallocation and platform billing/allocation
duration were unavailable. Queue delay is not GPU allocation time. Therefore
none of the observed windows proves complete billed cost or final compliance
with the1 allocated T4-hour ceiling. Full-cost/budget acceptance remained pending;
no missing field was filled with Python wall time, arm-time sums or account-wide
quota differences.

The CPU verifier's peak RSS could not be measured on this Windows run and was
recorded as null. No measured sub800MiB claim was made.

## Retained evidence and unresolved boundaries

Raw evidence and delegated reports were retained at
`E:/MolGap/artifacts/modern2d-native-preflight/retry_v4/inspection_20261017T223429Z`.
Key raw-file SHA256 pins:

| Relative path | SHA256 |
|---|---|
| `platform_observation.json` | `336e04d694023e7b792812f83b52aee654a7cd43533c0809a57e4526aff84b28` |
| `outputs/qualification/qualification_manifest.json` | `d2d752da084b3dfe5a2378aa1a5dda799fed0b960b86b39a5522ed25eb37d33f` |
| `metadata_hash_acceptance.json` | `231c3ae4f8f22eb66800e7041a8fd8a672dd76c32f10b1342e822c6ed4252752` |
| `checkpoint_hash_acceptance.json` | `8ae59232513854f8f691c8fd16b67f48bd16c226d3bc8aacdf2c21734e126000` |
| `outputs/bootstrap_allocation.json` | `e987e7298af71047b8ea961e05126295765210f4530f9cbab39ba818007af351` |
| `outputs/qualification/execution_report.json` | `f8b84ff84fc529cf58fcc44193b1dd9ef059bd75a5ddd70c25fc573249c7614a` |
| `outputs/qualification/allocation_ledger.json` | `4de30f2104308c7210805536a34ffcdf385ec03cc5f7697e2559fb7816ce5b73` |
| `luna_states/checkpoint_verification.json` | `87b254c73fba3ce1b169a7f05b4081acd0cc4db541d787db5dd923d5578dff95` |
| `luna_states/verify_qualification_checkpoints.py` | `2e0d8388fdfa153cf445fdcef34f26f488211e1b5a34021dd259a02cfb2a2d44` |
| `scratchinspection/luna_metadata/native_certificate_audit.json` | `6ef178b713b9bba59788f9d420d1895b60c1710818a294cff7cdbe6b3b9f15b1` |

All14 pre-existing dirty-file SHA pins matched the preserved review record,
including `tests/test_research_memory.py`. No derived RML index was rebuilt.
Terminal RML publication was not performed by this execution inspection;
prospective snapshots do not imply live jobs or completed scientific evidence.
Full allocation accounting, terminal publication and any separately authorized
formal-training release remain distinct from the accepted short native proofs.
