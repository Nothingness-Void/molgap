# Experiment Source / Real-Shard Preflight V1

`molgap.experiment_preflight` defaults to a CPU loader-only boundary. An explicit
versioned model-smoke mode adds one CPU train-batch diagnostic. Neither mode is
a training launcher, platform submitter, acceptance transaction, or source of
scientific authority. No status is named PASS.

## API

```python
run_experiment_preflight(
    spec, package_dir, output_root,
    expected_package_identity=trusted_package_identity,
    shard_manifest=authorized_manifest_path,
    shard_root=real_dataset_root,
    expected_shard_manifest_sha256=trusted_manifest_digest,
    timeout_seconds=300.0,
    mode="loader_only",  # optional; model_smoke_v1 explicitly opts into compute
)
```

`spec` is exactly ExperimentSpec. Pins must come from the caller's authorized
records, not be recomputed from an untrusted input and treated as authentication.
Output must not exist, including an empty directory; its parent must exist.
Invalid API/spec/output arguments raise. Package, manifest and per-arm artifact
failures produce structured non-passing reports. No existing output is resumed.

`validate_real_shard_manifest(spec, manifest_path, expected_sha256)` checks
declarations only. Its return value is not real-data verification or a report.

## Manifest Contract

The manifest is UTF-8 canonical JSON: sorted keys, compact separators, ASCII
escaping, no nonfinite numbers, duplicate keys, extra fields, BOM or newline.
Its format is `molgap-real-shard-preflight-manifest-v1` with exactly:

- `spec_identity`: the exact ExperimentSpec identity.
- `arms`: a list, permitting omitted arms to remain missing rather than guessed.
- `format`: the version above.

Each arm entry has exactly `arm_id`, `arm_identity` (canonical arm fingerprint),
`data` (the exact complete arm data object), `kind` (`real-packed-pcqm`),
`fixed_manifest`, and `files`. `data` binds dataset, split, features, target and
each role's membership, row-order and usage digests without changing Spec.

`fixed_manifest` has `path` and `sha256`. It points to the frozen family dataset
manifest, not this authorization envelope. Each `files` record has `path`,
`sha256`, positive integer `bytes`, and `role`. Roles must cover precisely the
arm's declared roles and may only be train/development. Paths are relative
POSIX paths under the explicit root; aliases, traversal, case collisions,
Windows devices/streams, symlinks, junctions and reparse points are rejected.
Every listed byte is copied into a private arm directory and hashed there.
The original data root is never passed to family code.

## Loader Support

GPTrans-T v1 uses only the package's `molgap.pcqm_gptrans_v4`:
`validate_fixed_assets`, `_load_datasets`, `_training_loader(epoch=0)` and
`_development_loader`. The authorized shard list must equal the fixed manifest's
ordered records, and that manifest's bytes must match the frozen module's
`MANIFEST_SHA256`. All three original packed files for the complete 100K train /
50K development contract are required. This version does not support tiny
subshards, 500K/full variants or fabricated fixtures. It constructs one batch
per role and checks observed size, feature dimensions, CPU placement and finite
targets. It does not call `_forward`.

For CPU inspection only, `LOADER_WORKERS` is set to zero in the isolated process.
Sampler, seed, batch size, role and source files are not rewritten. This avoids
nested DataLoader processes and is explicitly not production runtime parity.

K1 is `UNSUPPORTED_FAMILY_PREFLIGHT`: this boundary has no approved frozen PCQM
loader integration for that family. It never borrows code from another checkout.
Missing package GPTrans loader source is likewise unsupported.
For a declared GPTrans arm, real-shard presence and staging are checked before
unpacking or inspecting the package loader. Therefore an absent shard root or
listed shard is `MISSING_REAL_SHARD`, even when the package also lacks GPTrans
source; a K1 arm remains `UNSUPPORTED_FAMILY_PREFLIGHT`.

## Model Smoke V1

Only `mode="model_smoke_v1"` enables model computation. Unknown modes, non-string
modes and callback/provider keyword arguments are rejected before output creation.
The default API and V1 loader report remain unchanged. Smoke reports use
`molgap-experiment-model-smoke-v1`, the same independent observed boolean fields,
and additional `observations` and `checkpoint` objects. Package, manifest, shard,
spec and arm bindings still use the existing verification and private staging.

Before importing the runtime, smoke checks normalized source hashes for seven
reviewed numerical/adapter modules at ae7674d. Missing or changed code is non-pass,
not a fallback to the host. This deliberately rejects synthetic/stub model and
loader implementations, even inside an otherwise internally valid source package.
Updating those implementations requires reviewing/versioning this smoke contract.
These pins supplement trusted package identity; they are not a security sandbox
or a complete dependency/software certificate.

Both real role batches must be built before model work. Only the train batch is
used for forward/backward/step; development supplies loader evidence only. Full
100K training targets supply `_target_stats`; no protected role is read. The
package's `ExperimentSpec`, `gptrans_metadata` and `build_gptrans_model` select
reference, pair_prenorm or centered_logits, without host family imports.
Initialization is seeded fresh CPU FP32 initialization, not verification of a
declared frozen initial-state artifact. No CUDA path is enabled.

The bounded step reuses `_forward`, `make_adamw_compat`, `FrozenEpochScheduler`,
`ExponentialMovingAverage`, `model_state_sha256` and `assert_finite_state_dict`.
It follows `_optimizer_step`'s normalized-gap-L1, clipping and EMA order, splitting
the operations solely to record independent observations. It checks prediction
and normalized target shapes/finiteness, finite loss and gradients, finite
clipped norm, changed model state, finite model/EMA/optimizer tensors, and actual
AdamW step cursors equal to one. Loss, shapes, statistics, gradient norm/count,
state hashes and observed step count are recorded. It does not run a trainer.

Checkpoint is deliberately `CHECKPOINT_UNSUPPORTED` with `checked=false` and
`checkpoint_roundtrip_checked=false`. `_make_training_state` forces CUDA;
`_save_checkpoint` emits certificate-bound scientific fields. No runtime
certificate is supplied at this boundary, so it does not invoke that helper with
fabricated identity, write a substitute trainer checkpoint, or claim restoration
of model/optimizer/scheduler/EMA/RNG. Missing helper availability is recorded.
The highest attainable status in this version is
`MODEL_SMOKE_CHECKPOINT_UNSUPPORTED`, not `MODEL_SMOKE_VERIFIED`. A completed
forward/backward/step is retained as observed, but checkpoint evidence remains
missing. Errors use `MODEL_SMOKE_FAILED`; one arm never cancels its sibling.

## Isolation And Reports

The existing `verify_experiment_source_package` verifies a private snapshot of
the six package files. Its package/spec identity must match the caller's pins.
Each arm extracts an independent source tree, manually accepting only regular
tar members and rejecting duplicate members and unsafe paths. No `extractall`
is used. Each worker starts with `python -I -S`, no Python environment variables,
no site startup or editable-install `.pth` processing, and hidden accelerators.
Only explicit interpreter dependency directories are added. A meta-path guard
rejects molgap imports outside the unpacked source tree. Loaded module origins
are checked before data loading and after batching, and recorded relative to
that arm's unpacked root. The host provides the stdlib bootstrap, not family code.

Arms run sequentially in separate processes with independent timeouts and
temporary trees. Failure in one arm does not cancel another. Package or global
authorization-envelope corruption blocks all arms. For a declared GPTrans arm,
missing real-shard files are classified before package loader inspection; an
arm's file corruption blocks that arm. Temporary copies are removed after
completion.

Each `<output>/<arm_id>/preflight.json` and the final `<output>/preflight.json`
are canonical JSON, written by same-directory temporary file, fsync and atomic
replace. The summary is published last. Partial output after a crash is not a
completed summary and cannot be overwritten by a retry. Reports bind observed
spec/package/manifest/arm identities. Expected pins are separately labeled in
the summary; invalid artifact identities remain null rather than assumed valid.

The independent observed flags are `package_verified`, `shard_verified`,
`loader_batch_built`, `forward_checked`, `backward_checked`,
`optimizer_step_checked`, and `checkpoint_roundtrip_checked`. False means no
successful observation, not necessarily that the operation ran and failed.
`missing_evidence`, error and status preserve that distinction. Successful loader
inspection is `LOADER_VERIFIED_ONLY`, never complete numerical qualification.
`requested_device` is a declaration; `device` remains null until a CPU batch is
actually observed, including when source verification succeeds without loading.
Mixed outcomes are `MIXED_NONPASS`. No RML, READY or replay authority fields are
emitted, and no canonical research records are modified.

This is trusted-code process isolation, not an OS security sandbox. Packed
PyTorch files use the frozen pickle-capable loader and must be trusted. Hashes
bind bytes and explicit declarations; they do not independently prove scientific
role provenance or protect against an adversary controlling the process/OS.
Dependencies are host-installed; their versions are not certified here.

## Unexecuted Test Handoff

Tests were authored, not run as part of this implementation. Synthetic tests
exercise rejection and blocked semantics only. The real dual-arm integration
test skips unless `MOLGAP_PREFLIGHT_REAL_INPUTS` points to a user-authorized JSON
configuration with `spec`, `package_dir`, `expected_package_identity`,
`shard_manifest`, `shard_root`, and `expected_shard_manifest_sha256`. It requires
two GPTrans arms, real frozen artifacts and enough CPU RAM/disk for full private
copies. It deliberately makes the second arm's shard missing while preserving
the first arm's real loader result. It does not construct a model.

Suggested Luna commands, from this worktree, with the project virtualenv:

```powershell
Set-Location 'D:\w\ms1'
$env:PYTHONPATH = 'D:\w\ms1\src'
& 'D:\文档\molgap\.venv\Scripts\python.exe' -m pytest tests/test_experiment_preflight.py -k 'not real_dual_arm and not real_model_smoke' -q
& 'D:\文档\molgap\.venv\Scripts\python.exe' -m pytest tests/test_experiment_preflight.py -k real_dual_arm -q
& 'D:\文档\molgap\.venv\Scripts\python.exe' -m pytest tests/test_experiment_preflight.py -k real_model_smoke -q
```

Ensure that the test environment resolves `molgap` from this worktree's `src`,
not the main checkout's editable installation. The real test additionally needs
the explicit environment configuration described above. Neither command was
executed by the implementer.

The model test separately skips unless `MOLGAP_MODEL_SMOKE_REAL_INPUTS` points to
an explicitly authorized configuration with the same six fields described above.
It needs a package containing the reviewed adapter/runtime sources and their
dependencies. It performs real CPU model computation only when explicitly enabled;
allow substantial RAM/time for physical batch 128 and the unchanged 12-layer
model. It expects successful step observations plus unsupported checkpoint for
one arm and a missing-shard failure for the other. Use separately authorized
configs to cover each of the three adapter variants. No fake Torch graph is
used to assert numerical success. All commands above are a handoff, not an
implementation-time verification claim.
