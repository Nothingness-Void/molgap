# Experiment Source / Real-Shard Preflight V1

`molgap.experiment_preflight` defaults to a CPU loader-only boundary. It is not a production
preflight, training launcher, platform submitter, acceptance transaction, or
source of scientific authority. No status is named PASS. In default `loader-only-v1`, no model is built;
forward, backward, optimizer step and checkpoint round-trip remain unobserved.

## API

```python
run_experiment_preflight(
    spec, package_dir, output_root,
    expected_package_identity=trusted_package_identity,
    shard_manifest=authorized_manifest_path,
    shard_root=real_dataset_root,
    expected_shard_manifest_sha256=trusted_manifest_digest,
    timeout_seconds=300.0,
    mode="loader-only-v1",  # or explicit "gptrans-model-smoke-v1"
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
targets. The default mode does not call `_forward`.

For CPU inspection only, `LOADER_WORKERS` is set to zero in the isolated process.
Sampler, seed, batch size, role and source files are not rewritten. This avoids
nested DataLoader processes and is explicitly not production runtime parity.

K1 (`neural_atom_k1/1`) and EdgeState (`edge_state_gps/1`) support
`loader-only-v1` for accepted **topology** manifests. K1 uses the frozen
`ogb-train-full` manifest and selects exactly one train shard. EdgeState uses
one of the accepted `ogb-train-100k`, `ogb-train-500k-scnet-v1`, or
`ogb-train-1m` manifests and selects exactly one train and one development
shard. Select records directly from the fixed manifest's `assets.topology`;
geometry files and extra shards are rejected. The source package must include
`pcqm_topology.py`, `ogb_features.py`, `screen_policy.py`, `v4_runtime.py`,
`training_reproducibility.py`, and their package dependencies. The worker
checks the fixed manifest's canonical frozen fingerprint, selected shard bytes,
size and row range, finite targets, OGB atom9/bond3/RWSE16 features, and one
ordered CPU batch per role. It normalizes PyG's automatic `row_index` batch
offset before comparing source rows.

The outcome is `PARTIAL_LOADER_VERIFIED_ONLY`, with
`scope=selected-topology-shards`. It does **not** validate the other shards,
full-role sampler, target statistics, model construction, or trainability.
`gptrans-model-smoke-v1` remains GPTrans-only. Missing family source is
unsupported; missing selected data is missing, not pass.

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
authorization-envelope corruption blocks all arms. An arm's file corruption
blocks that arm. Temporary copies are removed after completion.

Each `<output>/arms/<arm_id>/preflight.json` and the final `<output>/preflight_summary.json`
are canonical JSON, written by same-directory temporary file, fsync and atomic
replace. The summary is published last. The old root `preflight.json` is never written. Partial output after a crash is not a
completed summary and cannot be overwritten by a retry. Reports bind observed
spec/package/manifest/arm identities. Expected pins are separately labeled in
the summary; invalid artifact identities remain null rather than assumed valid.

The independent observed flags are `package_verified`, `shard_verified`,
`loader_batch_built`, `forward_checked`, `backward_checked`,
`optimizer_step_checked`, and `checkpoint_roundtrip_checked`. False means no
successful observation, not necessarily that the operation ran and failed.
`missing_evidence`, error and status preserve that distinction. GPTrans's
complete frozen-shard loader result is `LOADER_VERIFIED_ONLY`; K1/EdgeState's
selected-shard result is `PARTIAL_LOADER_VERIFIED_ONLY`. Neither is complete
numerical qualification.
`requested_device` is a declaration; `device` remains null until a CPU batch is
actually observed, including when source verification succeeds without loading.
Mixed outcomes are `MIXED_NONPASS`. No RML, READY or replay authority fields are
emitted, and no canonical research records are modified.

This is trusted-code process isolation, not an OS security sandbox. Packed
PyTorch files use the frozen pickle-capable loader and must be trusted. Hashes
bind bytes and explicit declarations; they do not independently prove scientific
role provenance or protect against an adversary controlling the process/OS.
Dependencies are host-installed; their versions are not certified here.

## Real-Input Test Handoff

Synthetic tests cover loader mechanics and rejection, not real accepted data.
The real GPTrans dual-arm integration
test skips unless `MOLGAP_PREFLIGHT_REAL_INPUTS` points to a user-authorized JSON
configuration with `spec`, `package_dir`, `expected_package_identity`,
`shard_manifest`, `shard_root`, and `expected_shard_manifest_sha256`. It requires
two GPTrans arms, real frozen artifacts and enough CPU RAM/disk for full private
copies. It deliberately makes the second arm's shard missing while preserving
the first arm's real loader result. It is parameterized over both modes; only the explicitly selected smoke case constructs a model.

Suggested local commands from the desktop checkout, with torch, NumPy, PyG and
the frozen source package's dependencies installed:

```powershell
$env:PYTHONPATH = (Join-Path $PWD 'src')
& '.\.venv\Scripts\python.exe' -m pytest tests/test_experiment_preflight.py -k 'not real_dual_arm' -q
& '.\.venv\Scripts\python.exe' -m pytest tests/test_experiment_preflight.py -k real_dual_arm -q
```

Use the project `.venv` or another environment with the same dependencies.
The real integration cases require explicit authorization and
`MOLGAP_PREFLIGHT_REAL_INPUTS`; do not run them against protected roles. Both arms
must declare random initialization with the correct model state digest. The
second command runs the real CPU model smoke as well as loader inspection, so
it requires separate execution authorization and enough time/RAM for GPTrans-T.
K1/EdgeState real-input loader execution requires their authorized packed files;
the repository contains manifests but not those `.pt` assets.

## Explicit Model Smoke V1

Set `mode="gptrans-model-smoke-v1"` to request the diagnostic. Unknown versions
raise before creating output. The mode is recorded in each arm and the summary.
The device is always CPU; no accelerator selection or fallback is supported.
Missing real data, missing dependencies, invalid source, and unsupported K1
remain structured non-pass outcomes. A successful loader cannot satisfy an
explicit model request. `MODEL_SMOKE_FAILED` retains earlier observations but
never claims the checkpoint passed. Any mixed arm outcomes are `MIXED_NONPASS`.

The fixed dispatch supports GPTrans-T family version 1 with baseline,
`pair_prenorm` version 1, or `centered_logits` version 1. Model construction uses
only the verified package's `_make_model(variant=...)`; spec text is never an
import path. V1 supports declared random initialization only. Frozen-state
initialization is rejected because this API accepts no initial-state artifact.
After seeding with `configure_fp32_determinism`, the constructed state's hash
must equal `initialization.state_sha256` before any forward operation.
`initial_state_checked` remains false until that comparison passes; the observed
`initial_state_sha256` is null until measured.

Both real train and development loader batches must validate first. The smoke
uses the retained train batch and `_target_stats` over the complete train shards.
It uses the package's `_forward`, finite normalized-gap-L1, backward, finite
gradients, frozen gradient clipping, and exactly one non-fused AdamW step via
`make_adamw_compat`. The actual scheduler API is
`FrozenEpochScheduler(optimizer).step(0)`. It never calls `set_epoch` or the
CUDA-only `_make_training_state`. The observed finite loss is recorded as
`normalized_gap_l1`; otherwise that field is null. One fixed physical batch and
the per-arm timeout bound this diagnostic; there is no epoch loop or evaluation.

The diagnostic checkpoint has its own `molgap-diagnostic-checkpoint-v1` format.
It uses `atomic_torch_save` and `torch_load_compat`, then compares the loaded
payload exactly against a detached snapshot. It restores and compares model,
optimizer, scheduler, and available Python/NumPy/Torch/CUDA RNG state, including
the loader generator when present. Tensor dtype, shape, device and bytes must
match; no numeric tolerance is used. Serialization, restore, or comparison
failure prevents `checkpoint_roundtrip_checked` from becoming true.

This checkpoint lives only in the private temporary arm workspace and is removed
after completion. It is not a production checkpoint, resumable training output,
replay certificate, or admission evidence. Production `_save_checkpoint` is
never called, and no runtime certificate is fabricated. Success is named
`MODEL_SMOKE_VERIFIED_ONLY`, not PASS. It grants no training, RML, READY, replay,
or downstream authority. CPU observations do not qualify any GPU/DCU runtime.
