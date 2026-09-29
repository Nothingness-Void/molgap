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
the caller-pinned authorization-envelope digest and declarations on the host.
Its return value is not real-data verification or a report.

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
manifest, not this authorization envelope. GPTrans `files` records retain the
original `path`, `sha256`, positive integer `bytes`, and `role` fields. K1 records
have those fields plus a positive integer `rows`. This is an additive,
family-specific manifest-v1 record shape; existing GPTrans envelopes and their
meaning do not change. Roles must cover precisely the family spec's declared
roles: GPTrans train/development, K1 train only. K1 does not create development,
official-validation or test roles. Paths are relative POSIX paths under the
explicit root; aliases, traversal, case collisions, Windows devices/streams,
symlinks, junctions and reparse points are rejected. K1 also rejects hardlinked
fixed manifests and shard inputs. The original data root is never passed to
family code.

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

K1 `neural_atom_k1/1` baseline and `k1_pair_value/1` use only the package's
`molgap.pcqm_k1_full_runner`. K1 selects one or more topology records from the
frozen official full-train PCQM4Mv2 manifest. The fixed manifest's parsed
canonical digest must match `FULL_MANIFEST_CANONICAL_SHA256`; its full identity,
roles, graph/source contract, 68-shard topology aggregate and 3,378,606-row
boundaries must match the frozen runner constants. Every selected record's
path, SHA256, bytes, rows and `train` role must match one of those topology
records exactly. Selected rows must total at least the frozen physical batch
128. No unselected shard is needed or represented as loaded.

K1 uses two isolated package-only worker phases after host validation of the
authorization envelope. The host stages the fixed manifest in a private arm
directory; the first worker binds its frozen full topology and authorizes the
selected declarations as metadata only, without reading or verifying shard
content. The host then copies only the selected files into that private
directory and checks their hashes. The second worker rebinds the frozen
topology, matches selected path/SHA256/bytes/rows/train-role declarations, and
verifies staged file size and SHA256 before using the full trainer's packed-graph
decoder. It checks actual decoded row counts before collating a batch. Neither
worker receives the original data root as a family-loader input.
`_load_selected_preflight_graphs` is separate from the unchanged full-role
`_load_graphs` row gate. The worker calls the unchanged K1 `_loader`
with pass 0/start batch 0, the original physical batch and PyG collate path;
only `LOADER_WORKERS` is set to zero in that private CPU process. It checks
per-shard decoded rows and one batch's graph count, x/edge/RWSE dimensions,
finite RWSE/targets and CPU placement. This does not claim full-role sampler
parity, target statistics, training, or model execution. The pair-value addon
must be present in the frozen package with the declared source-byte SHA256 and
its known interface; this is not an addon model smoke.

Missing package K1/GPTrans loader source is unsupported. Missing, synthetic,
misdeclared, linked, undersized or incompatible K1 inputs cannot become a
loader-success report.

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
`missing_evidence`, error and status preserve that distinction. Successful loader
inspection is `LOADER_VERIFIED_ONLY`, never complete numerical qualification.
For K1 alone, loader success is instead
`SELECTED_REAL_SHARDS_LOADER_VERIFIED_ONLY` with
`verification_scope="selected_real_shards_only"` and `selected_real_shards`
listing the validated path/SHA256/rows/train role for each selected record.
These two report fields are K1-only, preserving GPTrans report fields. Failed
K1 reports leave scope null and the selected list empty. K1 `missing_evidence`
retains full-role content/assembly and model numerics, plus remaining train shards when
fewer than 68 were selected. Even selecting all 68 does not certify full-role
assembly or the old full-run preflight.
CLI exit code 0 for this K1 status means only that the selected train-shard
loader/collate diagnostic completed under this contract; it is not full-role,
training, model, replay or READY admission.
When loader-only reports have different success strings, the summary is
`ALL_ARMS_VERIFIED_WITHIN_SCOPE` only if every GPTrans arm is
`LOADER_VERIFIED_ONLY` and every K1 arm is
`SELECTED_REAL_SHARDS_LOADER_VERIFIED_ONLY`. Each arm retains its original
status and evidence; valid identical-status summaries remain unchanged. An
identical success string on an arm of the wrong family or in the wrong mode is
non-passing, not an exception to the family-specific check. The CLI
returns 0 for this aggregate status, meaning only that every arm completed its
own declared loader diagnostic. It does not establish full-role coverage,
training or model success, cross-family fairness, RML, replay or READY status.
Any missing, failed, invalid or unsupported arm, including K1 model-smoke,
remains non-passing (`MIXED_NONPASS` for different outcomes).
`requested_device` is a declaration; `device` remains null until a CPU batch is
actually observed, including when source verification succeeds without loading.
Other mixed outcomes are `MIXED_NONPASS`. No RML, READY or replay authority fields are
emitted, and no canonical research records are modified.

This is trusted-code process isolation, not an OS security sandbox. Packed
PyTorch files use the frozen pickle-capable loader and must be trusted. Hashes
bind bytes and explicit declarations; they do not independently prove scientific
role provenance or protect against an adversary controlling the process/OS.
Dependencies are host-installed; their versions are not certified here.

## Unexecuted Test Handoff

Tests were authored, not run as part of this implementation. Synthetic tests
exercise rejection, routing, mocked cross-family summary statuses and
decoder/collate interfaces without establishing real-shard preflight evidence.
The real dual-arm integration
test skips unless `MOLGAP_PREFLIGHT_REAL_INPUTS` points to a user-authorized JSON
configuration with `spec`, `package_dir`, `expected_package_identity`,
`shard_manifest`, `shard_root`, and `expected_shard_manifest_sha256`. It requires
two GPTrans arms, real frozen artifacts and enough CPU RAM/disk for full private
copies. It deliberately makes the second arm's shard missing while preserving
the first arm's real loader result. It is parameterized over both modes; only the explicitly selected smoke case constructs a model.

An optional K1 integration case uses `MOLGAP_PREFLIGHT_K1_REAL_INPUTS` with the
same pin/configuration keys. It requires an explicitly authorized K1-only spec,
frozen package and selected real train shards, and asserts shard-scoped loader
reports only. Synthetic tests exercise topology matching and collate interfaces
without emitting real-shard success or reading protected data. This case was
also authored but not run here.

Suggested Luna commands, from this isolated worktree, using a project virtualenv
with torch, NumPy, PyG and the frozen source package's dependencies installed:

```powershell
$env:PYTHONPATH = (Join-Path $PWD 'src')
& '.\.venv\Scripts\python.exe' -m pytest tests/test_experiment_preflight.py -k 'not real_dual_arm and not real_k1' -q
& '.\.venv\Scripts\python.exe' -m pytest tests/test_experiment_preflight.py -k real_dual_arm -q
```

The `.venv` must be provisioned in this worktree before these commands are used.
The real integration cases require explicit authorization and
`MOLGAP_PREFLIGHT_REAL_INPUTS`; do not run them against protected roles. Both arms
must declare random initialization with the correct model state digest. The
second command runs the real CPU model smoke as well as loader inspection, so
it requires separate execution authorization and enough time/RAM for GPTrans-T.
The optional K1 case is excluded from both commands and requires separate
authorization. Neither command was executed by the implementer.

## Explicit Model Smoke V1

Set `mode="gptrans-model-smoke-v1"` to request the diagnostic. Unknown versions
raise before creating output. The mode is recorded in each arm and the summary.
The device is always CPU; no accelerator selection or fallback is supported.
Missing real data, missing dependencies, and invalid source remain structured
non-pass outcomes. K1 model-mode requests return `UNSUPPORTED_MODEL_SMOKE`
with a structured `UnsupportedMode` error and no model observations, regardless
of whether K1 loader-only inputs were provided. A successful loader cannot satisfy an
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
