# V4 Execution Contract

The V4 execution layer keeps the scientific comparison contract independent
from platform setup and model-family code. A run is identified by four inputs:
the immutable data manifest, the scientific contract, a versioned source bundle,
and a small model/run configuration. Runtime certification belongs to a
platform plus its exact software and accelerator tuple.

## Standard Flow

1. Audit the runner and the code it executes with `molgap.v4_cli audit`.
2. Package an explicit committed allowlist with `molgap.v4_cli bundle`. The
   archive is content addressed and deterministic. Its source commit, SHA-256,
   and per-file inventory travel as sidecars.
3. Install that bundle once under the owner's persistent account directory on
   each remote platform. Keep the installed directory keyed by archive SHA.
4. Prepare one JSON run spec for each candidate. Architecture settings belong
   in `model_config`; shared data, optimizer, schedule, target, selection,
   exposure, and role-access identities belong in `scientific_contract`.
5. Run the short platform preflight once for each previously unseen runtime
   tuple. Save its accepted runtime certificate. A training run must present
   that certificate and must match the certified software/runtime fingerprint.
6. Compare the candidate with the one frozen reference through
   `validate_reference_screen_contract`; do not retrain the reference.

When source SHA is already installed, changing only `model_config` or output
paths requires transferring the small run spec, not the source tree. A code
change creates a new bundle only from the explicitly selected dependency
allowlist. The source bundle should not include datasets, model weights, virtual
environments, logs, or generated caches.

## Run Spec

`molgap-v4-run-spec-v1` separates `model_config_sha256` from
`scientific_contract_sha256`. This lets same-family variants share the same
data and training fingerprints while retaining distinct architecture identity.
The V4 dispatcher calls the family adapter declared as `module:function`; the
adapter receives the validated spec and must consume `model_config` directly.
It remains responsible for the model-specific forward pass and durable,
atomic checkpoints. `pcqm_gptrans_v4:run_v4_spec` is an adapter for the frozen
GPTrans reference configuration and deliberately rejects changed architecture
settings.

The first spec for a platform may omit `runtime_certificate_id` and is used for
preflight. The preflight writes a certificate. The training spec then includes
that certificate ID and is validated before the adapter is called. Runtime
certificates are reusable while platform, accelerator, installed packages,
precision, deterministic settings, and calibration semantics remain unchanged.
The calibration fingerprint binds the source bundle, model architecture and
configuration, data roles, optimizer, schedule, target transform, and batch, so
the certificate cannot be reused after one of those inputs changes.

`molgap.v4_cli bind-certificate` converts the preflight spec into the training
spec and can change only run paths such as `output` and `preflight_path`; it
does not change the scientific or calibration fingerprints.

## Known Runtime Gates

The common layer handles old and new PyTorch `load` and sample-standard-
deviation signatures, checks optimizer keyword support before graph loading,
preserves the existing state-hash format, certifies bounded floating-point
repeatability, and normalizes source line endings before hashing. Static audit
rejects direct calls that bypass these compatibility helpers.

K1 full and K1 scale specify fused AdamW. The compatibility helper fails early
if the platform Torch lacks fused support; it never silently substitutes an
unfused optimizer because that would change the declared optimizer contract.

## Submission-Time Target

Once the source bundle and runtime certificate are installed, local audit,
manifest preparation, and staging should take less than five minutes. This
excludes scheduler queue time and the first per-platform runtime calibration.
The remote transfer/queue interval must be measured on the destination before
describing the end-to-end target as verified.

