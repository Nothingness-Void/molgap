# Experiment Launch Receipt V1

`molgap.experiment_launch` is a pure local receipt/reconcile core, **not a
submitter**. It does not contact Kaggle, SCNet, IMS, or any other service. No
verified platform response adapter is supplied by this module. Its submitter
status is always `SUBMIT_UNIMPLEMENTED`. It does not import model runtimes,
construct models, consume data roles, run preflight, or invoke subprocesses.

## Public API

```python
from pathlib import Path
from molgap.experiment_spec import ExperimentSpec
from molgap.experiment_launch import (
    build_launch_receipt, reconcile_platform_response,
    read_launch_receipt, write_launch_receipt, canonical_json,
)

spec = ExperimentSpec.from_json(Path("experiment_spec.json").read_text("utf-8"))
package_dir = Path("frozen_package").resolve()
receipt_dir = Path("launch_receipts").resolve()
receipt_dir.mkdir(exist_ok=True)
# Retrieve this pin from the authorized package record, not an untrusted input.
pin = "<authorized-package-identity-sha256>"

# Local dry run: verifies source bytes and records NOT_SUBMITTED only.
receipt = build_launch_receipt(spec, package_dir, expected_package_identity=pin)
path = write_launch_receipt(canonical_json(receipt), receipt_dir, spec,
                           package_dir, expected_package_identity=pin)
restored = read_launch_receipt(path, spec, package_dir,
                             expected_package_identity=pin)

# Optional explicit observation, already translated by a reviewed external adapter:
# reconciled = reconcile_platform_response(
#     spec, package_dir, response_json, expected_package_identity=pin)
```

The dry-run API above is the local entry point; there is no submit CLI. The pin
placeholder must be replaced with a trusted SHA256. Package and receipt paths
must be absolute local concrete `Path` objects. Output directories must already
exist, be dedicated to receipts, and be outside the six-file source package and
research-memory indexes. Network paths, traversal, symlinks, junctions, reparse
points, and Windows alternate streams are rejected. Monitor paths are inert
relative strings, never commands or automatically accessed resources.

`build_launch_receipt` and `reconcile_platform_response` return detached plain
dictionaries. `canonical_json` serializes them; it is not itself a schema
validator. The reader and writer revalidate against the exact spec and pinned
package, including all derived fields and receipt identity.

## Wire Contract

All wire input is canonical JSON text: sorted keys, compact separators, ASCII
escaping, finite numbers, no duplicate keys, BOM, newline, or extra fields.
Executable objects, callbacks, import paths, credential/token fields, arbitrary
raw platform payloads, and unknown schema versions are not accepted. Never put
secrets in opaque references: validation is not a credential-redaction service.

Receipt `format` is `molgap-experiment-launch-receipt`, integer `version` is `1`.
Its exact fields are `format`, `version`, `binding`, `launch_identity`,
`receipt_identity`, `submission_state`, `reconciliation_result`,
`submitter_status`, `next_action`, `response`, `cost`, `limitations`,
`canonical_platform_reference`, `platform_version`, `physical_runs`, `timestamp`,
and `monitor_paths`.

`binding` has exactly:

- `spec_identity`, `spec_sha256`, `experiment_id`, and `logical_run_id`.
- `package_identity`, `source_commit`, and `source_archive_sha256`.
- Ordered `arms`, each containing `arm_id` and `arm_identity`.
- `requested_logical_platform`, the exact Spec platform name, not a remote run.

The builder requires exactly `ExperimentSpec`, revalidates its canonical
snapshot, calls `verify_experiment_source_package`, compares the trusted package
pin, and matches spec bytes/hash, experiment/run IDs, and ordered arm identities.
The package verifier validates source/inventory hashes. The spec has no source
commit field; the verified pinned package owns that source binding. Package
verification without `repo_root` is integrity verification, not independent Git
commit authentication. No source package bytes or hashes include the receipt.

Every optional observation uses `{"value": ..., "missing_reason": ...}`. Known
values require null reason; unknown values require null value and one of
`not_reported`, `not_observed`, `not_submitted`, `response_lost`, or
`adapter_unimplemented`. Timestamp, run mapping, reference, version, monitoring
paths and cost are never silently invented. V1 cost is always null with
`not_observed`; native-cost measurement remains a separate record, not a zero
or a hardware-normalized estimate. Timestamp is the supplied platform observation
time, not an invented submission time or an automatically refreshed local clock.
Known timestamps use UTC ISO8601 with `+00:00`.

## Response Envelope

The strict neutral envelope has exactly the fields below. Use `canonical_json`
to encode it before calling reconcile. This is a **synthetic dry-run** example,
not a Kaggle/SCNet/IMS response schema:

```python
missing = {"value": None, "missing_reason": "not_reported"}
envelope = {
    "format": "molgap-platform-response", "version": 1,
    "mode": "dry_run", "outcome": "accepted", "conflict_kind": None,
    "binding": receipt["binding"],
    "canonical_platform_reference": missing.copy(),
    "platform_version": missing.copy(), "physical_runs": missing.copy(),
    "timestamp": missing.copy(), "monitor_paths": missing.copy(),
}
dry = reconcile_platform_response(spec, package_dir, canonical_json(envelope),
                                 expected_package_identity=pin)
assert dry["submission_state"] == "NOT_SUBMITTED"
assert dry["reconciliation_result"] == "DRY_RUN_ONLY"
```

`mode` is `observed` or `dry_run`. Observed envelopes are caller assertions:
schema validation is not platform authentication, and even `ACCEPTED` does not
mean this library submitted anything. Future reviewed adapters must translate
authoritative platform evidence; mocks must use `dry_run`. The complete envelope
is retained under `response`; dry-run or identity-conflicting references are
never promoted into the receipt's effective platform fields.

Canonical references and versions are opaque restricted strings. No account,
kernel slug, scheduler job ID, or platform version is derived from a logical
name. `physical_runs.value` is null unless an explicit complete mapping is
observed. A known value is a nonempty list of objects with exactly
`run_identity`, `canonical_reference`, `platform_version` (a fact), and ordered
`arm_ids`. Every declared arm must occur exactly once. Duplicate physical
reference/run pairs, duplicate arms, unknown arms and partial mappings fail.
One physical run may map to both arms. Two arms/device counts never manufacture
two external runs. Distinct physical runs are retained only when supplied.

## State Semantics

| Input | Submission state | Result / action |
|---|---|---|
| No envelope | `NOT_SUBMITTED` | `SUBMIT_UNIMPLEMENTED`; implement a verified adapter |
| Matching dry-run envelope | `NOT_SUBMITTED` | `DRY_RUN_ONLY`; no effective platform identity |
| `accepted` | `ACCEPTED` | Records caller-observed acceptance, not scientific acceptance |
| `queued` or `pending` | `PENDING_RECONCILIATION` | Inspect authoritative state before any retry |
| `client_unknown_after_submit` | `PENDING_RECONCILIATION` | Lost response cannot imply acceptance or safe resubmission |
| `unknown` | `UNKNOWN` | Unknown is neither failure nor success |
| `rejected_permission` | `REJECTED` | Resolve permission without retry |
| `rejected_409` | `REJECTED` | Explicit classification, no automatic rename/retry |
| Matching `existing_found` | `RECONCILED_EXISTING` | Identity match, not a new submission |
| Any well-formed envelope with differing binding | `REJECTED` | `REJECTED_CONFLICT`; resolve identity conflict |

For `rejected_409`, `conflict_kind` must be exactly `name_conflict`,
`existing_version`, `permission`, or `other`. Result names are
`REJECTED_409_NAME_CONFLICT`, `REJECTED_409_EXISTING_VERSION`,
`REJECTED_409_PERMISSION`, and `REJECTED_409_OTHER`. Every other outcome requires
null `conflict_kind`. A 409 existing-version response is not successful
reconciliation. A separate `existing_found` observation must provide a known
canonical reference and a complete matching binding, including source commit,
archive, spec, package and arm order/identity. Malformed envelopes raise rather
than becoming status records. No retry loop, rename policy or submit fallback
exists.

## Persistence And Boundaries

`launch_identity` hashes experiment ID, logical run ID and requested platform.
It intentionally excludes the source/spec/package hashes so a changed source
under the same logical launch cannot silently overwrite the prior observation.
`receipt_identity` hashes all receipt fields except itself, including the input
observation. There are no implicit timestamps, so identical input is idempotent.

The filename is `<launch_identity>.json`. A same-directory temporary file is
flushed and fsynced, then atomically published with a no-replace hard link.
Concurrent identical writes are no-ops; conflicting bytes, including an empty
or noncanonical existing file, fail closed. Filesystems without hard-link support
fail rather than falling back to unsafe replacement. A crash can leave an
unpublished temporary file; it is not a receipt. This is not a transactional
database or a power-loss guarantee for directory metadata. An adversary changing
filesystem ancestors concurrently is outside this trusted-local-filesystem
boundary.

Changed observations, including later successful reconciliation, require a new
explicit snapshot directory. Old receipts are never mutated. This core does not
select a latest snapshot, merge evidence, authorize a successor, or manage
platform state. Supplying a new directory does not authorize a platform retry.

Preflight diagnostic statuses are not admission or acceptance and are not
inputs to this core. `TerminalDescriptor` remains the separate terminal
translation boundary: a launch receipt is not a terminal descriptor and cannot
close a trajectory. No RML, READY, replay-authority or role-use fields are
generated and no canonical research records are written. See
[source packaging](EXPERIMENT_PACKAGE.md), [preflight](EXPERIMENT_PREFLIGHT.md),
the [shared V5 contract](MOLGAP_COMMON_DIRECTION_V5_FINAL.md), and the
[server handoff](SERVER_AGENT_HANDOFF_V5_FINAL.md).

Future platform adapters must separately establish tested response schemas,
canonical account/project/version/run references, immutable source linkage,
credential isolation, submission authorization and remote durability. They must
obey each platform access boundary, preserve unknown fields as missing, classify
conflicts without guessing, and reconcile authoritative existing identities
before permitting any separately authorized retry. No such adapter is implemented
here, and no historical platform script is modified.

## Local Verification

Run this local-only suite from the selected checkout with its project virtualenv.
The server's executed scope is in the
[integration review](shared_experiment_verification.md):

```powershell
$env:PYTHONPATH = (Join-Path $PWD 'src')
& '.\.venv\Scripts\python.exe' -m pytest --noconftest tests/test_experiment_launch.py -q
```

Fixtures create a tiny synthetic local Git repository/package in pytest's
temporary directory; they neither access remote platforms nor consume molecular
data. The no-external-call test blocks network/process calls after fixture setup.
Synthetic test success does not establish actual submission, training results
or platform compatibility.
