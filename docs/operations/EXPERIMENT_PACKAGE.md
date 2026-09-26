# Experiment source package v1

`molgap.experiment_package` binds an explicit source allowlist to the strict
[ExperimentSpec](EXPERIMENT_SPEC.md) snapshot. It does not execute models,
read declared data roles, dispatch adapters, or submit jobs.

## API

- `build_experiment_source_package(spec, repo_root, relative_paths, output_dir)`
  requires exactly `ExperimentSpec`, reconstructs its canonical snapshot, and
  returns the published manifest dictionary.
- `verify_experiment_source_package(package_dir, repo_root=None)` returns the
  manifest only after all integrity checks. `inspect_experiment_source_package`
  has identical fail-closed semantics.

Pass a nonempty collection of unique POSIX relative file-name strings. The
manifest preserves caller order; the V4 inventory/archive sorts the same set.
No paths are discovered or appended. Changing allowlist order changes package
identity, though it does not change the sorted archive.

The output directory must be new or empty and cannot overlap selected source
files. An exclusive build marker reserves the directory. Construction happens
in a temporary child directory, with atomic file replacement and publication
of the manifest last. Failure cannot publish a complete manifest; interrupted
partial packages must be inspected/removed by the caller before retrying.
This API never updates an existing package. Content hashes detect later edits;
filesystem permissions are not an immutability or authentication mechanism.

## Six-file format

| File | Meaning |
|---|---|
| `source.tar.gz` | Deterministic LF-normalized regular-file archive |
| `SOURCE_COMMIT.txt` | Git commit plus LF |
| `SOURCE_ARCHIVE_SHA256.txt` | SHA-256 of compressed archive bytes plus LF |
| `SOURCE_FILES.json` | Unchanged `molgap-v4-source-inventory-v1` sidecar: source commit and sorted entries containing `path`, normalized-payload `sha256`, and `bytes` |
| `experiment_spec.json` | Exact `ExperimentSpec.to_json()` UTF-8 bytes, without trailing LF |
| `package_manifest.json` | Canonical UTF-8 JSON, without trailing LF |

The archive and first three sidecars are produced by the existing
`v4_bundle.build_v4_source_bundle`. Its tar implementation and semantics are
unchanged. The package reuses its `_payload`/`TEXT_SUFFIXES` normalization for
optional repository checks. Receipts, spec sidecars and manifests never enter
the source archive.

## Identity algorithm

Manifest format is `molgap-experiment-source-package-v1`; package status is
exactly `source_packaged`. Its stable fields are:

- `format`, `package_status`, `experiment_id`, `logical_run_id`;
- `spec_identity` (existing ExperimentSpec identity) and `spec_sha256` (exact
  canonical sidecar bytes);
- ordered `arms`, each containing `arm_id` and `identity`, where identity is
  `canonical_fingerprint(arm)` over the full declared arm;
- `source_commit`, `archive_sha256`, `inventory_sha256` (exact inventory bytes);
- `relative_allowlist`, preserving the caller's explicit order.

`package_identity = canonical_fingerprint(stable_fields)`. The fingerprint
uses sorted object keys, compact separators, ASCII JSON, and SHA-256. Only the
`package_identity` field itself is excluded. All other fields are required and
unknown fields are rejected. There are no timestamps, absolute paths or host
metadata. The manifest is serialized with those same canonical JSON settings.
Its identity therefore binds all its fields without self-reference. Source
commit, compressed archive hash, and spec/contract identity are distinct facts.

Verification reconstructs the entire expected manifest from the validated spec
and file bytes. It rejects changed hashes, declarations, arm lists, unknown
families/addons, noncanonical spec/manifest serialization, and duplicate JSON
keys. Hashes establish internal integrity, not signer authenticity: a party
that replaces all contents and recomputes all identities creates a different
self-consistent package. A consumer needing continuity must retain and compare
the original package identity externally.

## Rejection and verification boundaries

Absolute paths (including Windows drives), traversal, empty/dot components,
backslash aliases, control characters, trailing dots/spaces, case collisions,
symlinks/junctions in any path component, directories and nonregular files are
rejected. V4 enforces tracked, committed source and build-time HEAD binding.

The explicit conservative path policy rejects credential/secret/key tokens,
data/dataset/checkpoint tokens, record/output/result/artifact/receipt tokens,
`.git`, `.ssh`, `.aws`, `.azure`, `.env` names, private SSH key names, known
key/data/checkpoint/archive extensions, and the six reserved package filenames.
Token boundaries are dots, underscores and hyphens as well as path components;
for example `credentials.json` and `platforms/_records/receipt.json` fail. This
can reject source names such as `data_utils.py`; callers must choose a source
allowlist consistent with this policy. It is not content-based secret scanning
and never authorizes adding local data by inference.

Tar verification never extracts to disk. It rejects absolute/traversing names,
links (symbolic or hard), devices, directories, duplicate or extra members,
missing members and payload size/hash mismatches. Member sets must equal the
inventory and caller allowlist. The package directory must contain exactly the
six regular files.

With `repo_root`, verification checks that every source path remains tracked,
that its normalized worktree bytes match, and that the recorded source commit
contains a regular Git blob with matching normalized bytes. The host HEAD may
be a later or different commit. Without `repo_root`, verification is portable
and checks package contents only. Build also performs this repository binding
before publishing the receipt.

This manifest is derived packaging evidence only. `source_packaged` never
means READY, replay-ready, training success, measured cost, runtime acceptance,
or RML closure. Spec schema and terminal protocol versions are unchanged.

## Local Verification

Synthetic tests are in `tests/test_experiment_package.py`. They create temporary
caller-owned Git repositories, reuse the existing synthetic spec fixture, and
cover deterministic rebuilding, sidecar/declaration/source tampering, tracked
source boundaries, unsafe tar members, identity consistency and lack of runtime
authority. Symlink creation tests skip on hosts without that capability.

Executed server scope is in the
[integration review](shared_experiment_verification.md). From the selected
checkout and its project virtualenv:

```powershell
.venv\Scripts\python.exe -m pytest --noconftest tests/test_experiment_package.py tests/test_experiment_spec.py
```
