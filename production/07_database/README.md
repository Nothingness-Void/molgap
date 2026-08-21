# Database Delivery

This directory owns the versioned property-database build. It is the next
delivery stage after model selection; it does not own model architecture
experiments or historical Delta/UQ calibration.

## Current State

The database build has not started. The first target is a 1K dry run followed by
a 10K accepted pilot. A full commercial-molecule universe is not authorized
until both runs pass their acceptance checks.

## Authoritative Model

- Model: `repaired_2m_dense_2d`
- Task: B3LYP/6-31G* `homo`, `lumo`, `gap` in eV
- Public implementation: `src/molgap/inference.py`
- Lower-cost fallback: `repaired_2m_equal_2d`
- Track B PCQM Gap models are specialists and do not enter this database build.

## Row Contract

Every input molecule must produce one auditable row or one explicit rejection
record. The build must retain at least:

- input SMILES and canonical SMILES when parsing succeeds;
- stable source identifier and deduplication key;
- `homo`, `lumo`, and `gap` predictions in eV, or a null prediction with a
  reason code;
- model key/version and inference configuration;
- validity, allowed-element, molecular-weight, graph-success, and `in_domain`
  flags;
- expert-disagreement screening signal, when available, explicitly labeled as
  uncalibrated;
- build timestamp, row counts, source manifest, and SHA256 checksums.

Invalid, unsupported-element, out-of-range, and graph-failed rows must not be
silently removed. The release may provide a filtered view, but the raw build
ledger remains the audit source.

## Acceptance Order

1. Run the 1K dry run with atomic output and a machine-readable manifest.
2. Check finite predictions, row counts, deduplication, source alignment, and
   rejection-reason totals.
3. Run the 10K pilot with the same code and configuration.
4. Freeze the accepted manifest and only then scale to the target molecule
   universe.

Reusable inference and validation logic belongs in `src/molgap/`; this
directory should contain thin production entry points, manifests, and build
records. Delta and UQ fields must remain absent or explicitly non-authoritative
until they are refit and revalidated against the repaired-2M base.

## Pointers

- Execution order: `ROADMAP.md`
- Live model state: `CURRENT_STATE.md`
- Naming rules: `NAMING.md`
- Public inference: `src/molgap/inference.py`
