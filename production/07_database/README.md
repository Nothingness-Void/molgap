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

## Batch Entry Point

The reusable implementation is `src/molgap/database.py`; the production
adapter is `scripts/build_database.py`. The input CSV must contain a `smiles`
column. If no `--id-column` is supplied, the builder uses `source_id`, `cid`,
`id`, or `name` when present, then creates a deterministic
`input_row_{row:08d}` identifier. All input columns are retained in the raw
ledger.

Example 1K dry run after the accepted model bundle is present:

```powershell
.venv\Scripts\python.exe production\07_database\scripts\build_database.py `
  --input data\commercial\commercial_molecules_template.csv `
  --out-dir production\07_database\runs\dry_run_1k `
  --max-rows 1000 `
  --model-key repaired_2m_dense_2d
```

The output directory contains `predictions.csv` and `manifest.json`. The CSV
is an audit ledger: invalid SMILES and graph failures remain as rejected rows;
unsupported elements and molecular weights outside 200-1000 Da remain as
finite predictions with `in_domain=false` and an applicability reason. The
expert disagreement columns are population standard deviations across the
direct GPS experts and are explicitly uncalibrated screening signals, not UQ.

## Pointers

- Execution order: `ROADMAP.md`
- Live model state: `CURRENT_STATE.md`
- Naming rules: `NAMING.md`
- Public inference: `src/molgap/inference.py`
