# PCQM4Mv2 Official-Train Residual Scan Decision

## Decision

Accept the scan and raw hard-pool artifacts, but do not train directly on the
raw 200K pool. Build a cleaned, fixed Gap-only candidate pool first.

## Accepted Scan

- SCNet job `703665`: completed in 40m24s with exit code 0.
- Official train rows: 3,378,606; official valid/test excluded.
- Valid predictions: 3,378,573; invalid molecular graphs: 33.
- Durable output: 136 Parquet parts plus reports and a 200,000-row hard pool.
- Canonical-SMILES overlap with repair-v3 1.5M: zero.
- Independent acceptance job `704402` passed part counts, row accounting,
  source-index range/uniqueness, finite values, every part SHA256, hard-pool
  SHA256, and canonical uniqueness.

## Domain Audit

The raw pool is not a clean organic-electronics training set:

- 103,440 rows contain radical electrons.
- 344 are disconnected structures.
- 1,148 have fewer than five heavy atoms.
- Three contain noble-gas atoms.
- The largest errors are dominated by examples such as `[He].[He]`,
  `[Ar].[Ar].[Ar]`, methane, radicals, and strained hydrocarbons.

The conservative connected, closed-shell, at-least-five-heavy-atom filter
retains 95,909 rows. Its median/p90 teacher Gap error is `0.972/1.555 eV`.

## Training Constraint

PCQM4Mv2 supplies Gap labels, not the MolGap three-target HOMO/LUMO/Gap tuple.
Use the cleaned pool only through a masked Gap-only auxiliary objective, or
recover same-definition PubChemQC HOMO/LUMO labels before ordinary joint
training. Keep the official validation and test partitions excluded.
