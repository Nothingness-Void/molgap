# Accepted complementary acquisition

This directory is the strict, publishable checkpoint for complementary rare
PubChemQC acquisition rounds 06 and 07.

- Validated input: 114,500 rows
- Removed against the complete prior inventory: 47 rows
- Removed across accepted sources: 40 duplicate rows
- Accepted: 114,413 rows
- Labels: finite, with `gap = lumo - homo` to numerical precision

Round 07 terminated after an HTTP `IncompleteRead`. Its `high_gap` CSV contains
9,524 rows, but the atomic progress checkpoint proves durability only through
row 9,500. Acceptance therefore keeps the first 9,500 and discards the
uncheckpointed tail. The other three round-07 groups completed normally.

`accepted_manifest.json` records source hashes, row counts, removals, recovery
provenance, and accepted-file hashes. These rows are a future additive pool;
they do not alter the frozen exact-2M training table.
