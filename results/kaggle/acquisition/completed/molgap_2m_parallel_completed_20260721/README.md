# Broad Rounds 02-03 Acceptance

Round 02 completed with 60,000 valid rows. Round 03 completed with all 60,000
quota rows and valid hashes, but contained three duplicates across its parallel
aromatic and balanced workers. The immutable raw download is retained in
`round03/`; `round03_accepted/` removes those rows from balanced and regenerates
the manifest and SHA-256.

Final accepted counts:

| Source | Strict rows |
|---|---:|
| Round 01 | 60,000 |
| Round 02 | 60,000 |
| Round 03 | 59,997 |
| Total | 179,997 |

The general overnight 500K target was still running during this acceptance and
is not included in the total.
