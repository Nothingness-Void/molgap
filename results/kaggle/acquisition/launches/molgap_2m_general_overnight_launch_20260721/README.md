# General Overnight Acquisition Launch (2026-07-21)

Kaggle kernel:
`nothingnessvoid/active-molgap-general-overnight-fetch-r01`.

The CPU-only task accepts any molecule passing the global PubChemQC constraints:
allowed elements, molecular weight 200-1000, positive B3LYP Gap, and CID plus
canonical-SMILES exclusion. It does not target a chemistry bucket.

- Total target: 500,000 rows.
- Durable unit: five sequential 100,000-row chunks.
- Per chunk: CSV, atomic progress JSON, report JSON, log, SHA-256, and ZIP.
- Mounted exclusions: complete 1.5M table, prior repair union, and validated
  broad round 01.
- Final status: complete; 500,000 raw rows in 46.0 minutes.

Rounds 02 and 03 were still running when this task launched and therefore could
not be mounted as exclusions. Do not add the 500K target to the accepted pool.
Validation removed 529 rows overlapping prior accepted sources and retained
499,471 strict rows with no within-run duplicates. Accepted checkpoint:
`nothingnessvoid/molgap-2m-general-overnight-r01-accepted`.
