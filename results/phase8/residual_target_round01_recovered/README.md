# Residual-Target Round 01 Recovery

Kaggle kernel `nothingnessvoid/molgap-residual-target-fetch` ran for 16,711
seconds and published all four shard outputs, but ended with status `ERROR`.
The fetch subprocesses all returned zero. The outer runner incorrectly required
every rare-source quota to be exactly full after all 430 source files had been
scanned, then raised `RuntimeError` when the source was exhausted.

## Recovered data

- Raw shard rows: 45,520.
- Deduplicated rows: 45,457.
- Development top-up: 40,457.
- Scaffold-disjoint sealed set: 5,000.
- Development/sealed scaffold overlap: 0.
- Overlap with the complete 1.5M table: 0 CID, 0 canonical SMILES.
- Overlap with the prior repair candidate union: 0 CID, 0 canonical SMILES.

The recovered rows are valid and can support the first approximately 40K
additive pilot. The missing 14,480 quota rows were not lost rows; they were
unavailable under the declared rare predicates in the scanned source windows.

The cloud runner now reports `complete_partial` when all subprocesses finish
successfully but a source-exhausted bucket is under quota. It only raises when
a subprocess actually fails.
