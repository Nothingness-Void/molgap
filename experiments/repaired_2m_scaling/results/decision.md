# Phase 8 Repaired-2M Data Decision

## Decision

The repaired-2M corpus is accepted as the only authorized dataset for the
retention-D GPS7 control. It is a fixed-size replacement, not another scale-up.
No model training was submitted by this data step.

The original exact-2M sources remain immutable. The new materialized table is
`data/raw/phase8_repaired_2m.csv`; the row-level selection contract is
`repaired_2m_manifest.parquet`.

## Construction

- Frozen targeted prefix: 500,000 rows, unchanged.
- Existing exact-2M rows retained: 1,728,539.
- Accepted candidate rows added: 271,461.
- Existing exact-2M rows replaced: 271,461.
- Final rows: 2,000,000.
- Unique CID / canonical SMILES: 2,000,000 / 2,000,000.
- Candidate pool after quality and identity reconciliation: 2,431,316 rows.
- Selection seed: 20260723.

The mutable 1.5M rows were selected against the targeted-500K joint
`Gap x MW x aromatic-rings x rotatable-bonds` distribution. The candidate
inventory could satisfy 1,335,331 rows in the exact requested joint bucket.
The remaining 164,669 rows used the nearest available joint bucket under a
fixed weighted Manhattan distance. Within each bucket, existing exact-2M rows
were retained first and lower-frequency scaffolds were preferred.

The earlier independent marginal-quota draft was rejected before training
because it over-shifted common joint buckets and high-Gap chemistry.

## Distribution Audit

| Statistic | Targeted 500K | Existing exact-2M | Repaired-2M |
|---|---:|---:|---:|
| Gap 2-4 eV | 25.1600% | 19.6749% | 25.1837% |
| Gap >= 6 eV | 6.2606% | 5.8238% | 6.2606% |
| MW > 700 | 6.0990% | 4.5101% | 6.6182% |
| Aromatic rings >= 4 | 23.9808% | 16.5604% | 21.2971% |
| Rotatable bonds >= 10 | 17.2930% | 10.6987% | 15.1262% |
| Heavy atoms 35-50 | 25.7972% | 15.2916% | 19.4687% |
| Unique scaffolds | 193,330 | 673,702 | 792,197 |

The repair restores the known retention dimensions without forcing every
marginal to equal the targeted prefix. Remaining aromatic/flexible/heavy-atom
shortfalls are explicit candidate-availability limits, not silent quota
failures.

## Quality And Leakage

- The row ledger covers 3,437,037 source rows including historical overlaps.
- All mutable rows must pass valid-SMILES, finite-label, Gap identity,
  non-extreme-label, closed-shell, connected, non-noble-gas, and scaffold
  checks.
- The frozen targeted 500K contains 14,572 flagged rows, principally known
  disconnected structures. They remain only because the retention contract
  requires the prefix to stay row/identity stable; the flags remain in
  the ledger.
- No future sealed source path entered the manifest.
- A deterministic 2,000-row source-row audit passed.
- Materialized Gap identity maximum error: `3.552713678800501e-15 eV`.
- Materialized CSV SHA256:
  `0c7d19de211016bebc2aa8b3030665e8c5f239baebbdf21c01598e0ddf3777c3`.

## Evidence

- `source_inventory.json`: immutable source paths, sizes, and checksums.
- `selection_report.json`: selection contract and manifest checksum.
- `manifest_audit.json`: retained/replaced counts.
- `validation_report.json`: identity and source-reference acceptance.
- `materialization_report.json`: final CSV acceptance and checksum.
- `distribution_before_after.csv`: targeted/current/repaired marginals.
- `joint_bucket_delta.csv`: complete joint-bucket changes.
- `quality_by_source.csv`: explicit source-level quality flags.
- `selected_source_mix.csv`: selected provenance and category counts.

## Next Gate

Retention-D must use the exact retention-B initialization, 50% targeted replay,
split, optimizer, epochs, and seed. D is compared directly with B. GPS9,
Router, distillation, 3D, and fusion remain blocked until GPS7 D passes:

- common average regression no worse than `+0.0005 eV`;
- at least `0.001 eV` improvement on OOD or P8-hard;
- no worse than `+0.0005 eV` on the other domain;
- paired worst-bucket and confidence-interval audit.
