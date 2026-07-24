# Hard-Chemistry Rounds 04-05 Acceptance

Both Kaggle tasks completed successfully with source-exhausted partial quotas.
The downloaded outputs under `round04/` and `round05/` are immutable raw
evidence. Reconciled outputs are under `accepted_round04/` and
`accepted_round05/`.

| Measure | Round 04 | Round 05 | Total |
|---|---:|---:|---:|
| Raw output | 73,054 | 72,942 | 145,996 |
| Within-round duplicates removed | 455 | 598 | 1,053 |
| Strict accepted output | 72,599 | 72,344 | 144,943 |

The accepted rows have no overlap with the 1.5M table, repair union, broad
rounds 01-03, general overnight round 01, or each other. All required values
are present and finite, and `gap = lumo - homo` holds to floating-point
precision. The complete future acquisition inventory is now 824,411 strict
rows.

Combined checkpoint: `nothingnessvoid/molgap-2m-hard-round04-05-accepted`.

## Accepted bucket composition

| Bucket | Rows |
|---|---:|
| Flexible, 2.5-4.0 eV Gap | 49,796 |
| High-sp3 non-aromatic | 39,949 |
| Flexible very large | 24,624 |
| Multi-amide very large | 20,000 |
| High-sp3 very large | 5,653 |
| Macrocycle very large | 4,355 |
| Non-aromatic very large | 566 |

The 100K target per shard was deliberately not padded with general molecules.
The shortfall identifies genuinely source-limited buckets, especially
non-aromatic very-large and macrocycle very-large molecules.
