# Complementary rare acquisition rounds 06-07

## Outcome

| Item | Result |
|---|---:|
| Round 06 | `COMPLETE`, 60,000 rows |
| Round 07 | `ERROR`, 54,500 durable rows recovered |
| Strictly validated input | 114,500 |
| Prior-inventory overlap removed | 47 |
| Cross-source duplicates removed | 40 |
| Accepted | 114,413 |

Round 07 failed only in the `high_gap` worker after a transient
`http.client.IncompleteRead`. The other workers completed. The failed worker
wrote 9,524 rows, while its atomic checkpoint confirms 9,500; validation keeps
only that proven prefix. The fetcher now retries `IncompleteRead` in future
runs, so rerunning this round solely to fill the 5,500-row rare quota is not
worth the acquisition cost.

## Accepted coverage

| Bucket | Rows |
|---|---:|
| High-gap hetero | 18,000 |
| High-gap rigid | 6,499 |
| Hetero-dense mid-gap | 12,000 |
| Small hetero-dense | 10,000 |
| Sulfur-rich | 7,999 |
| Halogen-rich | 5,998 |
| Bridged polycyclic | 11,940 |
| Fused rigid | 11,979 |
| Donor-acceptor conjugated | 15,882 |
| Conjugated mid-gap | 14,116 |

All required labels are finite and `gap = lumo - homo` holds within floating
point precision. The full accepted future-acquisition inventory increases from
824,411 to 938,824 rows.

## Evidence

- Raw immutable downloads: `raw/round06/` and `raw/round07/`
- Machine acceptance report: `acceptance_report.json`
- Publishable files and hashes: `accepted/accepted_manifest.json`
- Private checkpoint: `nothingnessvoid/molgap-2m-complementary-round06-07-accepted`

This is candidate inventory only. It does not modify the exact-2M experiment,
the current model registry, or the sealed evaluation protocol.
