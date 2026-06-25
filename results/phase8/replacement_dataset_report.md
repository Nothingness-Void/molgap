# P8 Replacement Dataset Report

- old rows: 300,000
- targeted candidate rows used: 38,620
- output rows: 300,000
- output csv: `data\raw\phase8_replacement_300k.csv`

## Targeted Buckets Used

| bucket | n |
|---|---:|
| `large_mw_500_700` | 13,847 |
| `s_or_cl_hard` | 7,677 |
| `very_large_general` | 4,593 |
| `aromatic_edge_general` | 4,185 |
| `flexible_hard` | 3,842 |
| `large_aromatic_edge` | 3,801 |
| `very_low_gap` | 402 |
| `low_gap_aromatic_edge` | 273 |

## Coverage Shift

| flag | old n | old frac | replacement n | replacement frac | delta n |
|---|---:|---:|---:|---:|---:|
| p8_low_gap | 7,314 | 2.44% | 10,824 | 3.61% | +3,510 |
| p8_large | 19,637 | 6.55% | 45,129 | 15.04% | +25,492 |
| p8_aromatic_edge | 6,639 | 2.21% | 15,623 | 5.21% | +8,984 |
| p8_scl_hard | 25,443 | 8.48% | 39,024 | 13.01% | +13,581 |
| p8_flexible_hard | 9,410 | 3.14% | 19,366 | 6.46% | +9,956 |
| p8_any_hard | 46,788 | 15.60% | 85,408 | 28.47% | +38,620 |