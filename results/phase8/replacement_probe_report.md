# P8 Replacement Dataset Report

- old rows: 300,000
- targeted candidate rows used: 3,847
- output rows: 300,000
- output csv: `data\raw\phase8_replacement_300k_probe.csv`

## Targeted Buckets Used

| bucket | n |
|---|---:|
| `s_or_cl_hard` | 1,285 |
| `large_mw_500_700` | 1,050 |
| `very_low_gap` | 402 |
| `very_large_general` | 333 |
| `low_gap_aromatic_edge` | 273 |
| `large_aromatic_edge` | 201 |
| `flexible_hard` | 154 |
| `aromatic_edge_general` | 149 |

## Coverage Shift

| flag | old n | old frac | replacement n | replacement frac | delta n |
|---|---:|---:|---:|---:|---:|
| p8_low_gap | 7,314 | 2.44% | 8,190 | 2.73% | +876 |
| p8_large | 19,637 | 6.55% | 22,043 | 7.35% | +2,406 |
| p8_aromatic_edge | 6,639 | 2.21% | 7,383 | 2.46% | +744 |
| p8_scl_hard | 25,443 | 8.48% | 27,164 | 9.05% | +1,721 |
| p8_flexible_hard | 9,410 | 3.14% | 10,265 | 3.42% | +855 |
| p8_any_hard | 46,788 | 15.60% | 50,635 | 16.88% | +3,847 |