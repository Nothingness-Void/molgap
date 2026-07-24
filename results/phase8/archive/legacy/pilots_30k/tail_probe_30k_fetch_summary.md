# Phase 8 Tail Probe Fetch Summary

Date: 2026-06-30

Purpose: build a small B3LYP tail-like pool after expansion500k residual
analysis showed the remaining Gap error is concentrated in low-gap and very
large/flexible molecules.

This is a **probe pool**, not a new training set yet.

## Fetch result

Output CSV: `data/raw/phase8_tail_probe_30k.csv`

Rows: 20,829 unique molecules. The requested 30k cap was not reached after
scanning the available HF windows because the strict rare buckets are genuinely
sparse after excluding the existing 500k training set and previous top-ups.

Bucket counts:

| bucket | rows |
|---|---:|
| very_low_gap | 3,121 |
| low_gap_aromatic_edge | 566 |
| low_gap_general | 7,200 |
| very_large_general | 4,384 |
| very_large_tail | 2,558 |
| flexible_hard | 3,000 |

Key coverage:

| criterion | rows |
|---|---:|
| Gap < 2.5 eV | 3,121 |
| Gap < 3.2 eV | 11,656 |
| MW >= 800 | 4,424 |

Quantiles:

| field | p01 | p05 | p10 | p50 | p90 | p99 |
|---|---:|---:|---:|---:|---:|---:|
| Gap | 1.456 | 2.109 | 2.367 | 3.148 | 4.912 | 6.128 |
| MW | 233.3 | 287.3 | 330.4 | 526.5 | 879.1 | 989.5 |

Report JSON: `results/phase8/archive/legacy/pilots_30k/tail_probe_30k_fetch_report.json`.

## Notes

- Exact `very_low_gap` + `low_gap_aromatic_edge` priority scanning produced no
  new rows after 48 HF files / 211k parsed objects, so the probe was relaxed.
- Added probe-only fetch buckets in `scripts/phase8/archive/legacy/data_coverage/fetch_targeted_topup.py`:
  `low_gap_general` and `very_large_tail`.
- `low_gap_general` filled its scaled quota (7,200 rows). `very_low_gap` and
  `very_large_tail` remained sparse after the full scan.
- Next step, if this branch is pursued: assemble a small top-up/replay dataset
  and run a short warm-start probe. Do not replace the common eval set.
