# Phase 8 PCQM Gap-Head Pilot Decision

## Decision

The retention-B frozen-encoder PCQM Gap-head pilot is rejected. All three
SCNet jobs completed correctly, but the learned head regressed on both the
official-valid local subset and the fixed common/OOD/P8-hard evaluation.
No model or registry entry is promoted.

## Accepted Execution

- Clean graph/embedding job: `706147`, completed in `00:14:14`.
- Head training job: `706148`, completed in `00:12:11`.
- External evaluation job: `706149`, completed in `00:00:28`.
- Clean official-train hard pool: 95,909 rows.
- Official valid/test and the future sealed 20K were excluded from training.

## Metrics

Candidate minus routed-v4 MAE deltas:

| Scope | Average delta | Gap delta |
|---|---:|---:|
| Common, 1,977 rows | +0.01615 eV | +0.04592 eV |
| OOD, 999 rows | +0.02153 eV | +0.06356 eV |
| P8-hard, 978 rows | +0.01065 eV | +0.02791 eV |
| PCQM valid, 4,981 rows | n/a | +0.08837 eV |

The PCQM valid Gap MAE changed from `0.29169` to `0.38006 eV`. The failure is
large and statistically separated from zero. The pilot is closed; do not tune
the same frozen-head formulation.

## Evidence

- `clean_pool_report.json`
- `train_metrics.json`
- `completion_manifest.json`
- `common_metrics.json`
- `pcqm_metrics.json`
