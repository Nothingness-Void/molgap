# Kaggle Account Organization (2026-07-21)

## Result

The private Kaggle account is organized by lifecycle instead of experiment
number. Dataset slugs were preserved so mounted-input references remain valid.
Kernel titles and slugs were renamed through Kaggle's title editor without
creating new versions or rerunning any workload.

- `ACTIVE`: one reusable acquisition runner.
- `EVAL`: two fixed acceptance-evaluation runners.
- `ARCHIVE`: completed, negative, failed, or historical evidence.
- One untouched default notebook with zero versions was deleted.

## Active and evaluation kernels

| Role | Kaggle ref |
|---|---|
| Active broad acquisition | `nothingnessvoid/active-molgap-broad-candidate-fetch-r01` |
| Common/OOD/P8-hard evaluation | `nothingnessvoid/eval-molgap-1m-common-ood-hard` |
| PCQM4Mv2 proxy evaluation | `nothingnessvoid/eval-molgap-1m-pcqm4mv2` |

## Renamed archive kernels

| Previous ref | Current ref |
|---|---|
| `molgap-residual-target-fetch` | `archive-molgap-residual-target-fetch-recovered` |
| `molgap-repair-v2-pure-2d-control-eval` | `archive-molgap-repair-v2-2d-eval-negative` |
| `molgap-1m-repair-v2-candidate-fetch` | `archive-molgap-repair-v2-fetch` |
| `molgap-1m-pcqm-replay-fusion` | `archive-molgap-1m-pcqm-replay-fusion-negative` |
| `molgap-1m-replay-fusion` | `archive-molgap-1m-replay-fusion-negative` |
| `molgap-fusion-1m` | `archive-molgap-1m-fusion-probe` |
| `gps-2d-pretrain-300k` | `archive-gps-2d-pretrain-300k-error` |
| `schnet-optuna-on-300k-molecules` | `archive-schnet-optuna-300k` |
| `molgap` | `archive-molgap-legacy` |
| `schnet-300k-kaggle-gpu-multi-gpu-aware` | `archive-schnet-300k-kaggle` |

All refs above use owner `nothingnessvoid/`. Historical result documents may
retain the old refs as evidence of what was submitted at the time; Kaggle keeps
redirects after title renames.

## Dataset metadata

All 12 private datasets now use one of these title prefixes:

- `MolGap | Active | ...`
- `MolGap | Eval | ...`
- `MolGap | Reference | ...`
- `MolGap | Archive | ...`

Descriptions explicitly identify negative candidates, immutable evaluation
labels, acquisition-exclusion data, and active runtime assets. The applied
metadata is recorded in `applied.json`; no dataset version or file was changed.

## Naming rule

Future Kaggle titles must begin with `ACTIVE |`, `EVAL |`, or `ARCHIVE |` for
kernels, and `MolGap | <Role> |` for datasets. Do not create generic notebook
names. Reuse an existing active slug for another round only when the code and
data contract are unchanged; otherwise create a new `rNN` slug.
