# Failed attempt acceptance - 2026-10-13 JST

## Identity and retention

Kaggle1 `nothingnessvoid/molgap-k1-mean2-single-ema-500k-20261013`,
kernel138365645/version1: authoritative scheduler ERROR, failureMessage null.
[Source verification](source_verification.json) checks exact remote entry bytes,
dataset membership and the retained immutable package. Runtime reached the
all-arm preflight barrier after its launch/archive/package verification.
Source4d4de9a7, package9731ab24, Spec7c88fd5d; complete digests are in the receipt.

Retrieved only four diagnostic files, listed and hashed in
[retrieval manifest](retrieval_manifest.json). No output manifest or trained
artifact exists in the complete four-file API inventory. The output API itself
does not pin a version; version1 metadata and exact source reconciliation bind
this observation. The anomalous API lastRunTime is retained without using it
as an execution timestamp.

## Independent arm observations

| Arm | Phase | Formal training | Epoch / step / presentations | Qualification |
|---|---|---|---|---|
| mean2 | preflight failed, exit1 | not started | 0 / 0 / 0 | not qualified |
| single_ema | preflight failed, exit1 | not started | 0 / 0 / 0 | not qualified |

Both fail at `family_training_data.load_training_data` resolving the frozen
manifest path, before opening graph shards or executing a model. Train and
development graphs/labels were not consumed by these workers; official
validation/test remain sealed. No MAE, speed comparison, selected/resume state,
training trace or scientific endpoint can be accepted.

## Native cost and outcome

The entry ledger observes Tesla T4x2 allocation and219.285577661s wall,
438.571155322 device seconds. This includes setup and idle assigned GPUs,
not GPU busy time or complete platform billing. Per-arm preflight windows
are11.504980323s and11.504354333s; they are nested in entry time and must not
be added again to its total. Queue, CPU device time and platform tails stay
unknown.

Mechanical failure inspection is complete. Both attempts are NO_TRAIN and
the research comparison is INCONCLUSIVE. RML closure must preserve distinct
arm identities and exclusions for absent formal training/reference/runtime/
artifact evidence; it cannot publish capability complete or model promotion.
