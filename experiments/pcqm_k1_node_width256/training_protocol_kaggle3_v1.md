# K1 width256 Kaggle3 single-candidate training

Date: 2026-10-01. Owner: desktop. User explicitly instructed submission to Kaggle3.
This contract releases one width256 candidate and supersedes the prior missing-
reference waiting condition once the recovered reference inputs pass validation.

The scientific intervention remains atom width192 ->256, nine layers, edge64,
slot64, original gated aggregation, pure2D OGB atom9/bond3/RWSE16, direct Gap eV.
One seed42,100K training rows,50K internal-development rows,40epochs,FP32,
batch128/drop-last,AdamW lr0.0004/weight-decay0.00001/clip1,cosine eta_min1e-6,
31240 updates and3998720 sample presentations. No auxiliary objective,
pretraining, ensemble, reference retraining, additional seed or scale-up.
The executable family-built recipe owns all numeric settings and identities.

Frozen CPU initial state: `qualification_initial_state.pt`, tensor digest
`41bbbadeb1056c2cdaa2e84555dfcae2f747dc0c8dcb7394577d7634503fabc2`.
Actual parameter count6035201 versus3658817; below the user's twofold ceiling.

## Reference and acceptance

Reference: the original K1-192 arm of authenticated COMPLETE Kaggle3 kernel
`nvoid912/molgap-k1-v4-ssma-accuracy-100k-s42-v1`, observed version1, source
`f21920ba5d9f3fe148cfa18803d53c0a5ec8f670`. Its40epoch exposure and50K
prediction rows passed hash-bound mechanical inspection and owner runtime
qualification. The retained selected model is epoch40, Gap MAE
0.1412944608205557eV. See `reference_binding/` and the pinned raw output records.
This reuse does not close or judge the separate SSMA question.

The reference recipe's target hash is original raw float32 bytes. Static
`k1-screen-v1` artifact adaptation preserves this encoding; other adapters retain
their float64 encoding. The initial rejection and correction are retained.
The candidate uses the same original internal-development identity and target
hash. No labels, predictions or frozen reference recipe were modified.

Freeze all comparison identities and actual source before release; only
architecture identity is a declared scientific intervention. Candidate runtime
qualification is required on its actual assigned T4, including finite/nonzero
gradients, repeatability, complete optimizer/scheduler/RNG resume and selected-
state roundtrip. Width initialization equivalence is inapplicable because tensor
shapes differ. Measure/report optimizer-inclusive overhead; no SSMA25% veto.
CPU real-shard checks precede publication; they do not certify T4 behavior.

Complete mechanical/scientific/role/cost and paired uncertainty checks after
retrieval. The inherited K1 minimum gain is0.003eV. Row bootstrap does not
estimate training stochasticity; one seed has no measured seed variance.
No automatic promotion, successor or500K/full admission follows this run.

## Roles and cost

Only official-training-prefix train[0:100000] and development[100000:150000].
Official validation, test-dev and challenge remain untouched by this run.
Dataset: accepted private `nvoid912/pcqm4mv2-ogb-fixed-100k-v1`; manifest
`1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d`.
Strip retained geometry fields; do not construct3D or feed geometry to the model.

Kaggle's T4 allocation is two physical devices; one candidate uses assigned
device0. Planning estimate:4wall hours,8allocated T4 hours,4useful assigned T4
hours; these are estimates, not measurements or a new early-stop policy.
CPU qualification estimate0.1CPU/wall hours. Actual CPU, queue, bootstrap and
allocation scopes must be retained separately; unavailable values remain unknown.
Use atomic selected/resume checkpoints, canonical trace and durable Kaggle
outputs. Preserve exact package/config/source/dataset/job/version provenance.
An interrupted run requires reconciliation, not automatic retraining or retry.
