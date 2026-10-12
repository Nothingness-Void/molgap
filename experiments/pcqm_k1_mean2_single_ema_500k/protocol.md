# Mean2 versus single-forward EMA at 500K

Owner: desktop. User-authorized on 2026-10-13: Kaggle1 API, one T4x2
Notebook, one arm per device, 500K, 60 epochs per arm, nine-hour total wall
ceiling. No automatic retry, continuation, monitoring, full training or adoption.

## Question

Compare the old mean2 **training recipe** with single-forward plus true EMA.
Both use the current shared accelerated trainer, not two copies of a historical
and live trainer. This compares complete recipes, not execution-only speedups.
Mean2 has two stochastic forward L1 losses per update and live development
selection. Candidate has one forward and EMA0.9999 after each optimizer update,
with EMA development selection. No teacher, consistency loss or Clean-BN step.
The unresolved local Clean-BN result is not a qualifying reference.

## Frozen common parameters

K1, 3,658,817 parameters, seed42 and transported initial tensor digest
`8ef6d4ba1abdca04d8740a11ec0e04587358117b1e9f9534b4fc19d2b6caedbd`.
FP32, no TF32, deterministic algorithms, physical BS128, no accumulation,
AdamW lr0.0004/wd0.00001/clip1, cosine by epoch, 60 epochs, eta_min0.000001.
Torch2.11.0. Identical train order: torch randperm seed42+pass; discard32
tail rows per pass. Per arm: 3,906 steps/pass, 234,360 total steps,
29,998,080 sample presentations. Never compress the schedule to fit the budget.

Private dataset: `nothingnessvoid/pcqm4mv2-ogb-fixed-500k-scnet-v1`.
Original manifest SHA256:
`630d30046d6cdc1f91fb169cd1eb4720bd5b352dc1ebeb641a11ead1ebae9751`.
Train source_idx[0,500000); development[500000,550000), 50K selection rows.
These are official-train-derived and previously consumed for selection, not an
independent test. Official validation, test-dev, test-challenge and Track A
external roles are prohibited. Cached geometry is stripped; no geometry built.

Train-only normalization is transported from the accepted same500K checkpoint:
mean5.413434769952119 eV, sample std1.176728647311213 eV. Source receipt is
`normalization_source.json`. No labels or predictions supply training gradients
outside the declared training role.

## Budget and outcomes

Launch config sets allocation_wall_limit_seconds32400 and cleanup_reserve_seconds180.
The ceiling includes bootstrap, qualification, both concurrent arms, development,
checkpointing and cleanup. Maximum assignment is18 T4 device-hours; this is a
ceiling, not measured cost or Kaggle quota deduction. Record actual entry/device
windows separately; queue and billing tails remain unknown unless observed.

Both arms must pass training-only repeat/resume qualification before either
trains. Keep atomic last state, selected state, aligned finite development
predictions, trace, source/config and exact returned job/version. Budget cutoff
means STOP_FOR_COST; compare only common optimizer-step exposure. No partial
endpoint can masquerade as completed60 epochs. No performance-based early stop.

If both finish, compare frozen selected endpoints and the matched final exposure;
record paired-row uncertainty and time reduction as different measurements.
Gain>=0.003eV and positive paired-row95% interval may nominate a separate review,
never automatic promotion. Single seed and shared-host timing do not establish
training variance or whole-platform cost savings. Native/mechanical/scientific/
replay acceptance remain separate. Infer no benefit from EMA at500K beforehand.
