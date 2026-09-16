# K1 PairToken matched-V4 500K bridge

## Question

Does the normalized learned cross-node PairToken that passed the fixed 100K
screen preserve a material advantage over frozen K1 when scaled to the accepted
fixed 500K/50K PCQM roles?

## Frozen change

The only architecture change relative to K1-v4 is one 32-channel
pre-normalized all-ordered-node-pair token after layer 6. Its return projection
is zero initialized, so the candidate is exactly K1 at initialization. It adds
22,848 parameters and no geometry, teacher, target residual, prediction fusion,
or dense atom-to-atom attention.

## Matched contract

The executable contract is `training_contract.json`. It exactly matches the
accepted `pcqm-fixed500k-dev50k-matched60-v4` K1 reference: fixed cross-platform
graph bytes, seed 42, deterministic FP32/no TF32, physical BS128, drop-last,
3,906 steps per epoch, 60 epochs, unfused/non-foreach AdamW, normalized L1,
epoch-indexed cosine schedule, and raw-model development selection. Platform
identity may differ after an optimizer-inclusive runtime certificate passes.

The frozen K1 reference is `0.10485986978054046 eV`. It is not retrained. The
candidate must improve by at least `0.003 eV`; row-paired inference is performed
only after the baseline prediction payload is retrieved. Official validation,
test-dev, test-challenge, and shadow roles remain sealed.

## Durability

Every epoch atomically replaces `last_checkpoint.pt`, `trace.json`, and
`progress.json`; selected model and aligned development predictions are retained
separately. A scheduler interruption may resume only from a hash-accepted
checkpoint with the same source, runtime and scientific contract.

## Decision boundary

Passing makes PairToken a 500K shortlist mechanism only. It does not authorize
another seed, full training, official-role access, or production promotion.

