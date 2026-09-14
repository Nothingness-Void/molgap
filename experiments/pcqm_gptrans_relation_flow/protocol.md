# Relation-flow protocol — frozen 2026-09-14

## Authorization

First of two newly authorized Kaggle2 rounds (previous loop round 2 of 3).
Two independent candidates in one T4x2 notebook. Desktop SCNet/IMS/full jobs
are excluded. The second round requires controller attribution and its own
protocol before submission; the monitor cannot select a successor.

## Candidates

`pair_prenorm`: per-pair, across-32-channel LayerNorm without affine parameters
or new state, immediately before all 12 attention blocks consume pair state.
Persistent residual pair accumulation and readout remain unchanged.

`centered_logits`: before node-to-pair projection, subtract each head/query's
mean raw logit over valid keys. Node-attention softmax/dropout is untouched.
Padding never enters the mean. Pair accumulation and all other paths remain.

No combination arm. Both have exactly 5,246,817 parameters; seed-42 tensor
state and parameter names match the untrained reference, not a trained warm
start. Construction preserves the reference RNG stream. Novel model metadata
and source identities must persist in preflight/checkpoint/best/completion.

## Scientific contract

Inherit the complete [GPTrans reference contract](../pcqm_gptrans_t_100k_v4/protocol.md)
unchanged, not K1's optimizer/schedule: accepted train100K/dev50K identity;
seed42; physical BS128; FP32/no TF32; 12x256/pair32/8 heads; 60 epochs;
46,860 optimizer steps; 5,998,080 sample presentations; normalized direct Gap;
AdamW LR1e-3/wd0.05, warmup4/cosine1e-6, clip1, EMA0.9999; best dev EMA.
No teacher, pretraining, geometry, extra seeds, shadow or official roles.

Reuse immutable GPTrans reference best-model SHA
`f4da386ae1e32f6953b645c0bdb8e208aaba1f1d7b8ebec132776bd63c22d6ab`.
No baseline retraining. Exact match is enforced by executable v4 comparison,
and a deterministic runtime certificate is emitted for each isolated worker.

## Execution and acceptance

Static/AST, source-inventory and orchestration tests run locally without model
construction or inference. Remote checks first verify parameter/state identity,
padding invariance, centered-logit offset invariance and finite gradients.
Train-only optimizer-inclusive preflight precedes each candidate training;
at least 15% memory reserve and estimated train-only duration <=6 hours.
Notebook hard budget is 10 wall hours, at most 20 T4 device-hours for the round;
allowance for second round must be checked before release. Budget/infra stops
are not scientific losses. No training contract modification on a retry.

Each worker has one visible GPU and isolated optimizer/RNG/checkpoints/logs.
Preflight/training use fresh subprocesses. Atomic per-epoch checkpoints include
optimizer, scheduler, EMA and RNG; best model/predictions and trace are separate.
Kaggle output capture plus per-epoch checkpoint chunks preserve partial evidence.
On failure, preserve both workers' results; never rerun a completed worker.

Mechanical acceptance recomputes all 50K aligned dev errors, verifies source,
data, initial state, model/checkpoint/payload hashes, finite values and matching
v4 contracts. Gain >=0.003 eV plus paired row-bootstrap upper95%<0 nominates a
mechanism shortlist only. Bootstrap does not measure training stochasticity.
Neither a small gain nor a favorable comparison with a differently optimized
K1 result establishes superiority over K1 or full-scale transfer.
