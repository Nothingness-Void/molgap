# Persistent pair readback — frozen 2026-09-14

## Single question, two independent arms

Let P be the incoming pair state, D the unchanged GPTrans node-to-pair update,
and M=P+D. The original direct pair-to-node message is softmax(D) * D summed
over valid keys. All softmax operations mask padded keys.

- `memory_value`: softmax(D) * M. Only the value source changes.
- `memory_message`: softmax(M) * M. Both direct-message routing and value source
  use accumulated relation memory. It tests the complete readback alternative.

The latter is not a factorial interaction claim: differences between these two
arms address routing conditional on memory values. Node attention, raw logits,
pair residual update, FFN, graph readout and data are unchanged. Neither arm
uses pair normalization or centered logits. No new parameters, 5,246,817 each;
identical untrained seed42 tensors, no trained checkpoint initialization.

## Immutable reference and contract

Inherit all of [GPTrans V4](../pcqm_gptrans_t_100k_v4/protocol.md): fixed 100K
train/50K dev; OGB atom/bond plus shortest path20; seed42; 12 layers256/pair32,
8 heads; physical BS128/drop_last/781 steps per epoch; 60 epochs/46,860 steps;
5,998,080 presentations; strict FP32/noTF32; AdamW1e-3/wd0.05, warmup4/cosine
1e-6, clip1, EMA0.9999, normalized Gap L1, best development EMA selection.
Do not match to K1's different optimizer, or reuse a 500K scalar as comparator.
Reference model SHA:
`f4da386ae1e32f6953b645c0bdb8e208aaba1f1d7b8ebec132776bd63c22d6ab`.
No reference retraining. No official/shadow/test/geometry/teacher/pretraining.

## Gates, budget and recovery

This consumes the second and final reopened round: no successor scientific
submission is authorized after its terminal result. No extra seed/scale or
desktop SCNet/IMS access. Run two isolated T4 workers with independent RNG,
optimizers, logs and atomic checkpoints; preserve complete RNG when wrapping
the frozen model. Local tests are AST and orchestration only; remote synthetic
checks cover full-model finite forward/backward, parameter/init identity,
masked readback invariance and dependence on persistent memory. No new graph
cache construction. Train-only preflight requires deterministic repeated
optimizer steps, >=15% reserve and estimated training <=6 hours per arm.

Prior round consumed about 6.13 T4 device-hours. Exact remaining account quota
is not API-verifiable; do not claim a balance. Request one T4x2 notebook only;
if Kaggle rejects allocation for quota, stop and preserve the prepared release.
Expected duration is about 3-4 wall hours (6-8 device-hours) based on first-round
measurements, not guaranteed. Hard wall ceiling 10 hours; no auto-extension.
Atomic per-epoch resume and independent checkpoint chunks every ten epochs;
all completed arms retained, never retrained after a partial failure.

Acceptance recomputes saved metrics/paired bootstrap, hashes all artifacts and
checks matching V4 fingerprints/certificates. Gain >=0.003 and bootstrap
upper95%<0 yields a mechanism shortlist only. Also report against the separately
shortlisted PreNorm score as context, not an interaction or scale claim.
Source/best/checkpoint metadata must identify which memory variant was trained.
