# Frozen protocol — K1 return allocation round 3

Protocol date: 2026-09-13.

## Causal question

Does decoupling K1's learned source atoms from its recipient atoms improve
transfer without adding capacity or changing global information content?

## Architectures

- Reference: immutable 3,658,817-parameter `neural_atom_k1_v4`.
- Uniform return: 3,658,817-parameter
  `neural_atom_k1_uniform_return`; learned source pooling and original slot
  processing, followed by normalized `1/N` return to valid atoms.
- Inverse-score return: 3,658,817-parameter
  `neural_atom_k1_inverse_return`; learned source pooling and original slot
  processing, followed by softmax of the negative source logits.

Both candidates retain the nine-layer persistent real-bond EdgeState path,
RWSE16, 192 hidden channels, one 64-channel molecular slot, exchange layers
3/6/9, mean graph pooling, and direct Gap head. Return mass is one per graph.
They add no parameters and equal K1's initial graph function exactly.

## V4 comparison contract

Use only `kaseichou/pcqm4mv2-ogb-fixed-100k-v1`, manifest SHA-256
`1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d`
and aggregate SHA-256
`bc83a4bd9a7fd7fd6fc6fd085d78ba3caab0e40f997d1932e0f344e701dd40c5`.
Train source rows `[0,100000)` and select only on source rows
`[100000,150000)`. Official validation, test-dev, and test-challenge remain
unread.

Each arm uses seed 42, deterministic FP32 with TF32 disabled, physical batch
128, no accumulation, `drop_last=true`, 40 epochs, 781 optimizer steps and
99,968 sample presentations per epoch, AdamW at `4e-4`, weight decay `1e-5`,
gradient clip 1.0, normalized direct-Gap L1, and cosine decay to `1e-6`.
Row-order SHA-256 is
`e85736669a04029e0fa40e993a085b2e0226b4be98a923a141f999a672ce3f34`.
The accepted K1-v4 reference is reused without retraining.

## Isolation and preflight

One private Kaggle2 T4x2 script launches one independent process per candidate,
with one visible T4 and independent RNG, model, optimizer, scheduler,
checkpoint, and output directory. It rejects every other accelerator layout.

Before training, each arm must verify exact shared K1 initialization, exact
initial predictions, expected parameter count, physical batch 128, finite
gradient flow after two optimizer steps, one-slot source and return mass equal
to one, zero padding mass, and its declared recipient equation. Atomic epoch
checkpoints, best payloads, hashes, and runtime certificates are mandatory.

## Acceptance and stopping

No-inference acceptance recomputes aligned 50,000-row development MAE against
the immutable reference. Promotion requires matching contracts and hashes, at
least 15% memory reserve, gain of at least `0.003 eV`, and a paired
candidate-minus-reference absolute-error bootstrap 95% interval entirely below
zero.

A passing candidate earns shortlist status only. A negative result closes this
mechanism and the three-round K1 sequence. No result automatically permits an
extra seed, shadow access, scale-up, official-role read, or submission.
