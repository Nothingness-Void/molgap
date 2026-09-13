# Frozen protocol — K1 single-slot processor round 2

Protocol date: 2026-09-13.

## Causal question

Does collapsing or removing K1's mathematically degenerate length-one latent
self-attention improve transfer while preserving its atom-selection and
broadcast bottleneck?

## Architectures

- Reference: immutable 3,658,817-parameter `neural_atom_k1_v4`.
- Collapsed arm: 3,633,857-parameter
  `neural_atom_k1_collapsed_mha`. Each exchange retains atom keys/values,
  one slot, the slot value and output projections, both slot norms, the slot
  FFN, and the node return. Slot query/key projections are absent.
- No-attention arm: 3,608,897-parameter
  `neural_atom_k1_no_slot_attention`. Each exchange retains atom keys/values,
  one slot, both slot norms, the slot FFN, and the node return, but has no slot
  self-attention module.

The local nine-layer persistent real-bond EdgeState path, RWSE16, 192 hidden
channels, exchange layers 3/6/9, one 64-channel slot, mean graph pooling, and
direct Gap head remain unchanged. Both candidates exactly equal K1's initial
graph function through the zero-initialized node return.

## V4 comparison contract

Use only `kaseichou/pcqm4mv2-ogb-fixed-100k-v1`, manifest SHA-256
`1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d`
and aggregate SHA-256
`bc83a4bd9a7fd7fd6fc6fd085d78ba3caab0e40f997d1932e0f344e701dd40c5`.
Train source rows `[0,100000)` and select only on source rows
`[100000,150000)`. Official validation, test-dev, and test-challenge remain
unread.

Each arm uses seed 42, deterministic FP32 with TF32 disabled, physical batch
128, no accumulation, `drop_last=true`, 40 epochs, 781 steps and 99,968 sample
presentations per epoch, AdamW at `4e-4`, weight decay `1e-5`, gradient clip
1.0, normalized direct-Gap L1, and cosine decay to `1e-6`. Row-order SHA-256 is
`e85736669a04029e0fa40e993a085b2e0226b4be98a923a141f999a672ce3f34`.
The accepted K1-v4 reference is reused without retraining.

## Isolation and preflight

One private Kaggle2 T4x2 script launches one process per candidate, with one
visible T4 and independent RNG, model, optimizer, scheduler, checkpoint, and
output per process. The job rejects any other accelerator layout.

Before training, both arms must verify exact shared K1 initialization and exact
initial predictions, expected parameters, physical batch 128, finite gradients
after two optimizer steps, and removal of every slot-attention module. The
collapsed arm must additionally reproduce a fresh K1 length-one attention
output from the copied value/output projections in evaluation mode. Atomic
epoch checkpoints, best payloads, hashes, and runtime certificates are
mandatory.

## Acceptance and stopping

No-inference acceptance recomputes the aligned 50,000-row development MAE and
compares each arm with the immutable reference. Promotion requires all
contracts and hashes, at least 15% memory reserve, gain of at least `0.003 eV`,
and a paired candidate-minus-reference absolute-error bootstrap 95% interval
entirely below zero.

A passing arm earns shortlist status only. A negative round receives a dated
attribution before the final authorized round. No result automatically permits
another seed, shadow access, scale-up, official-role read, or submission.

