# Frozen protocol — K1-G and K1-R PCQM-100K v4 screen

## Question

Can either of two isolated global-allocation changes improve frozen Neural-Atom
K1 on the accepted cross-platform PCQM4Mv2 100K benchmark?

## Arms

- `neural_atom_k1_v4`: unchanged 192-wide, nine-layer local persistent
  EdgeState K1 with one 64-channel atom/global slot at layers 3, 6, and 9.
- `neural_atom_k1_g`: K1 plus one bounded scalar gate per molecule at each
  exchange layer. Each gate sees current node mean/max pooling and starts at
  exactly one, with range 0.5--1.5.
- `neural_atom_k1_r`: K1 plus one 32-channel relation slot at each exchange
  layer. It attentively pools the current directed real-bond EdgeState and
  joins it with the existing atom slot before a zero-initialized node return.

G and R are independent candidates. They are never combined, and neither adds
geometry, path caches, atom slots, prediction fusion, residual targets,
pretraining, or teachers.

## V4 scientific contract

- Dataset: private accepted mirror
  `kaseichou/pcqm4mv2-ogb-fixed-100k-v1`, manifest SHA-256
  `1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d`.
- Roles: official-train-derived rows 0--99,999 for training and
  100,000--149,999 for development. Official validation and every test role
  remain absent and unread.
- Inputs: OGB nine-field atom categories, OGB three-field real bonds, RWSE16;
  cached geometry is physically removed before batching.
- Direct scalar Gap in eV; train-role mean/sample-standard-deviation transform;
  normalized L1 loss; minimum development Gap MAE selection.
- Seed 42, strict FP32/no TF32, deterministic algorithms, physical BS128 on
  one visible GPU per model, no accumulation, `drop_last=True` for training.
- Exactly 781 optimizer steps and 99,968 sample presentations per epoch for 40
  epochs: 31,240 steps and 3,998,720 presentations per arm.
- AdamW `4e-4`, weight decay `1e-5`, clip 1.0, cosine to `1e-6`.
- A platform/runtime certificate must pass a repeated real optimizer-step
  calibration before each arm trains.

The P100 notebook trains only the reference. The T4x2 notebook isolates K1-G
and K1-R in separate processes, RNGs, optimizers, checkpoints, and GPUs.

## Gate

The one K1-v4 result becomes the reusable frozen baseline for this complete
scientific contract. Each candidate must improve it by at least 0.003 eV, have
a paired-row bootstrap interval entirely in its favor, keep at least 15%
reserved-memory headroom, complete all 40 epochs, and pass artifact/hash
acceptance. A smaller improvement is directional evidence only. This screen
does not authorize another seed, shadow-role access, scale-up, full training,
official evaluation, or submission.

