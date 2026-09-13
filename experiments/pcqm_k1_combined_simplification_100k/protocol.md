# Frozen protocol — K1 combined simplification

## Question

Do the two separately favorable K1 simplifications combine into a material
gain, or do they encode the same effective regularization?

## Candidate

`neural_atom_k1_no_attention_uniform_return` retains the nine-layer persistent
real-bond EdgeState path, RWSE16, 192 hidden channels, learned atom-to-slot
pooling, one 64-channel slot, slot FFN, exchanges at layers 3/6/9, mean graph
pooling, and the direct Gap head. It removes the mathematically degenerate
length-one slot self-attention and returns the processed slot uniformly with
unit mass over valid atoms. Expected parameters: `3,608,897`.

## V4 contract

Use the accepted Kaggle1 mirror `nothingnessvoid/pcqm4mv2-ogb-fixed-100k-v1`,
manifest `1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d`.
Train `[0,100000)` and select only on `[100000,150000)`. Use seed42,
deterministic FP32/no-TF32, physical BS128, no accumulation, 40 epochs, 781
full steps per epoch, AdamW `4e-4`, weight decay `1e-5`, clip 1.0, normalized
Gap L1, and cosine decay to `1e-6`. Row-order fingerprint is
`e85736669a04029e0fa40e993a085b2e0226b4be98a923a141f999a672ce3f34`.

Preflight must prove exact K1 predictions at initialization, removal of all
three slot-attention modules, one learned source distribution per graph,
uniform unit-mass return, zero padding mass, finite gradients, exact parameter
count, and an accepted Kaggle1 runtime certificate.

## Decision gate

Primary promotion requires at least `0.003 eV` gain over the immutable K1-v4
reference, paired bootstrap upper bound below zero, and at least 15% memory
reserve. Acceptance also compares aligned errors against both parent
simplifications to identify additivity or redundancy. A negative result closes
the combination; no extra seed or scale-up follows automatically.

Official validation, test-dev, and test-challenge remain unread.
