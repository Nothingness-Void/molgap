# Protocol: K1 StructuralLite Slot 100K

## Question

Can K1 combine a distinct structural input signal with a simpler molecular
exchange path strongly enough to clear the material 100K gate?

## Single candidate

The candidate keeps K1-v4 node width, nine local persistent-edge blocks,
RWSE16, direct normalized Gap objective, and molecular exchanges at layers 3,
6 and 9. It makes exactly two previously isolated directional changes:

- add the zero-initialized selective MoSE residual beside RWSE16;
- remove the length-one slot self-attention and use uniform normalized return.

No pair state, dense node attention, geometry, pretraining, auxiliary target,
fusion, router, extra slot, new readout, or scalar exchange-strength control is
allowed.

## Frozen screen

- Roles: fixed official-train-derived 100,000 train / 50,000 development.
- Reference: immutable aligned K1-v4 seed-42 prediction payload.
- Seed 42; deterministic FP32; TF32 disabled.
- Physical batch 128 per device; one visible device; `drop_last`.
- AdamW `4e-4`, weight decay `1e-5`, clipping `1.0`; cosine; 40 epochs.
- Direct normalized Gap L1; best development checkpoint selected each epoch.
- Exactly 31,240 optimizer steps and 3,998,720 sample presentations.
- Atomic checkpoints, payload, manifest, source/config identity, and runtime
  certificate are mandatory.
- Official validation, test-dev, and test-challenge remain unread.

Preflight must prove the exact reference-compatible K1 initialization, the
zero initial MoSE contribution, uniform return mass, absence of slot
self-attention, finite forward/backward values, exact parameter count, and one
device memory fit.

## Decision rule

Promotion requires all of:

- candidate gain at least `0.003 eV` over immutable K1-v4;
- candidate-minus-reference paired absolute-error interval entirely below zero;
- complete V5 mechanical acceptance and aligned source indices/targets;
- no protected-role access and no contract deviation.

Failure closes this composition. Passing only permits a separately authorized
500K transfer/cost decision; it does not authorize full-scale training.

## Existing evidence pointers

- K1 combined simplification decision at Git commit `7f1f713`.
- K1 selective MoSE residual decision at Git commit `ff85edd`.
- K1 causal audit terminal state at Git commit `6829d01`.
- K1 PairToken 500K transfer closure and later pair-route terminal decisions
  remain on their owning experiment branches.

