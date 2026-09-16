# Frozen protocol — K1 all-pair relation token

## Scientific question

Does K1 benefit from one pre-normalized all-pair relation bottleneck while its
existing local EdgeState and three atom-slot exchanges remain unchanged?

At layer 6 only:

```text
current node states
  -> ordered node-pair channels
  -> per-pair LayerNorm
  -> one learned-query relation token
  -> zero-initialized uniform graph broadcast
  -> unchanged K1 layers 7-9
```

There is no atom-to-atom attention matrix, geometry, new graph cache, target
residual, prediction fusion, teacher, or auxiliary label. The return projection
is zero initialized, making the candidate function exactly K1 at initialization.

## Immutable v4 benchmark

Use the accepted cross-platform fixed PCQM cache with manifest SHA-256
`1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d`
and aggregate SHA-256
`bc83a4bd9a7fd7fd6fc6fd085d78ba3caab0e40f997d1932e0f344e701dd40c5`.
Train rows are `[0,100000)` and development rows are `[100000,150000)`.
Official validation, test-dev, and test-challenge remain sealed.

Use seed 42, direct Gap, deterministic FP32/no TF32, one accelerator, physical
batch 128, `drop_last`, 40 epochs, 31,240 optimizer steps, 3,998,720 sample
presentations, AdamW (`lr=4e-4`, `weight_decay=1e-5`, clip 1.0), cosine decay to
`1e-6`, normalized-Gap L1, and minimum development MAE selection. Reuse the
immutable K1-v4 reference; do not retrain it.

## Gates

The same-job preflight must verify exact K1 initial predictions, one layer-6
relation token, normalized all-pair assignment mass, zero padding mass,
zero initial residual, candidate gradients after two optimizer steps,
deterministic recovery, and at least 15% reserved memory.

Saved-artifact acceptance performs no model inference. Promotion requires at
least `0.003 eV` development gain over K1-v4, a favorable paired bootstrap
interval, complete recovery artifacts, and the memory gate. A directional or
negative result closes this mechanism without extra seeds, scale-up, or sealed
role access.

