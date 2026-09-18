# Frozen protocol - K1 topology-conditioned sparse pairs

## Architecture

Retain complete K1-v4. Immediately after the existing layer-6 molecular-slot
exchange, add one zero-initialized residual path:

```text
directed pairs within three topological hops
  -> order-1..5 non-backtracking path counts
  -> shortest-distance bucket + pairwise RWSE
  -> per-pair LayerNorm
  -> learned sender selection per receiving atom
  -> directional pair-to-receiver return
```

The candidate has no persistent dense all-pair state, self pair, second graph
token, coordinate input, geometry target, fusion, teacher, or auxiliary loss.
The bounded path counts are a documented proxy inspired by simple-path
structural encoding; they are not described as exact SPSE.

## Frozen training contract

- Data manifest SHA-256:
  `1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d`.
- Train rows `[0,100000)`; development rows `[100000,150000)`.
- Direct Gap, seed 42, deterministic FP32/no TF32, physical BS128,
  `drop_last`, 40 epochs and 31,240 optimizer steps.
- AdamW, learning rate `4e-4`, weight decay `1e-5`, gradient clip `1.0`,
  cosine decay to `1e-6`, normalized-Gap L1.
- Reuse K1-v4; do not retrain the baseline.

## Gates

Profile both Kaggle T4 and Kunshan DCU with the same source, batch and graph
identity. Select the higher-throughput platform only after both profiles pass
exact nesting, finite two-step gradients, at least 15% memory reserve and a
projected runtime below six hours.

Scientific qualification requires at least `0.003 eV` development gain and a
strictly favorable paired bootstrap interval. A scalar-only pass remains
pending until the frozen K1 prediction payload is aligned and accepted.
