# Kunshan projected-moment readout seed-42 protocol

Protocol date: 2026-09-06. Plan: `kunshan_discovery_plan.md`, K2.

## Question and architecture

Keep the accepted GraphState9 encoder unchanged. The control uses final mean
pooling. The candidate adds one final, permutation-invariant readout residual:

1. normalize each final 192-channel atom state;
2. project it without bias to 32 channels and apply SiLU;
3. pool the projected first moment and centered second moment;
4. normalize the concatenated 64 channels and map them back to 192 channels;
5. add that map to the ordinary mean embedding before the unchanged Gap head.

The 64-to-192 return is zero initialized, so the candidate starts with exactly
the control prediction. It adds 18,944 parameters for an expected total of
3,684,753, below the 4M ceiling. There is no query attention, max pooling,
extra GraphState exchange, new geometry, target residual, or prediction fusion.

## Frozen comparison

Use cache `3e4206fd239942ab79f9c4978cd4334f6025ba14e8f1cd1df78c20060a0d1d22`:
100,000 official-train-derived training graphs and 10,000 internal-validation
graphs. Official validation and test-dev stay unread. Train a fresh GraphState9
control and candidate sequentially on one Kunshan DCU with seed42, FP32,
batch48, workers0, AdamW `1.6e-4`, weight decay `1e-6`, train-standardized L1,
cosine decay to `1e-6`, at most 40 epochs, and patience8.

Remote preflight must verify exact parameter counts, common initialization,
initial prediction identity, a finite nonzero new-path gradient, and batch
separation. Atomic epoch checkpoints preserve model, optimizer, scheduler and
all RNG states. Mechanical acceptance requires complete hashes, aligned unique
validation rows, finite recomputed MAE, unchanged roles, and no model inference.

Promotion requires at least `0.001 eV` lower paired validation MAE, no more
than 1.5x control epoch time, and at least 15% memory reserve. A smaller gain is
retained as an observation without seeds43/44 or full-data authorization.
