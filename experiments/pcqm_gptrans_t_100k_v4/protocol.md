# GPTrans-T 100K V4 reference protocol

## Question

What is the one reusable GPTrans-T baseline for later architecture candidates
under the reference-screen V4 contract?

## Immutable contract

- Official PCQM4Mv2 training role only.
- Train source indices `0:100000`; development `100000:150000`.
- Accepted graph manifest SHA-256
  `1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d`.
- Twelve GPTrans layers, node width 256, pair width 32, eight heads, shortest
  path cap 20, direct normalized Gap target, seed 42.
- Strict FP32, TF32 disabled, deterministic algorithms required.
- One visible accelerator, physical BS128, no accumulation, global epoch
  permutation, and exactly 781 full optimizer batches per epoch. The final 32
  rows of each permutation are excluded; no short optimizer batch is allowed.
- AdamW with `foreach=False` and `fused=False`, LR `1e-3`, weight decay `0.05`,
  four-epoch warmup, 60-epoch cosine schedule to `1e-6`, clipping 1.0, and EMA
  0.9999.
- Best development EMA selection. The reused development role makes this a
  selection reference rather than an unbiased final estimate.
- Stochasticity and material-gain floors are both `0.003 eV`.

The baseline is trained once. A later candidate may run on another platform
only when all V4 scientific fingerprints match and both runtime certificates
pass `validate_reference_screen_contract`.

## Execution gates

The preflight hashes every input shard, runs two identical optimizer steps,
requires bitwise-identical losses and model states, measures optimizer-inclusive
memory and throughput, and emits `molgap-runtime-certificate-v1`. Training
refuses a changed runtime, source archive, cache, initialization, or certificate.

Every epoch atomically saves model, optimizer, scheduler, EMA, RNG, trace, and
the exact next-epoch cursor. Best weights and aligned 50K predictions are stored
separately. Finiteness is checked every 50 optimizer steps without forcing a
host synchronization on every batch. No completion claim is accepted without
local recomputation of the development MAE and all hashes.
