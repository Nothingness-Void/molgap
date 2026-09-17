# GPTrans Pair PreNorm 500K V5 protocol

## Question

Does the only GPTrans relation-flow mechanism that passed the frozen 100K gate
retain a material benefit at 500K under an exact V5 matched comparison?

## Why a new reference is required

The historical GPTrans 500K run is positive context but predates the V4/V5
drop-last contract. It cannot be promoted into a strict V5 comparator. This
experiment trains one reference exactly once and freezes its artifacts for all
future candidates with this complete fingerprint. It is not a per-candidate
baseline habit.

## Frozen comparison

- arms: unchanged GPTrans-T core and parameter-free Pair PreNorm;
- fixed accepted 500K train / 50K development cache;
- seed 42, strict FP32, one DCU and physical BS128 per arm;
- 60 epochs, 234,360 optimizer steps and 29,998,080 presentations per arm;
- AdamW `1e-3`, weight decay `0.05`, warmup 4 epochs, cosine to `1e-6`,
  gradient clipping 1.0 and EMA 0.9999;
- best development EMA selection;
- no geometry input, teacher, pretraining, fusion, extra seed, official
  validation, test-dev, or test-challenge access.

The only candidate mechanism is channel-wise LayerNorm on each pair state
before all twelve attention blocks. It adds no parameters. The two arms use
independent processes and accelerators but identical scientific contracts.

## V5 gates

Each arm first passes a separate optimizer-inclusive preflight with deterministic
repeat steps and at least 15% memory reserve. Training jobs are scheduler-bound
to their own successful preflight. Checkpoints include model, EMA, optimizer,
RNG, trace, next epoch, contract, source and runtime identities.

Acceptance is inference-free and requires complete aligned 50K prediction
bundles, hashes, matching contracts, paired error analysis and deterministic
bootstrap. Pair PreNorm qualifies only if its gain is at least `0.003 eV` and
the paired 95% upper bound is below zero. A qualifying 500K result can create a
`READY_FOR_DESKTOP` evidence package; it never launches full training or reads
protected roles.
