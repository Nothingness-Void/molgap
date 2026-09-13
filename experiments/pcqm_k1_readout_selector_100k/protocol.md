# Frozen protocol — K1 readout and selector round 1

Protocol date: 2026-09-13.

## Causal questions

1. Can an 8-by-8 RepSet summary add final-set information missing from K1's
   mean graph readout?
2. Can sharing K1's atom selector across exchange layers improve transfer by
   reducing layer-specific addressing freedom?

Only those mechanisms differ from frozen `neural_atom_k1_v4`.

## Architectures

- Reference: 3,658,817-parameter `neural_atom_k1_v4`, reused from its accepted
  P100 v4 record and not retrained.
- RepSet arm: unchanged K1 plus the authors' hidden-set equation with eight
  hidden sets, eight elements per set, and a 64-channel set embedding. A
  zero-initialized 64-to-192 map adds this to the ordinary mean representation
  before the unchanged direct-Gap head. Expected parameters: 3,683,985.
- Tied-selector arm: one shared 192-to-64 key map, normalization, and direct
  64-channel query select atoms at layers 3, 6, and 9. Node-value maps, slot
  seeds, slot attention/FFNs, and return projections remain independent.
  Expected parameters: 3,622,401.

Both candidates must exactly equal K1's initial graph function. They remain
pure 2D and direct-Gap; no geometry, teacher, pretraining, target residual,
prediction fusion, extra seed, or sealed role is allowed.

## V4 comparison contract

Use only `kaseichou/pcqm4mv2-ogb-fixed-100k-v1`, manifest SHA-256
`1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d` and
aggregate SHA-256
`bc83a4bd9a7fd7fd6fc6fd085d78ba3caab0e40f997d1932e0f344e701dd40c5`.
Train source rows `[0,100000)` and select only on source rows
`[100000,150000)`. Official validation, test-dev, and test-challenge remain
unread.

Each independent arm uses seed 42, deterministic FP32 with TF32 disabled,
physical batch 128, no accumulation, `drop_last=true`, 40 epochs, 781 steps
and 99,968 presentations per epoch, AdamW at `4e-4`, weight decay `1e-5`, clip
1.0, normalized direct-Gap L1, and cosine decay to `1e-6`. Row-order SHA-256 is
`e85736669a04029e0fa40e993a085b2e0226b4be98a923a141f999a672ce3f34`.

## Isolation and preflight

One private Kaggle2 T4x2 script launches one process per candidate. Each
process sees one physical T4 and owns an independent model, seed/RNG stream,
optimizer, scheduler, checkpoint directory, runtime certificate, and output.
The job must reject any other accelerator layout.

Before full training, each arm verifies source/cache identity, optimizer-step
determinism, exact K1 initial output, expected parameters, mechanism invariants,
finite nonzero mechanism gradients after two optimizer steps, and physical
batch 128. Epoch checkpoints and best development payloads are written
atomically and hash-manifested.

## Acceptance and stopping

No-inference acceptance recomputes development MAE from the aligned 50,000-row
payload and compares it with the immutable K1-v4 payload. A candidate passes
only if all contracts and hashes pass, memory reserve is at least 15%, MAE gain
is at least 0.003 eV, and the paired absolute-error bootstrap 95% interval is
entirely favorable.

Round 1 ends after both arms receive a dated decision. A passing arm earns only
shortlist status. A failed round requires evidence-based attribution before
the coordinator may release one distinct round-2 experiment; the monitor may
not choose or submit it.

