# QM9 adaptive local denoising protocol

**Frozen: 2026-09-09.** Track C, seed-42 screening only.

## Question

Does molecule-adaptive atom-specific isotropic coordinate corruption produce a
better transferable initialization than fixed isotropic corruption for the
same ETKDG-aware EdgeState encoder?

## Immutable data contract

- Reuse the accepted GAPE-lite QM9 role identities: 30,000 train and 3,000
  validation graphs, split fingerprint `62f1cdefdaec6877`.
- Inputs are OGB 9-channel atom categories, OGB 3-channel real-bond categories,
  RWSE16, and one deterministic ETKDGv3+MMFF94s heavy-atom conformer.
- ETKDG atom order must match the accepted 2D graph exactly. Failed conformers
  remain aligned through an explicit mask and reason code.
- No QM9 test graph is constructed or read. No PCQM official validation,
  test-dev, external row, DFT coordinate, pretrained checkpoint, or target
  outside the train/validation roles is used.
- CPU cache construction and independent no-model acceptance precede GPU use.
  Every shard and aggregate manifest is SHA-256 bound.

## Shared downstream architecture

All three arms use the same inference architecture:
`OGBGeometrySparseTriangleEdgeStateGPSWrapper` in `distance_angle` mode—GPS9,
RWSE16, persistent real-bond EdgeState64, sparse adjacent-edge WedgeState16,
and local ETKDG distance/angle bottom fusion. The prediction is direct Gap in
eV. Pretraining-only noise generators and vector heads are discarded; they are
not inference parameters.

## Paired arms

| Arm | Encoder exposure |
|---|---|
| `scratch40` | 40 direct-Gap epochs |
| `fixed10_gap30` | 10 atom-local vector-denoising epochs at isotropic `sigma=0.1`, then 30 direct-Gap epochs |
| `adaptive10_gap30` | 10 atom-local vector-denoising epochs with invariant atom-specific Gaussian scales, prior `sigma=0.1`, and KL weight `1.0`, then 30 direct-Gap epochs |

The fixed and adaptive arms use the same denoising head and corruption RNG
contract. The temporary denoising head is an equivariant edge-direction
readout: invariant final-node states predict scalar directed-bond coefficients,
which are multiplied by corrupted bond unit vectors and reduced to atom-local
3D vectors. Thus the objective cannot learn an orientation-dependent Cartesian
shortcut. The adaptive generator consumes only clean, invariant local node
environments. Reparameterized sampling and the KL term prevent a zero-noise
shortcut. Its log scale is parameterized as
`log(0.1) + 0.5 * tanh(raw)`, bounding sigma to approximately
`[0.0607, 0.1649]` while preserving the `0.1` initialization.

## Training and resource contract

- One private Kaggle task with explicit `NvidiaTeslaT4` allocation.
- GPU 0 runs `scratch40`; GPU 1 runs `fixed10_gap30` and then
  `adaptive10_gap30`. A device never hosts two independent models at once.
- Seed 42, FP32, AdamW, learning rate `4e-4`, weight decay `1e-5`, cosine
  schedule to `1e-6`, gradient clipping `1.0`, physical batch exactly 128,
  no gradient accumulation, and exactly 40 encoder passes. Early stopping is
  disabled because unequal stopping would invalidate the equal-exposure
  comparison; the best validation epoch is still retained independently.
- The three arms start from the same downstream encoder tensor hash. Data row
  order and per-arm RNG streams are recorded.
- Every epoch writes an atomic resumable checkpoint. Models, traces, validation
  payloads, and completion manifests are independently retrievable.

## Acceptance and scientific gate

Acceptance is mechanical and executes no local model inference. It must verify
the cache and source identities, exact same-task/same-platform comparison,
physical batch 128, matched downstream contract and exposure, finite values,
artifact hashes, recomputed per-row validation MAE, parameter counts,
throughput, memory, pretraining loss, and adaptive sigma statistics.

The adaptive arm is nominated for a separately frozen PCQM-100K transfer only
if both conditions hold:

1. it beats `scratch40` by at least `0.003 eV`; and
2. it beats `fixed10_gap30` by at least `0.001 eV`.

The fixed arm may diagnose whether local denoising itself helps, but it cannot
promote this adaptive-noise question. Failure closes this exact architecture,
noise prior, allocation, and seed without another QM9 run. A pass does not
authorize another seed, PCQM work, desktop work, official-role access, SCNet,
or IMS; each requires a new decision.
