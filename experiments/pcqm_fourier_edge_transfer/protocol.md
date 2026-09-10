# Frozen protocol — PCQM-100K Fourier EdgeState transfer

## Question

Does the single-harmonic Fourier replacement that passed QM9 transfer to the
official-train-derived PCQM-100K direct-Gap task when separated from its
one-slot global skeleton?

## Arms and causal controls

- `full_gps`: fresh unchanged EdgeState GPS9 + RWSE16 absolute anchor.
- `neural_atom_k1`: fresh one-slot latent-global skeleton control.
- `fourier_edge_k1`: the same K1 model with only all nine persistent
  EdgeState proposal MLPs replaced by the accepted single-harmonic Fourier
  function.

No harmonic, width, layer placement, feature, or readout is changed from the
accepted QM9 candidate.

## Frozen data and training contract

- accepted official-train-derived PCQM cache ancestry
  `eb7c843e33f430ac755bc575d80153aba87677cea1ad5bb0dcf73cca906e2c21`;
- exactly 100,000 train and 10,000 internal-validation graphs;
- pure 2D model inputs: OGB categorical atoms, real bonds, and RWSE16;
- cached geometry fields are verified for provenance and removed before model
  batching; no coordinate, distance, angle, wedge, or geometry-valid field
  enters any model;
- seed 42, FP32, physical batch 128 per independent model/device;
- AdamW `4e-4`, weight decay `1e-5`, gradient clip 1;
- cosine 40 epochs to `1e-6`, identical row order and sample exposure;
- Kaggle2 T4x2: GPU0 trains `full_gps`; GPU1 trains `neural_atom_k1` then
  `fourier_edge_k1`, each with fresh deterministic initialization and its own
  optimizer/checkpoint/output state.

Official PCQM validation, test-dev, the future shadow role, QM9 test, desktop
results, checkpoints, pretraining, teachers, residual targets, prediction
fusion, ETKDG/3D inputs, SCNet, IMS, and full-scale work are excluded.

## Gates

The candidate authorizes exactly one once-read shadow audit only if:

- gain versus fresh `full_gps` is at least `0.003 eV`;
- gain versus fresh `neural_atom_k1` is at least `0.001 eV`;
- epoch-time ratio versus K1 is at most `1.25`;
- memory reserve is at least 15%; and
- no-model acceptance, causal initialization checks, role flags, and all
  artifact hashes pass.

Failure requires written causal attribution before another architecture may be
submitted. A scientific failure closes Fourier harmonics, KAN width,
placement, seed, optimizer, and schedule variants. An infrastructure-only
failure may be repaired only under this unchanged contract.
