# Frozen protocol — QM9 Fourier EdgeState dynamics

## Question

Does replacing only the persistent real-bond update MLP with a single-harmonic
Fourier-KAN provide a material direct-Gap architecture gain on the efficient
one-slot global skeleton?

## Arms

- `full_gps`: fresh unchanged EdgeState GPS9 + RWSE16 absolute anchor.
- `neural_atom_k1`: fresh one-slot latent global replacement from the preceding
  accepted comparison; matched control for the candidate skeleton.
- `fourier_edge_k1`: identical to `neural_atom_k1`, except every persistent
  64-dimensional EdgeState proposal uses a single-harmonic Fourier-KAN instead
  of the two-linear SiLU MLP.

The Fourier layer is
`sum_i(a_ji cos(x_i) + b_ji sin(x_i)) + bias_j` after LayerNorm. No KA-GNN
proximity edge or other input/architecture component is imported.

## Frozen training contract

- accepted pure-2D cache aggregate
  `80a2d256ccbbde8de5862e7ad4fd61ea7c1255a19217e590033a94cabe1c6340`;
- exactly 30,000 QM9 train and 3,000 internal validation graphs;
- direct Gap, OGB categorical atom/bond features, RWSE16;
- seed 42, FP32, physical batch 128 per independent model/device;
- AdamW `4e-4`, weight decay `1e-5`, gradient clip 1;
- cosine 40 epochs to `1e-6`, identical row order and sample exposure;
- Kaggle2 T4x2: GPU0 trains `full_gps`; GPU1 sequentially trains
  `neural_atom_k1` and `fourier_edge_k1` with independent optimizer,
  checkpoint, RNG reset, and output state.

No QM9 held-out/test graph, official PCQM validation/test-dev, checkpoint,
pretraining, teacher, residual target, prediction fusion, 3D/ETKDG geometry,
non-covalent edge, desktop/SCNet result, IMS asset, or full-scale result enters
the comparison.

## Remote preflight

- all shared K1/Fourier parameters and buffers have identical initialization;
- exactly the nine EdgeState proposal modules differ;
- the candidate implements the frozen manual `cos/sin/einsum` equation with
  one harmonic and finite outputs/gradients;
- all arms complete a finite forward/backward on one visible GPU;
- candidate parameter count is at most 4.2M;
- no role or batch contract changes.

## Gates

`fourier_edge_k1` is nominated for exactly one paired PCQM-100K transfer only
if all conditions hold:

- gain versus `full_gps` is at least `0.003 eV`;
- gain versus `neural_atom_k1` is at least `0.001 eV`;
- mean epoch-time ratio versus `neural_atom_k1` is at most `1.25`;
- peak-memory reserve is at least 15%; and
- all acceptance and artifact-hash checks pass.

A scientific failure closes harmonic-count, KAN width, placement, seed,
optimizer, and schedule variants of this exact question. An infrastructure-only
failure may be repaired under the unchanged scientific contract.
