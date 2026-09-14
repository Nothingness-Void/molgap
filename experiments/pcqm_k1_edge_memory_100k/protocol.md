# K1 real-bond storage/read separation

## Question

Does repeatedly normalizing the *stored* real-bond residual discard useful
history, or is that normalization essential regularization? This is not adding
another LayerNorm: K1 already has both context and output LayerNorms.

Write `F_l(z)` for the existing context-normalized edge MLP and `N_l` for
the existing affine output LayerNorm. Node contributions are `C_l(h)`.

| Arm | Persistent storage | Delta | Read passed to local node block |
|---|---|---|---|
| Frozen K1 (not retrained) | `E'=N_l(E+delta)` | `F_l(E+C_l(h))` | `E'` |
| `neural_atom_k1_edge_read_norm` | `E'=E+delta` | `F_l(E+C_l(h))` | `N_l(E')` |
| `neural_atom_k1_edge_context_read_norm` | `E'=E+delta` | `F_l(N_l(E)+C_l(h))` | `N_l(E')` |

The second arm is a normalization-placement ablation of the first, not a
stack of previous winners. All 3,658,817 parameter tensors retain their seed42
initial values and names under the base encoder. Nine local blocks, OGB
features, RWSE16, slot64 at layers3/6/9, learned return, pooling and Gap head
are unchanged. Initial *functions* need not match because the information
flow changes immediately; acceptance explicitly records this instead of
claiming a zero-initialized nested extension.

## Immutable benchmark

Reuse `../pcqm_k1_variants_100k/training_contract.json` for all scientific
fields except architecture/run identity and arms. Only the two arms above are
submitted. Kaggle2 fixed100K manifest SHA
`1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d`;
train `[0,100000)`, development `[100000,150000)`. Seed42, physical BS128,
strict FP32/noTF32, drop-last, 40epochs/31,240steps/3,998,720presentations,
AdamW4e-4/wd1e-5, cosine1e-6, clip1, normalized Gap L1, minimum-dev selection.
No EMA, pretraining, teacher, geometry, fusion or extra seeds. Official,
shadow and test roles remain sealed. Existing K1-v4 reference is reused.

## Execution and acceptance

Two T4 workers have independent model/RNG/optimizer/checkpoints. A train-only
preflight checks identical base tensors, all nine storage/read equations,
finite states/gradients, parameter count and optimizer-inclusive runtime
calibration. Remote checks measure memory/read RMS; they do not establish that
historical activations exploded. No model executes locally.

Epoch checkpoints contain optimizer, scheduler, Python/NumPy/CPU/CUDA RNG,
source/runtime/data/order identities and best-artifact hashes. Every ten
epochs a separate recovery archive includes the checkpoint and selected
weights/payload. Resume requires exact identity; never silently restart from
scratch or change scientific settings. Runtime is capped at ten wall hours;
failure retains artifacts. Packaging validates committed source inventory
before dependency installation, avoiding the former missing-src.zip failure.

Frozen no-inference acceptance rehashes artifacts, checks finite ordered
50K predictions, recomputes MAE and validates V4 scientific fingerprints and
runtime certificates. Shortlist requires >=0.003eV gain against K1-v4,
paired-bootstrap upper95%<0 and >=15% reserved-memory margin. Below-threshold
directional gains are not promotion. Reused dev and a single seed limit claims.

## Three-round authority

The user's 2026-09-15 authorization releases at most three new rounds;
`results/authorization.json` is the counter. This is round1. Controller analysis
of accepted terminal evidence must precede round2/3 selection. Neither monitor
nor runner chooses successors. No blind submission of three notebooks, closed
route rescue, extra seed, scale bridge or desktop/SCNet/IMS work is released.
Infrastructure retries preserve the frozen contract and are separately recorded.
If both arms fail, close this storage/read question rather than tuning its gain,
norm, width or optimizer. A material win is a shortlist only; a later isolated
interaction requires a new frozen protocol within the remaining round budget.
