# Pure-2D R3 validation tournament protocol

## Question

Can a chemically constrained frontier readout or a better allocation of
persistent pair updates turn the compact R2 near-match into a pure-2D
architecture that strictly beats the completed PairGPS2D refinement?

## Frozen validation tournament

All candidates use the accepted QM9 30,000/3,000/3,000 split, split seed 42,
encoder seed 42, RWSE16, FP32, batch 48, AdamW, learning rate `4e-4`, weight
decay `1e-5`, 20 epochs, and patience eight. Model selection reads only train
and validation. The 3,000-row test role is not constructed or evaluated by the
tournament kernel.

Five candidates are permitted:

1. `edge_state_structural_gps`: the sparse persistent-edge architecture under
   the exact QM9 optimization contract, providing a same-environment anchor.
2. `edge_state_structural_orbital`: the same encoder with a center/Gap head
   that reconstructs HOMO and LUMO exactly.
3. `pair_gps_2d_r3_orbital`: the completed R2 backbone with only the consistent
   center/Gap head changed.
4. `pair_gps_2d_r3_triplet`: the R2 head with learned masked attention over
   triplet intermediates and a true pre-normalized pair residual.
5. `pair_gps_2d_r3_combined`: both R3 repairs in one encoder.

The frontier head operates in eV and then returns the existing normalized
three-target contract. It enforces `LUMO - HOMO = Gap`; it is not a residual
target, frozen-model correction, ensemble, or prediction fusion. All inputs
remain pure 2D.

## Validation and test gates

The frozen PairGPS2D validation reference is average MAE `0.1100691929 eV`
and Gap MAE `0.1318935603 eV`. A candidate is eligible only if both validation
metrics are strictly lower and its measured parameter count is at most
4,800,000. Among eligible candidates, the lowest validation average MAE wins.

If no candidate is eligible, all five branches close without a test read. If
one wins, its already-trained best checkpoint is frozen and a separate kernel
may read the QM9 test role once. That checkpoint must beat the frozen test
average `0.1117789894 eV` and Gap `0.1340952218 eV` before matched
PubChemQC-100K validation is authorized.

## Resource and durability contract

- Kaggle2 budget: at most 30 GPU hours across validation, one frozen QM9 test,
  and any subsequently authorized PubChemQC-100K work.
- The validation tournament has a four-hour expected duration and a six-hour
  stop bound on one GPU. No candidate is silently retried or extended.
- The already accepted, sharded QM9 RWSE cache is mounted read-only. GPU output
  contains only per-candidate atomic checkpoints, models, embeddings, metrics,
  preflight evidence, and an incrementally replaced tournament summary.
- Every candidate must pass a remote FP32 forward/backward with finite values
  and the parameter gate before training starts.
- The user's no-local-model rule replaces local model execution: local work is
  limited to syntax, AST, manifest, and contract validation.
