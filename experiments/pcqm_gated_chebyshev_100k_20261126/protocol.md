# Gated MPNN four-hop Chebyshev 100K

The [Chebyshev paper](https://arxiv.org/abs/1606.09375) gives an efficient sparse polynomial filter. The existing GPS++ screen bundles other paths and does not answer this filter contrast. Both arms here use the same five-layer 192-wide OGB atom/bond gated MPNN, mean readout, symmetric normalized adjacency, and five learned projections per block. The candidate projects T0 through T4. The control uses the identical five independently trainable projections on T0/T1/T0/T1/T0. Parameter shapes and seeded tensors are fixed. The only tested change is the support of the extra diffusion branch. Both arms retain the same local bond-gated update.

Train on the fixed 100K and score only the same 50K internal development rows. Two T4 devices, one arm per card. FP32/no TF32, seed 42, BS128, 60 passes, 46,860 updates, same AdamW, EMA and checkpoint semantics. Wall cap 6 h including cleanup. No coordinates or eigensolve.

Primary comparison uses the main pass-60 EMA. Rescore the verified final checkpoints once with the existing gptrans scorer on exact development rows and match the native trace; compare 2,000 aligned row-bootstrap draws. Nominate only for candidate-minus-control <= -3.0 meV and 95% upper <0. Retain the selected EMA comparison, full traces and measured costs separately through strict RML. No automatic scale transfer.
