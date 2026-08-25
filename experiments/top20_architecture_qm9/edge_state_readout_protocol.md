# EdgeState readout fallback protocol

## Question

If every R3 candidate misses the frozen validation gate, can the accepted
persistent-edge encoder improve sample efficiency by exposing its final node
and bond states through a learned molecule readout?

This fallback was frozen before any R3 validation result was observed. It is
not run when R3 produces an eligible winner.

## Architecture

`edge_state_structural_readout` keeps the complete RWSE16, nine-layer,
192-channel EdgeState Structural GPS encoder unchanged. Its readout contains:

- learned softmax attention over final node states;
- mean pooling of the final 64-channel directed bond states;
- a 32-channel bottleneck that maps attentive-node deviation and bond summary
  to an additive graph-embedding update.

The bottleneck's final projection is initialized to zero. The candidate is
therefore exactly the mean-pooled EdgeState model at initialization, while all
new components remain trainable end to end. It uses no prior checkpoint,
prediction residual, ensemble, fusion, coordinates, or conformer.

## Trigger and frozen gate

The run is permitted only if the five-candidate R3 tournament completes with
no eligible validation winner. It reuses the same accepted QM9
30,000/3,000/3,000 split, split and encoder seed 42, RWSE16 cache, FP32,
batch 48, AdamW, learning rate `4e-4`, weight decay `1e-5`, 20 epochs, and
patience eight. Only train and validation may be constructed.

The measured model must remain at or below 4,800,000 parameters and must
strictly improve both frozen PairGPS2D validation references: average MAE
`0.1100691929 eV` and Gap MAE `0.1318935603 eV`. A failure closes this
hypothesis without a test read. A pass freezes one model for the separately
authorized single QM9 test gate; it does not authorize PubChemQC-100K by
itself.
