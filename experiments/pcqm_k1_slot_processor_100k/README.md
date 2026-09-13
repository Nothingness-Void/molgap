# K1 single-slot processor round 2

This experiment is round 2 of the bounded K1 architecture sequence. It asks
whether K1's self-attention over one latent token contains unnecessary
parameterization. Read `protocol.md` for the frozen contract,
`evidence_selection.md` for the causal basis, and `STATUS.md` for operations.

The two isolated candidates are:

- `neural_atom_k1_collapsed_mha`: retain only the value and output maps that
  can affect length-one self-attention;
- `neural_atom_k1_no_slot_attention`: remove that attention residual while
  retaining atom selection, the slot FFN, and the slot-to-node return.

Neither candidate changes the local EdgeState path, K1 exchange locations,
data, target, training contract, or frozen reference.

