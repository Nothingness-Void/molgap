# K1 readout/selector round 1

This directory owns the first round of the separately authorized three-round
K1 architecture sequence. It compares two independent changes against the
reusable frozen K1-v4 reference:

- `neural_atom_k1_repset_readout`: a conservative RepSet node-set summary
  added at representation level after the unchanged K1 encoder;
- `neural_atom_k1_tied_selector`: one atom selector shared across K1's three
  global exchanges while value, slot processing, and return remain per-layer.

Read `evidence_selection.md` for why these questions survived the history
audit, `protocol.md` for the frozen causal comparison, and `STATUS.md` for the
remote state. Detailed metrics belong in a future dated `decision.md`.

