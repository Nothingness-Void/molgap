# K1 / PairToken scale attribution

This experiment asks why PairToken's matched 100K gain did not retain its
magnitude at 500K.  It is an attribution sequence, not a new architecture
search.

Round 1 trains the missing K1 reference under the exact matched60-v4 500K
contract already used by PairToken.  Rounds 2 and 3 are released only after
Round 1 terminal acceptance: aligned residual analysis and frozen-checkpoint
PairToken intervention, respectively.

Read `protocol.md` for the frozen sequence and `STATUS.md` for live state.
