# ESGPS6-304 at 500K

This experiment compares scratch training with local atom/bond/angle relation
pretraining for a six-layer, 304-channel persistent EdgeState GPS. It replaces
the cancelled nine-layer width-304 run while retaining the same accepted data,
batch, seed, optimizer, target, and equal encoder-exposure rule.

The immutable protocol is [protocol.md](protocol.md). Remote artifacts use a
new dedicated result root and never overwrite the cancelled run.
