# Evidence selection

## Joint attribution from attempts 1 and 2

Four paper-style latent groups and four channel-wise selection heads both fit
the training role more tightly than K1 and generalized worse. K1's useful
regularization therefore appears to require one shared atom distribution and
one compact molecular token. More slots and more simultaneous distributions
are closed.

## Remaining question

K1 asks the same learned 64-channel query of every molecule at each exchange
layer. K1-G showed that scaling an update after this selection is too late;
K1-R showed that adding a second relation representation is unnecessary. The
unresolved low-complexity question is whether the sole query should adapt to
the molecule before selecting atoms.

Attempt 3 adds one zero-initialized 192-to-64 projection at each of layers 3,
6, and 9. It maps the mean current node state into a query offset. There remains
exactly one atom distribution, one 64-channel token, and the same transposed
return. This adds 36,864 parameters, about 1.0% over K1, without changing the
local EdgeState path or global communication rank.
