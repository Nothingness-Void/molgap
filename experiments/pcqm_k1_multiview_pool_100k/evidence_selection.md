# Evidence selection

## Predecessor result

Attempt 1 replaced K1's one atom distribution with four paper-style latent
atom groups. It had identical size and initialization, fit the training role
more tightly, and generalized worse. Its paired bootstrap interval was entirely
unfavorable. The decision closes more global groups, not all richer allocation.

## Remaining uncertainty

K1's useful prior is a small number of global exchanges through one compact
molecular representation. Its single 64-channel query, however, requires all
channels to select the same atom distribution. Different channel subspaces may
need to summarize different local chemistry while still producing only one
molecular token.

Attempt 2 therefore reshapes the existing 64-channel key, value, and query into
four 16-channel heads. Each head normalizes over original atoms, the four pooled
views concatenate into one 64-channel token, and each head reuses its transpose
for return. Slot count, parameters, exchange layers, local EdgeState path,
training contract, and prediction head remain unchanged.

This is not a rescue of the failed four-slot grouping: atoms do not choose among
competing global groups, and no extra global token is introduced.
