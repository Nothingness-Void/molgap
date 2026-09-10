# Neural-Atom mixer evidence selection

## Decision context

The cardinality channel failed for a specific reason: unnormalized local source
content was partly useful relative to a size shortcut, but did not add enough
information beyond nine EdgeState/GPS blocks and RWSE16. The next experiment
must therefore change the allocation of global communication rather than add
another local support statistic.

Repository evidence narrows the question:

- full atom-to-atom GPS attention is not consistently efficient;
- a single shared GraphState was strong at 100K but did not remain the accepted
  full-scale incumbent, suggesting that one global vector may be too low-rank;
- the closed R7 graph-token experiment retained every GPS attention block and
  broadcast one pooled vector, so it did not test replacement by heterogeneous
  latent communication;
- path attention, ring/fragment states, PNA, directed bonds, geometry,
  pretraining, residuals, fusion, and cardinality channels are already closed
  or out of scope.

## External mechanism evidence

Neural Atoms (ICLR 2024) implements a small learned latent set between atoms:
learned queries pool node states, latent self-attention exchanges information,
and the same atom-to-latent assignment projects distinct mixtures back to
atoms. The official 2D implementation was inspected at repository commit
`6be4cfe63ad16c3d0bf5c37a55704da17feb8f3d`; this is implementation evidence,
not a claim of PCQM transfer.

- Paper: <https://arxiv.org/abs/2311.01276>
- Official implementation: <https://github.com/tmlr-group/NeuralAtom>

## Selected question

Replace dense atom-to-atom global attention with a four-slot, low-rank
Neural-Atom mixer after local EdgeState blocks 3, 6, and 9. Compare it with:

1. a fresh unchanged full-GPS EdgeState baseline; and
2. a parameter-identical one-active-slot mixer.

All four slot parameters exist in both mixer arms; masking alone changes the
active slot count. Thus four slots beating one slot supports heterogeneous
global channels rather than merely adding a virtual graph state or parameters.

This is attempt 1 of the three-route post-cardinality budget. A scientific loss
closes slot-count/placement/width/seed/schedule variants of this exact mechanism
before any successor is considered.
