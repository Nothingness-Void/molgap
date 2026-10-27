# Fixed prediction fusion on internal 500K development

Prospective CPU-only diagnostic requested 2026-10-27. On the same historically
selection-used 50K development rows, compare retained K1 clean-BN predictions,
GPTrans BondLocal+PairNorm learning-rate 1e-3 and 5e-4 predictions, their fixed
50:50 pairs, and their fixed equal three-model mean. Compare with the retained
historical K1+GPTrans mean as context. Compute the pair and three-model oracle
as the per-row minimum absolute error, using labels only as an unattainable
upper bound, never as a deployable router. No fitted weights, training, inference,
official validation or test. Verify retained file hashes, exact row IDs, labels,
and finite values before computing. The question is whether K1 complements the
new GPTrans and whether it improves on the two-GPTrans mean. Results are
exploratory and cannot establish independent generalization or full-scale gain.
