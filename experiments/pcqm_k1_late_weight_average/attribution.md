# Attribution — 2026-10-08

**Supported:** selected49 learned weights outperform final60 learned weights
after identically configured training-member BN adaptation on full50K development.
Late raw deterioration therefore is not explained solely by stale BN. Equal
two-endpoint parameter or prediction averaging fails to improve selected-clean.
Keeping selected BN on averaged weights causes an additional state mismatch;
clean BN removes much of that penalty. Endpoint mean is not stepwise EMA.

**Unidentified:** which optimization/regularization cause produced worse final
weights; true EMA benefit; capacity saturation; causal pretraining contribution;
training-time BN/dropout harm. The train prefixes and extension retained in prior
diagnostics have different sample uncertainties, so this full development result
does not independently prove underfitting or overfitting. Native GPTrans has
different weights/recipe, so its1.771meV advantage is contextual, not an isolated
architecture effect.

**Disposition:** close this exact sparse average, preserve selected49 clean
state as diagnostic reference, and use the existing500K consistency question
before a new structural module. Row bootstrap measures molecule uncertainty,
not training-seed variance. Consumed development is not independent release.
