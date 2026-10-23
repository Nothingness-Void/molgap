# Evidence review: training-time mean correction

2026-10-24, before new model execution. Desktop RML query for BatchRenorm,
batch-renorm and renormalization found no named prior trajectory. Names alone
cannot exclude equivalent mechanisms; the owning records below define scope.

- [Frozen BN diagnostic](../pcqm_k1_bn_mechanism_a100/terminal_decision.md):
  fixed500K checkpoint changed0.104904078307 to0.103658943601eV after clean
  calibration, same weights. Dropout-on calibration was worse than dropout-off.
  This motivates testing a statistics mismatch mechanism, not assuming a
  training-time defect or a guaranteed BRN gain.
- [Norm-mechanism decision](../pcqm_k1_norm_mechanisms_100k/terminal_decision_20261023.md)
  tests normalization position/scope/gain; it does not apply detached training-
  time batch-to-running affine corrections.
- [Extended LN decision](../pcqm_k1_norm_exposure_100k/terminal_decision.md)
  establishes recipe sensitivity and constrains naive exposure-only explanations.
- [MsgNorm attribution](../pcqm_k1_message_norm_100k/attribution.md): learned
  relative neighbor amplitude improved1.364meV versus fixed, below3meV.
  It neither corrected running statistics nor established benefit over originalK1.
- Separate owner `codex/exp/k1-message-droppath-100k` asks graph-wise message
  regularization; scheduler140239597/v1 was COMPLETE at this review. Its own
  terminal acceptance remains with that owner. Do not duplicate or alter it.
- Separate draft owner `codex/exp/k1-norm-regularization-100k` tests WD1e-4
  versus1e-3 in BN/LN recipes. It is not a BRN correction test.

[Batch Renormalization, Ioffe2017](https://arxiv.org/html/1702.03275v2)
defines detached r/d corrections and gradually relaxed bounds. Its reported
results concern minibatch dependence; they are not matched PCQM4Mv2/K1 evidence.
Here both arms preserve the same variance-EMA and schedule, isolating addition
of d beyond r. Caps3/5 follow the paper's example;5+5pass ramp is an explicitly
chosen bounded screen recipe, not a reproduction or a retrospectively tuned fit.

GRED and NeuralWalker remain distinct representation routes requiring input and
runtime qualification; ESA remains a family reproduction question. This pair
uses an existing K1 representation and accepted graph cache, with one localized
operator change and a predeclared buffer diagnostic. No claim is made that this
route is intrinsically stronger than those unqualified alternatives.

Cheapest decisive test of the LEARNED training intervention is a same-job100K
pair. Frozen inference alone cannot reproduce altered training gradients. Two
new interventions avoid baseline retraining while preserving an identifiable
incremental d question. Absolute BRN-vs-BN superiority remains unresolved.
