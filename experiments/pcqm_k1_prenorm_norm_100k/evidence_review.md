# Evidence review

Retained Kaggle2 norm screen: Post-BN Live0.141294461eV/pass40;
Post-Ada0.145184176/pass39; candidate-minus-control3.889715meV with paired-row95%
interval[2.827877,4.949146]. The exact substitution failed. Its zero-dropout and
proven-overfitting interpretation was corrected: actual dropout0.05, online
train metrics, development still improving late, and no retained EMA dev curve.
Canonical old outcome is preserved. Exact local copies, Git source identities
and SHA pins are in `reference_binding/post_norm/manifest.json`.

[PPGT equations3/7](https://arxiv.org/html/2504.12588v4) describes adaptive gain
and pre-norm residuals. The old experiment only replaced local post-residual
norms; its global mixer LayerNorms were unchanged. This leaves a new placement
question. No claim that a newer norm must win is made.

The completed500K same-checkpoint state diagnostic separately observed
prefix-Clean0.109049479,RawEMA0.105396973 and uniform-Clean0.105419359eV.
It supports BN output-state sensitivity, not a unique training cause or a
general mandatory-BN policy. This run does not repeat that diagnostic or scale up.

Desktop RML was queried before design; accepted K1 dropout/consistency/BN and
cost studies do not answer this local norm-by-placement interaction. The two
new cells hold the remaining recipe fixed. The retained controls are contextual
for placement; the prospective same-job Pre-BN/Pre-Ada pair owns the primary
normalization comparison. Failed gates are not repaired by tuning after results.
