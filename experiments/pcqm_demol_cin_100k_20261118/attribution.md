# DeMol-CIN 100K attribution

The selected-main-EMA point estimate improves4.805meV and passes the numerical
gate. Strict causal comparison remains blocked by an original cross-arm runtime
fingerprint mismatch; see decision.md. No causal positive or negative architecture
verdict is claimed.
This tests the whole pre-registered sparse atom/bond/ring cellular model
against BondLocal+PairNorm, not the isolated effect of rings or the virtual hub.
No component-specific causal attribution follows without a matched ablation.

The candidate is worse in passes1-2, then leads by pass3. Its best is pass35,
while training continues to60 without a new primary best. Final development
MAE regresses3.791meV from its selected state. Online training MAE falls from
0.086001 at selection to0.035448 at the final pass; because this is a running
metric under changing weights, it is not a fixed-weight train/dev gap estimate.
The observed pattern supports late development regression, not exposure shortage;
overfitting is a plausible explanation, not established mechanistic causation.

All four pre-defined physical Gap bins and all four target quartiles improve
at selected weights. The largest physical-bin delta is-19.724meV at Gap>=8eV
(987 rows); the smallest is-2.037meV below4eV (3725 rows). Global bias shift
is not the dominant descriptive signal in the saved-artifact diagnosis.
This weakens a single-bin-only explanation but does not identify which
cellular component caused the gain.

Auxiliary EMA0.9999 selects pass59 at0.136257468142eV, worse than the primary
by1.622247meV. It does not justify changing the frozen primary selection.
A fixed50:50 saved-prediction blend scores0.128400027286eV (descriptive);
its searched optimum is post-hoc and is not a promotion result or new contract.

The candidate process wall is1.7403x the reference. This establishes a real
100K accuracy/cost tradeoff; it does not establish500K transfer, throughput
bottlenecks or efficiency on another accelerator. The next scientific choice
must use this result and cost evidence, rather than automatically adding modules
or extending the training horizon. No successor training is authorized here.
