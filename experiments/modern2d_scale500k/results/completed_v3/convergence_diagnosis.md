# Convergence diagnosis — 2026-10-27

This saved-trace analysis did not train, infer, open protected roles, alter a
frozen endpoint or submit a continuation. Numeric sources and hashes are in
[convergence_diagnosis.json](convergence_diagnosis.json); the original accepted
[terminal decision](decision.md) remains byte-unchanged.

| Family | Last499,840 presentations, Common-Dev MAE | Gain | Final LR |
|---|---|---:|---:|
|PPGT mean|0.1814968 to0.1714706|0.0100262eV|1e-6|
|ESA count|0.1376774 to0.1376542|0.0000232eV|1e-6|

PPGT completed only approximately4 physical500K passes/2M presentations.
Its last interval still improved0.0003744eV; the five-interval gain was much
larger. Both online train and development errors fell late. This did not show
an intrinsic architecture ceiling or demonstrate that a longer schedule would
close an incumbent gap. Its smaller exposure budget and expensive updates
made the completed endpoint an insufficient test of ultimate convergence.

ESA completed approximately12 physical passes/6M presentations. Its final
five-interval online training error improved about0.00414eV after rescaling,
while development improved only0.0000232eV. The last-five-observation development
range was about0.000123eV. This supports a low-marginal-return region under the
tested recipe, without proving an architecture limit or seed-level uncertainty.

Both schedules reached a1e-6 floor:1000 times below PPGT's peak and300 times
below ESA's. A frozen, tiny-learning-rate tail cannot identify whether the
remaining bottleneck is update size or representation/generalization. Continuing
the same floor indefinitely is therefore a weak diagnostic.

Recent complete evaluation intervals measured roughly751 seconds per99,968
presentations for PPGT and105 seconds for ESA. At unchanged cadence, one additional
physical500K pass would take about63 minutes and9 minutes respectively, excluding
new bootstrap/native qualification overhead. These are descriptive projections,
not equal-hardware causal speedups or a released compute budget.

Investment judgment: do not send either family to full training from this
record. PPGT has more unconverged-tail evidence but poor extension economics.
ESA is the cheaper candidate for one separately authorized, bounded learning-rate
plateau diagnostic, using the same frozen checkpoint/initial state for both
arms and an equal added-step budget. Such a new experiment must distinguish its
own schedule/optimizer-reset semantics from the completed original recipe.
It must not overwrite the original trajectory or automatically trigger another
seed, extension or architecture search. No new trial was submitted here.

The normalized online training errors were multiplied by the frozen target
standard deviation only to express their scale in eV. They are moving-weight
training-window averages, not clean fixed-weight train-set evaluation; their
gap from Common-Dev is descriptive and cannot alone prove overfitting.
