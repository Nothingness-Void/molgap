# Optimizer attribution

The same-job pair changes the declared hidden-matrix optimization recipe and
keeps the architecture, initialization, roles, physical batch, single forward,
precision, objective, schedule horizon and selected Live policy fixed. The
baseline reproduces the retained original scalar to displayed precision.
The result supports this fixed FP32 Muon/AdamW package under100K; it does not
isolate orthogonalization from momentum, update scaling or their interactions.

| Pass | AdamW MAE/eV | Muon MAE/eV | Muon minus AdamW/meV |
| --- | ---: | ---: | ---: |
| 5 | .231598794 | .183393553 | -48.205242 |
| 10 | .188106254 | .159993559 | -28.112695 |
| 20 | .160280138 | .143160775 | -17.119363 |
| 30 | .145307466 | .134460106 | -10.847360 |
| 40 | .141294464 | .132520661 | -8.773804 |

Muon learns faster per presentation early, and a material difference remains
after complete cosine decay. Its relative advantage narrows late, but it does
not vanish or reverse. Both select their final pass; this is not evidence that
another extension must help. Normalized stochastic TRAIN quantities and Live
development eV MAE are different metrics; no clean-fit generalization gap,
capacity ceiling or unique causal diagnosis follows from subtracting them.

The existing saved-artifact diagnosis reports no dominant global bias shift.
Its label-based strata and oracle blend are post-hoc descriptions, not new
deployable models, promotion metrics or router inputs. Primary selected single-
model scores and the prospective gate remain unchanged.

Muon costs about2.03x baseline process wall at equal exposure. Canonical traces
retain no per-pass timing values, so no mature-epoch median or isolated optimizer
overhead is asserted. Whole-worker cost includes all training/evaluation/IO;
the result does not establish equal-compute superiority or isolate the cost of
Newton-Schulz from hardware realization. Native T4 allocation cost remains
separate from process wall and from unmeasured queue/CPU time.

The useful conclusion is a substantial100K single-model accuracy gain with a
substantial training-cost tradeoff. A complete matched500K comparison would test
portability; no500K performance, full-scale release or automatic successor is
implied by this evidence.

Evidence owners: closure_v1/measured_pcqm-k1-muon-100k-20261106-s42.json,
the receipt-pinned per-arm canonical_trace.json under D:/molgap-artifacts/retrieved/
pcqm-k1-muon-100k-20261106-s42/v1/version1/, and that directory's _execution
allocation ledger/report. Shared acceptance owns final strict qualification.
