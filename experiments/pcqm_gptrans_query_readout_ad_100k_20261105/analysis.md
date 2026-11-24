# Attribution — reduced propagation with dynamic readout

The8+4 construction did not preserve the12-layer parent's selected accuracy.
This is a complete60-pass endpoint regression, not a provisioning or unfinished
training result. All50K retained prediction rows are aligned by the shared
diagnostic. The change jointly removes later BondLocal/all-atom updates and
replaces them with query-only evolution; it does not isolate one attention operator.

The shared diagnosis reports late-horizon fade and no dominant global-bias-shift
explanation. D selects pass53, while A selects42. Later selection alone does not
show that more exposure would recover the lost quality. Online live training
metrics and selected EMA development metrics have different weight and timing
semantics, so their difference cannot identify causal overfitting or underfitting.

CPU structural verification showed differentiable split memory and actual
single-row attention. Those checks establish implementation properties, not that
the omitted per-atom computation is unnecessary for molecular prediction.
The strongest supported conclusion is specific: this frozen8+4 recipe loses
1.493meV relative to full12, despite its slightly shorter process duration.

The canonical trace has no per-pass wall measurements; process totals cannot
be relabeled as median mature-epoch timing. No inference-latency conclusion is
available from the synthetic CPU smoke timing or native qualification profiles.
Quality already fails the conjunctive frozen gate, so missing mature timing is
not grounds for another run solely to complete attribution. No new model/data
execution was performed during acceptance. Protected-role use is determined by
the generated audit from retained data-open provenance, not inferred from authority.

## Matched-exposure development curve

Values are EMA Gap MAE in eV from the retained canonical traces.

| Completed pass | A full12 | D full8_query4 |
|---:|---:|---:|
| 10 | 0.17334341 | 0.17339824 |
| 20 | 0.15037288 | 0.14974333 |
| 30 | 0.14207532 | 0.14328457 |
| 40 | 0.13962667 | 0.14142196 |
| 50 | 0.13972734 | 0.14102332 |
| 60 | 0.13986689 | 0.14112826 |

D briefly leads at pass20 but trails at all listed observations from30 onward.
Its final normalized online-live training MAE is0.05587999 versus A0.05525093;
the retained evidence does not show a dominant training-fit improvement being
lost only at evaluation. These online metrics are not fixed-weight training
errors and cannot determine why omitted all-atom propagation hurts.
