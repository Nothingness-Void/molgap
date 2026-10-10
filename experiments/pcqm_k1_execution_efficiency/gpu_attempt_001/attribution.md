# Execution attribution

The previously missing CUDA discriminator is now present. CPU layout's local
four-window signal is consistently positive, unlike its CPU-only signal. The
controlled execution mechanism moves repeated topology/shape discovery out of
device-dependent layout and shares dense indexing among three mixers, with
explicit pooling size. The experiment does not separately isolate these parts;
it cannot assign the7.03% median saving to a particular operator or synchronization.

Loss extraction, finite guards and RNG capture remain in the loop. Removing just
the explicit barrier gives mixed signs; it is not a reliable standalone fix here.
Nonblocking transfer is a separate positive local candidate, not proof of overlap
with a full pinned-memory loader or additive benefit with CPU layout.

Foreach/fused optimizer modes are much faster locally, but their low-bit state
differences mean exact interchangeability fails. This suggests investigating
optimizer execution overhead under a separately frozen numerical/quality contract;
it does not identify an accuracy defect, authorize a recipe change, or quantify
kernel-launch savings without an operator-level profile.

The baseline replay and resumed continuation pass, so no checkpoint defect is
observed in this short GPU scope. Longer-run numerical drift, full loader and graph
size effects, native T4/A100 performance, evaluation/EMA/publication cost and MAE
remain unqualified. Four short timing repeats are not independent training seeds.
No model underfit/overfit or scientific-scale conclusion follows from this test.
