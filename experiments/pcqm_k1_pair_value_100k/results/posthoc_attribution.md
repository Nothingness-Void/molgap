# K1 PairToken value-decoupling trace audit

The [terminal decision](../decision.md) owns the observed failed point gates
and `INCONCLUSIVE` strict RML qualification. This local audit reads only its
accepted 40-observation `canonical_trace_v2.json`; it does not load a
checkpoint or consume another evaluation role.

The candidate selected the last epoch, 39, at 0.13896267 eV. Development MAE
fell by 2.466 meV from epoch 29 to 39 while fixed-100K live training MAE in
normalized target units fell from 0.082456 to 0.074340. These metrics have
different units and roles. The last recorded learning rate was about
1.615e-6. Thus the candidate was still improving at its frozen endpoint,
and the trace alone cannot rule out a training-horizon limitation. It also
cannot show that extra exposure would clear either predeclared point gate:
the accepted result missed the K1-v4 3 meV gain gate and was 0.633 meV
worse than original PairToken.

There is no locally accepted, row-aligned K1-v4 prediction payload for this
candidate's required paired comparison, and deterministic checkpoint resume
was not replayed. Without the immutable reference trajectory and paired
per-row errors, the audit cannot separate the value-decoupling mechanism
from training variation, calibration, or subgroup tradeoffs. The cheapest
next evidence is recovery and hash acceptance of the *original* reference
payload if it still exists, plus the contracted resume check. Retraining a
reference solely to fill this gap would change the comparison and is not
authorized by this audit.
