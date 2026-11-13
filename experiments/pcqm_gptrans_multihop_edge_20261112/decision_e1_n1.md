# E1 (100K screen) and N1 (100K init noise) decision, 2026-11-13

## Scope
Covers runs s43, s44 and n1-s45 (attempt v1) only. E2 (500K, s47/s48) is decided separately under its own frozen criteria; per protocol Amendment 2, E1 results never terminate E2 and E2 results never alter the E1 gate. This file does not close the route.

## Evidence
Development MAE in meV, role `fixed100k-next50k` (50,000 rows), best-development-EMA endpoint. d_s = candidate minus reference (negative is better). CI is a row bootstrap (2000 resamples), not seed variance.

| Arm | Dev MAE (meV) | Pair d_s (meV) | Row-bootstrap 95% CI |
|---|---|---|---|
| ref_s43 | 138.51349 | | |
| mh_s43 | 139.17193 | d_43 = +0.65845 | [-0.24686, +1.54055] |
| ref_s44 | 139.28710 | | |
| mh_s44 | 138.62045 | d_44 = -0.66666 | [-1.52093, +0.25883] |
| ref_s45 | 140.00867 | | |
| ref_s46 | 142.21825 | | |

mean(d_43, d_44) = -0.0041 meV.

## E1 ruling
Frozen gate "mean(d_43, d_44) <= -3.7 meV and both d_s < 0" fails on both conditions (mean is about 0.0 meV; d_43 > 0).
- mh_s43: NEGATIVE_UNDER_CONTRACT
- mh_s44: NEGATIVE_UNDER_CONTRACT
- ref_s43: CLOSED
- ref_s44: CLOSED

## N1 ruling
D = max - min over the four reference arms = 142.21825 - 138.51349 = 3.70476 meV > 1.85 meV, so the N1 hypothesis (D <= 1.85) is rejected. Four-arm sample sd is 1.596 meV; the frozen criterion is the range D, not the sd.
- ref_s45: CLOSED
- ref_s46: CLOSED
Both are reference arms; there is no same-run candidate comparison. Cross-job runtime differences are not controlled.

## Confidence note
Because D > 1.85 meV, two E1 pairs lack power to resolve a 3.7 meV effect. The E1 NEGATIVE_UNDER_CONTRACT outcome stands under the frozen gate (N1 never changes the E1 gate or outcome) but carries this confidence note. The paired differences (+0.66, -0.66) have opposite signs and are smaller than the reference init range.

## Terminal attribution
failure_mode: `below_gate`. The same-job paired effect is indistinguishable from zero at 100K. The alternative explanation recorded in the protocol (BondLocal plus 12 pair-propagation layers and the distance table already recover path content at this scale) is consistent with, but not established by, this evidence. Whether the mechanism helps at 500K is `insufficient_evidence` here and is left to E2. No claim of module harm, underfitting or overfitting. Cost: see cost records (not restated here).

## Roles
Official validation, test_dev and test_challenge were not read. The development role is selection-used, not held out.
