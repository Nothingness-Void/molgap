# Full K1 and GPTrans-T Fusion Decision, 2026-09-15

## Accepted execution

The IMS R3 chain completed mechanically. K1 reached the frozen 156,250
optimizer steps and 20,000,000 sample presentations. The separate fusion
acceptance verified 73,545 aligned official-validation rows, the frozen
14,933/58,612 calibration/holdout split, model and prediction hashes, finite
metrics, and sealed test-dev/challenge roles.

## Results

| Model | Evaluation role | Gap MAE (eV) |
|---|---|---:|
| EdgeState convergence reference | full official validation | **0.099638** |
| K1 | full official validation | 0.106672 |
| GPTrans-T | full official validation | 0.109012 |
| K1/GPTrans-T 50:50 | full official validation | 0.102227 |
| K1 | frozen four-fifths holdout | 0.106719 |
| GPTrans-T | frozen four-fifths holdout | 0.109095 |
| K1/GPTrans-T 50:50 | frozen four-fifths holdout | 0.102260 |
| K1/GPTrans-T calibrated 55.2:44.8 | frozen four-fifths holdout | **0.102186** |

K1 beat GPTrans-T by 0.002340 eV on all official-validation rows. The fixed
blend improved over K1 by 0.004445 eV, establishing useful prediction-error
complementarity. On the untouched holdout, calibration improved over K1 by
0.004533 eV and over GPTrans-T by 0.006909 eV.

The fitted coefficient improved over the fixed 50:50 holdout blend by only
0.000074 eV. The useful effect is therefore model averaging, not evidence for
a valuable learned scalar router. The fixed full-validation blend remained
0.002588 eV worse than the accepted EdgeState convergence reference on the
same 73,545-row role.

## Decision

Accept the artifacts and the complementarity result, but reject this fusion as
a Track B advancement candidate. Do not promote K1, GPTrans-T, or their blend;
do not change the production registry or open test-dev/challenge. The one-time
official-validation calibration role is consumed and must not be reused for
coefficient tuning. Any later fusion involving a materially changed checkpoint
requires a separately frozen question and evaluation protocol.

Machine-readable evidence is in `k1/`, `fusion/`, and `logs/`. The accepted K1
bundle SHA-256 is
`60b95d86736829fd38690fb33654199a753f693a333aa55f451d1983653fa3b7`;
the fusion prediction SHA-256 is
`b580d8295f159d7efc284d6bdb143b05452b4ce1c05bfac6a869ceba2ca7234b`.
