# K1 causal audit Round 1 decision

## Mechanical acceptance

Kunshan job `122268096` completed on one Hygon DCU with exit code `0:0`.
It performed frozen-checkpoint inference only on the 50,000-row development
role. The frozen K1-v4 payload was reproduced to `1.14e-08 eV` MAE difference
and `8.58e-06 eV` maximum prediction difference. The checkpoint, payload,
cache, geometry, row count, physical batch 128, parameter count, and sealed-role
flags matched the frozen contract.

## Finding

No exchange is dispensable. Removing layer 3, 6, or 9 worsened overall MAE by
`0.12033`, `0.10290`, and `0.15425 eV`, respectively; removing all three
worsened it by `0.44029 eV`. The result rejects the hypothesis that one K1
exchange is simply harmful and should be deleted.

Layer 6 nevertheless has the clearest topology-dependent magnitude mismatch.
For the small/sparse/low-conjugation stratum, its update-to-hidden RMS ratio was
`0.86267`, versus `0.47186` in the middle control (`1.83x`). Its node-dispersion
ratio was `1.58592`, versus `1.09045` in the middle control. For the complete
topology-extreme union, the corresponding values were `0.60155` versus
`0.47186` and `1.22214` versus `1.09045`. This contrast is stronger than the
same update-magnitude contrast at layers 3 and 9.

Layer 9 had the largest deletion penalty, including on topology extremes. It is
therefore treated as necessary late integration rather than the first repair
target. Layer 3 also concentrates assignments on small molecules, but its
state-magnitude distortion is smaller than layer 6.

## Decision

Round 2 is released as a no-training, same-checkpoint intervention on layer 6
only. It varies the residual exchange coefficient over
`0.50, 0.75, 1.00, 1.25, 1.50`; all other operations remain bitwise identical.
The purpose is to distinguish over-strength from under-strength before any
architecture is trained. No Round-3 model is released by this decision.

The intervention supports Round 3 only if one direction improves overall and
topology-extreme MAE coherently without materially damaging the middle control.
Otherwise K1 architecture discovery pauses rather than spending the final
round blindly.
