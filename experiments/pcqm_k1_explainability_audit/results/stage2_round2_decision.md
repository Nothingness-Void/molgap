# K1 causal audit Round 2 decision

## Mechanical acceptance

Kunshan job `122270044` completed on one Hygon DCU with exit code `0:0`.
It performed frozen-checkpoint inference only on the 50,000-row development
role. The K1-v4 baseline reproduced within `1.14e-08 eV` MAE and
`8.58e-06 eV` maximum prediction difference. Checkpoint, payload, cache,
row count, FP32 physical batch 128, parameter count, and sealed-role flags all
matched the frozen contract.

## Finding

The trained layer-6 exchange strength is a sharp optimum, not a globally
mis-scaled residual. Every non-unit coefficient worsened overall,
topology-extreme, and middle-control MAE:

| Layer-6 coefficient | Overall delta | Topology-extreme delta | Middle delta |
|---:|---:|---:|---:|
| 0.50 | `-0.030194 eV` | `-0.036625 eV` | `-0.026203 eV` |
| 0.75 | `-0.007758 eV` | `-0.009540 eV` | `-0.006652 eV` |
| 1.25 | `-0.006605 eV` | `-0.007125 eV` | `-0.006283 eV` |
| 1.50 | `-0.023996 eV` | `-0.026061 eV` | `-0.022715 eV` |

Here a negative delta means worse than the frozen `1.00` coefficient. Every
paired 95% interval was strictly negative. The response is locally convex
around the trained value: both attenuation and amplification fail, including
for the topology group that exhibited the largest layer-6 update magnitude in
Round 1.

The Round-1 magnitude contrast therefore does not identify a scalar control
error. It is more consistent with content- and direction-dependent updates
whose magnitude is co-adapted with the downstream local blocks and layer-9
exchange. A post-hoc scalar gate cannot repair that missing selectivity.

## Decision

The causal sequence stops after Round 2. Round 3 is not spent on a trained
layer-strength gate, coefficient schedule, or another scalar rescaling. The
frozen K1-v4 architecture remains unchanged.

Any later architecture screen must pose a distinct information-flow question.
The strongest remaining evidence is the complementarity between K1's stable
rank-one molecular exchange and GPTrans Pair PreNorm's all-pair relation flow.
That hypothesis requires a separate protocol and cannot be described as a
continuation of this three-round causal audit.

