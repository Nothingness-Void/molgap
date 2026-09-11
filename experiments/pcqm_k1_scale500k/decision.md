# K1 fixed-data 500K scale decision

Decision date: 2026-09-11

## Question

Does the frozen one-slot Neural-Atom K1 architecture preserve its PCQM-100K
advantage over fresh Full-GPS after scaling to the accepted cross-platform
500K/50K dataset?

## Acceptance

Kaggle2 kernel `kaseichou/molgap-pcqm-k1-scale500k-s42` version 3 completed
both 40-epoch arms and passed the frozen no-inference acceptance. Both arms
used source commit `36215d9539acdd75542608637ec1e2db5341d3ff`, seed 42,
FP32, physical batch 128, fused AdamW, and the same cosine schedule. The fixed
manifest SHA-256 was
`630d30046d6cdc1f91fb169cd1eb4720bd5b352dc1ebeb641a11ead1ebae9751`;
its graph payload is byte-identical across Kaggle1, Kaggle2, SCNet, and IMS.
Official validation, test-dev, and the consumed shadow role remained unread.

## Result

| Arm | Development Gap MAE | Parameters | Mean epoch | Derived throughput |
|---|---:|---:|---:|---:|
| Full-GPS | 0.1146809459 eV | 4,771,073 | 454.046 s | 1,101.21 graphs/s |
| Neural-Atom K1 | 0.1048590988 eV | 3,658,817 | 394.272 s | 1,268.16 graphs/s |

K1 improved MAE by `0.0098218471 eV` (`8.56%` relative). The paired bootstrap
95% interval for K1 minus Full-GPS was
`[-0.0105039993, -0.0091549593] eV`, entirely favorable and far beyond the
predeclared `0.001 eV` gate. K1 used `23.31%` fewer parameters, reduced mean
epoch time by `13.16%`, and increased paired-task throughput by `15.16%`.

The scale result agrees with the earlier PCQM-100K paired gain
(`0.009461689 eV`) and the independent shadow gain (`0.006076947 eV`). The
500K advantage retained about `103.8%` of the 100K paired effect rather than
decaying with scale. K1 selected epoch 38; epoch 39 regressed after the learning
rate had fallen below `2e-6`, so the frozen 40-epoch screen supplies adequate
convergence evidence for this decision.

## Attribution

K1 removes nine dense atom-to-atom global-attention operations and retains
local persistent real-bond EdgeState processing. Three layers exchange
information through one shared 64-channel latent molecular slot. The repeated
gain, lower training error gap, smaller parameter count, and higher throughput
support global-allocation simplification as the causal mechanism: bounded PCQM
benefits from sparse molecule-level communication without repeated dense
all-atom attention.

This does not establish superiority over a separately configured 304-wide or
pretrained desktop model. Cross-platform data identity now permits contextual
comparison, but a causal architecture claim still requires matched training
contracts.

## Decision

K1 passes the frozen 500K scale gate and is the sole server-side architecture
candidate eligible for a desktop full-scale budget decision. Freeze the K1
architecture and hashes unchanged. Do not tune slots, width, placement,
optimizer, schedule, or seed on the consumed roles; do not run seeds 43/44;
do not submit another server GPU experiment or read official validation/test.

Desktop may independently approve one full-scale run under its own frozen
protocol and compute budget. Its schedule must be defined in optimizer steps or
sample exposure rather than copying 40 epochs onto the 6.76-times larger full
training role. Passing this bridge is not production promotion and is not an
official leaderboard result.
