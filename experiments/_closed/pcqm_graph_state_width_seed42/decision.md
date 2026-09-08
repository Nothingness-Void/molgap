# GraphState Width Seed-42 Decision

## Scope

On 2026-09-08, a paired PCQM-100K screen compared the accepted GraphState64
encoder with an otherwise identical GraphState128 encoder. Both candidates
used the same seed-42 initialization contract, immutable 100K/10K data roles,
ETKDGv3/MMFF94s cache, optimizer, schedule, and direct Gap target.

Kaggle repeatedly ignored the requested T4x2 shape. The protocol-authorized
fallback ran one candidate per isolated kernel on two matching
`Tesla P100-PCIE-16GB` devices. The corrected mechanical acceptance passed all
source, cache, row, target, initialization, finite-value, checkpoint, payload,
hash, accelerator-identity, and sealed-role checks.

## Result

| Candidate | Parameters | Best epoch | Validation Gap MAE | Throughput |
|---|---:|---:|---:|---:|
| GraphState64 | 3,665,809 | 35 | 0.1301066 eV | 602.25 graphs/s |
| GraphState128 | 3,803,985 | 38 | 0.1305125 eV | 561.57 graphs/s |

GraphState128 regressed by `0.0004059 eV`. The paired bootstrap 95% interval
for candidate-minus-baseline error was `[-0.0009570, 0.0016199] eV`, which
crossed zero. It retained `0.9325x` baseline throughput, so the compute gate
passed, but the required `0.001 eV` material accuracy gain failed.

The wider state added 138,176 parameters (`3.77%`), reduced throughput by
`6.75%`, and required a later best epoch without improving validation error.
The evidence therefore did not support GraphState width as the limiting
bottleneck under this architecture and training contract.

## Decision

The GraphState64-to-128 width increase was rejected and closed at seed 42.
Seeds 43/44, 1M bridging, full-scale training, official validation, and
test-dev were not authorized. GraphState64 remained the comparison anchor;
this result did not change the production registry or the Track B delivery
candidate.

Mechanical acceptance:
[`mechanical_acceptance.json`](results/mechanical_acceptance.json), SHA-256
`00bfd88f917f93d60117ea89c5134e2c053d9b39b3859943423d3e719a8d4fef`.

Terminal log SHA-256:

- GraphState64: `de1984e7434f39b60006aaded18c1fa0b19c5ea6ff4d4219fea721162ef3543d`
- GraphState128: `9a511ab4478b41eda81819f84c4df3e42120c20875642ef62a9ce237f9913986`
